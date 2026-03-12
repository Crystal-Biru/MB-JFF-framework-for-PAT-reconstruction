from pyexpat import model
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torchvision.models as models
import numpy as np
import pywt
from utils import calc_ssim
import os
from models.DASandLUT import DASAndPixelInterpolator, DASAndPixelInterpolator_MSOT


class mse(_Loss):
    def __init__(self, args, **kwargs):
        super(mse, self).__init__()
        self.args = args

    def forward(self, pred, gt, **kwargs):

        return {'mse_loss':F.mse_loss(pred,gt)}


class adjoint(_Loss):
    def __init__(self, args, **kwargs):
        super(adjoint, self).__init__()
        self.args = args

        self.ssim_loss = SSIM()  # 添加SSIM损失

    def forward(self, pred, gt, minmax=np.array([0, 255]), **kwargs):

        mse_l = F.mse_loss(pred, gt)
        ssim_l = self.ssim_loss(pred, gt, minmax=minmax)

        return {'mse_loss':mse_l, 'ssim_loss':ssim_l}


class forward(_Loss):
    def __init__(self, args, **kwargs):
        super(forward, self).__init__()
        self.args = args

        if args.channel_num == 128:
            self.das_recon = DASAndPixelInterpolator()
        else:
            self.das_recon = DASAndPixelInterpolator_MSOT()
        

    def forward(self, pred, gt, input=None, **kwargs):
        mse_loss = F.mse_loss(pred, gt)

        das_pred, lut_pred = self.das_recon(pred, output_type='both')
        das_gt, lut_gt = self.das_recon(gt, output_type='both')
        # 计算 DAS 重建的 MSE 损失
        das_mse_loss = F.mse_loss(das_pred, das_gt)
        LUT_mse_loss = F.mse_loss(lut_pred, lut_gt)

        return {
            'mse_loss': mse_loss,
            'das_loss': das_mse_loss,
            'lut_loss': LUT_mse_loss
        }

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, pred, gt, minmax=[0, 1]):
        batch_size = pred.size(0)
        ssim_vals = []
        for idx in range(batch_size):
            ssim = calc_ssim(pred[idx, 0].detach().cpu().numpy(), 
                             gt[idx, 0].detach().cpu().numpy(), minmax[idx])
            ssim_vals.append(ssim)
        ssim_vals = np.array(ssim_vals)
        return 1 - ssim_vals.mean()
