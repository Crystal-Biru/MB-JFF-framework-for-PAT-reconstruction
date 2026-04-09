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

        self.ssim_loss = SSIMLoss()  # 添加SSIM损失

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

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.register_buffer('window', self.create_window(window_size))

    def create_window(self, window_size, sigma=1.5):
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        _1D_window = g.unsqueeze(1)
        _2D_window = _1D_window @ _1D_window.t()
        window = _2D_window.unsqueeze(0).unsqueeze(0)
        return window

    def _normalize(self, x, minmax):
        if isinstance(minmax, np.ndarray):
            minmax = torch.from_numpy(minmax).to(x.device).to(x.dtype)
        elif not torch.is_tensor(minmax):
            minmax = torch.tensor(minmax, device=x.device, dtype=x.dtype)

        if minmax.ndim == 1:
            min_val = minmax[0]
            max_val = minmax[1]
            denom = max(max_val - min_val, 1e-6)
            x = (x - min_val) / denom
        else:
            min_val = minmax[:, 0].view(-1, 1, 1, 1)
            max_val = minmax[:, 1].view(-1, 1, 1, 1)
            denom = torch.clamp(max_val - min_val, min=1e-6)
            x = (x - min_val) / denom
        return x

    def ssim(self, pred, gt, window, size_average=True):
        padding = self.window_size // 2
        mu1 = F.conv2d(pred, window, padding=padding, groups=1)
        mu2 = F.conv2d(gt, window, padding=padding, groups=1)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=padding, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(gt * gt, window, padding=padding, groups=1) - mu2_sq
        sigma12 = F.conv2d(pred * gt, window, padding=padding, groups=1) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        if size_average:
            return ssim_map.mean()
        return ssim_map.mean([1, 2, 3])

    def forward(self, pred, gt, minmax=[0, 1]):
        pred = self._normalize(pred, minmax)
        gt = self._normalize(gt, minmax)
        ssim_val = self.ssim(pred, gt, self.window, size_average=self.size_average)
        return 1 - ssim_val
