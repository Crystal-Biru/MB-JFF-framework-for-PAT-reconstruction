import torch
import argparse
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
import logging
import os
import time
from tqdm import tqdm
import random
from utils import AverageMeter,ConsoleLogger,calc_rmse,calc_psnr,calc_ssim,save_results,InputPadder
from loss import *
import lpips
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

from dataset_PACT import dataset_PACT
from models.FD_UNet import U_Net, FD_UNet, FD_UNet_CBAM
from models.srgan import SRGAN_Generator, SRGAN_Discriminator
from neuraloperator.neuralop.models import FNO
from models.forward_operator import Forward_Network
from models.DASandLUT import DASAndPixelInterpolator, DASAndPixelInterpolator_MSOT
import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def create_code_snapshot(root, dst_path, extensions=(".py", ".json"), exclude=()):
    """Creates tarball with the source code"""
    import tarfile
    from pathlib import Path

    with tarfile.open(str(dst_path), "w:gz") as tar:
        tar.add(root, arcname='code', recursive=True)

def get_args():
    parser = argparse.ArgumentParser()

    # =========for hyper parameters===
    parser.add_argument('--gpu', type=str, default='0', help='GPU indices to use (e.g., "0,1")')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--trial',type=str, default='baseline')
    parser.add_argument('--mode', type=str, default='train',choices=['train','test','test_offline'])
    parser.add_argument('--seed',type=int,default=1)

    # ==========define the task==============
    parser.add_argument('--channel_num', type=int, default=128)  # 128 for Linear-Data, 64 for MSOT-Data
    parser.add_argument('--time_step_num', type=int, default=4096)  # 4096 for Linear-Data, 2030 for MSOT-Data
    parser.add_argument('--p0_pixel', type=int, default=256)  # p0 image size

    # =========for training===========
    # 修改：为了实现gradient accumulation，减小实际的batch size
    parser.add_argument('--train_batchsize', type=int, default=2)  # 减少实际batch size
    parser.add_argument('--val_batchsize', type=int, default=2)   
    parser.add_argument('--effective_batch_size', type=int, default=16)  # 新增：目标的effective batch size
    parser.add_argument('--max_epoch', default=300, help='max training epoch', type=int)

    parser.add_argument('--lr', default=5e-4, help='learning rate', type=float)
    parser.add_argument('--weight_decay', default=0.0001, help='decay of learning rate', type=float)

    parser.add_argument('--freq_print_train', default=20, help='Printing frequency for training', type=int)
    parser.add_argument('--freq_print_val', default=20, help='Printing frequency for validation', type=int)
    parser.add_argument('--freq_print_test', default=50, help='Printing frequency for test', type=int)
    parser.add_argument('--load_model', type=str, default='')

    # ==========loss function============
    parser.add_argument('--loss_type',type=str,default='forward', 
                        choices=['mse','forward'])
    parser.add_argument('--w_pxl',type=float,default=1.0)
    parser.add_argument('--w_per',type=float,default=0.3)
    parser.add_argument('--w_ssim',type=float,default=0.5)

    # ========for model ==============
    parser.add_argument('--model_type', type=str, default='Forward', 
                        choices=['Forward'])
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--inner_channel', type=int, default=64)
    
    # ========for saving results==============
    parser.add_argument('--save_epoch',type=int, default=1)
    parser.add_argument('--save_sinogram', action='store_true', help='Save sinogram outputs during testing')
    parser.add_argument('--save_dir', type=str, default='./results_forward_Linear', help='Directory to save sinogram results')


    # parse configs
    args = parser.parse_args()
    return args

def setup_multi_gpu(args):
    """Setup multi-GPU configuration"""
    if args.multi_gpu and torch.cuda.device_count() > 1:
        gpu_ids = [int(x) for x in args.gpu.split(',')]
        print(f"Using GPUs: {gpu_ids}")
        return gpu_ids
    else:
        return [int(args.gpu.split(',')[0])]

def save_state(model, optimizer, LOGGER, logdir, name='best_perf'):
    checkpoint_dir = os.path.join(logdir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    LOGGER.info('=> saving checkpoint to {}'.format(checkpoint_dir))
    states = dict()
    
    # 如果是DataParallel模型，需要保存module的state_dict
    if hasattr(model, 'module'):
        states['model_state_dict'] = model.module.state_dict()
    else:
        states['model_state_dict'] = model.state_dict()
    
    states['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(states, os.path.join(checkpoint_dir, name+'.tar'))

def validate(val_dataloader, LOGGER, device, model, das_recon_model,
             loss_val_log, loss_val_psnr, loss_val_ssim, loss_val_das, args):

    with torch.no_grad():
        model.eval()
        for it, batch in enumerate(val_dataloader, 0):
            input = batch['p0'].to(torch.float32).cuda(device)
            gt = batch['sinogram'].to(torch.float32).cuda(device)
            batch_size = len(input)
            minmax = np.array([batch['d_min'].numpy(), batch['d_max'].numpy()]).T
            
            output = model(input)
            das_recon = das_recon_model(output, output_type='das_only')
            fbp = das_recon_model(gt, output_type='das_only')
            
            for idx in range(len(output)):
                rmse = calc_rmse(output[idx, 0].cpu().numpy(), gt[idx, 0].cpu().numpy())
                loss_val_log.update(rmse, 1)
                psnr = calc_psnr(das_recon[idx, 0].cpu().numpy(), fbp[idx, 0].cpu().numpy(), minmax=minmax[idx])
                loss_val_psnr.update(psnr, 1)
                ssim = calc_ssim(das_recon[idx, 0].cpu().numpy(), fbp[idx, 0].cpu().numpy(), minmax=minmax[idx])
                loss_val_ssim.update(ssim, 1)
                das_rmse = calc_rmse(das_recon[idx, 0].cpu().numpy(), fbp[idx, 0].cpu().numpy(), minmax=minmax[idx])
                loss_val_das.update(das_rmse, 1)

        message = 'RMSE_val {loss1.ave:.5f} || DAS_test {loss4.ave:.5f} || PSNR {loss2.ave:.5f} || SSIM {loss3.ave:.5f}\t'.format(
            loss1=loss_val_log, loss2=loss_val_psnr, loss3=loss_val_ssim, loss4=loss_val_das)
        LOGGER.info(message)

def test(test_dataloader, LOGGER, device, model, das_recon_model, args):
    rmse_log = AverageMeter()
    das_log = AverageMeter()
    psnr_log = AverageMeter()
    ssim_log = AverageMeter()

    with torch.no_grad():
        model.eval()
        sample_count = 0

        for it, batch in enumerate(test_dataloader, 0):
            input = batch['p0'].to(torch.float32).cuda(device)
            gt = batch['sinogram'].to(torch.float32).cuda(device)
            batch_size = len(input)
            minmax = np.array([batch['d_min'].numpy(), batch['d_max'].numpy()]).T
            
            output = model(input)
            das_pred = das_recon_model(output, output_type='das_only')
            fbp = das_recon_model(gt, output_type='das_only')
            
            for idx in range(len(output)):
                rmse = calc_rmse(output[idx, 0].cpu().numpy(), gt[idx, 0].cpu().numpy())
                rmse_log.update(rmse, 1)
                das_rmse = calc_rmse(das_pred[idx, 0].cpu().numpy(), fbp[idx, 0].cpu().numpy(), minmax=minmax[idx])
                das_log.update(das_rmse, 1)
                psnr = calc_psnr(das_pred[idx, 0].cpu().numpy(), fbp[idx, 0].cpu().numpy(), minmax=minmax[idx])
                psnr_log.update(psnr, 1) if psnr < float('inf') else None
                ssim = calc_ssim(das_pred[idx, 0].cpu().numpy(), fbp[idx, 0].cpu().numpy(), minmax=minmax[idx])
                ssim_log.update(ssim, 1)

        message1 = 'RMSE_test: {loss1.ave:.5f} || DAS_test: {loss4.ave:.5f} || PSNR_test: {loss2.ave:.5f} || SSIM_test: {loss3.ave:.5f}'.format(
            loss1=rmse_log, loss2=psnr_log, loss3=ssim_log, loss4=das_log)
        message2 = 'RMSE_std: {loss1.std:.5f} || DAS_test: {loss4.std:.5f} || PSNR_std: {loss2.std:.5f} || SSIM_std: {loss3.std:.5f}'.format(loss1=rmse_log,loss2=psnr_log,loss3=ssim_log, loss4=das_log)
        LOGGER.info(message1)
        LOGGER.info(message2)
    LOGGER.info('Finish Testing')

def is_model_better(loss_val_log, loss_val_psnr, loss_val_ssim, 
                    loss_val_lpips, best_perf_record):
    if loss_val_lpips.ave == 0:
        count = 0
        if loss_val_log.ave < best_perf_record[0]:
            count += 1
        if loss_val_psnr.ave > best_perf_record[1]:
            count += 1
        if loss_val_ssim.ave > best_perf_record[2]:
            count += 1
        return count >= 2
    else:
        count = 0
        if loss_val_log.ave < best_perf_record[0]:
            count += 1
        if loss_val_psnr.ave > best_perf_record[1]:
            count += 1
        if loss_val_ssim.ave > best_perf_record[2]:
            count += 1
        if loss_val_lpips.ave < best_perf_record[3]:
            count += 1
        return count >= 3

def train():
    args = get_args()

    if args.channel_num == 128:
        das_recon_model = DASAndPixelInterpolator()
    else:
        das_recon_model = DASAndPixelInterpolator_MSOT()

    # Setup multi-GPU
    gpu_ids = setup_multi_gpu(args)
    device = torch.device(f'cuda:{gpu_ids[0]}')
    
    # 计算gradient accumulation steps
    accumulation_steps = args.effective_batch_size // args.train_batchsize
    actual_effective_batch_size = accumulation_steps * args.train_batchsize
    
    # ----------record args info in log file--------------
    LOGGER = ConsoleLogger('train_'+args.trial, 'train')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)
    LOGGER.info(f'Using GPUs: {gpu_ids}')
    LOGGER.info(f'Gradient Accumulation Steps: {accumulation_steps}')
    LOGGER.info(f'Actual Batch Size: {args.train_batchsize}')
    LOGGER.info(f'Target Effective Batch Size: {args.effective_batch_size}')
    LOGGER.info(f'Actual Effective Batch Size: {actual_effective_batch_size}')
    torch.manual_seed(args.seed)
    
    # -------save current code---------------------------
    save_code_root = os.path.join(logdir, 'code.tar')
    dst_root = os.path.abspath(__file__)
    create_code_snapshot(dst_root, save_code_root)
    
    # dataset============================================================
    train_set = dataset_PACT()
    train_dataloader = DataLoader(train_set, batch_size=args.train_batchsize,
                                  shuffle=True, num_workers=16, drop_last=True)
    val_set = dataset_PACT()
    val_dataloader = DataLoader(val_set, batch_size=args.val_batchsize, 
                                shuffle=False, num_workers=16, drop_last=False)
    test_set = dataset_PACT()
    test_dataloader = DataLoader(test_set, batch_size=args.val_batchsize, 
                                 shuffle=False, num_workers=16, drop_last=False)
    LOGGER.info('Initial Dataset Finished')
    
    # model
    if args.model_type == 'Forward':
        model = Forward_Network(type='type3', unet_type='fd_unet', target_channels=1, inner_channel=32,
                                matrix_path='photoacoustic_system_matrix_A_limited_view.mat')
    else:
        raise ValueError(f"Model type {args.model_type} is not supported.")
    
    # Move model to primary GPU first
    model = model.to(device)
    
    # Wrap model with DataParallel if multiple GPUs are available
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        LOGGER.info(f'Model wrapped with DataParallel using GPUs: {gpu_ids}')
    
    # Loss function
    Loss_img = eval(args.loss_type + '(args=args)').to(device)
    best_perf_record = [100.0, 0.0, -2.0, 2.0]

    # Load pretrained model if specified
    if args.load_model:
        model_path = args.load_model
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No checkpoint found at {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle DataParallel model loading
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        LOGGER.info(f'---------------Finishing loading models----------------')

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    # Adjust scheduler step size based on gradient accumulation
    total_steps = len(train_dataloader) * args.max_epoch // accumulation_steps
    LOGGER.info(f'Total training steps: {total_steps}')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )

    # Training loop
    for epoch in range(args.max_epoch):
        LOGGER.info(f'---------------Training epoch : {epoch}-----------------')
        batch_time = AverageMeter()
        loss_log = AverageMeter()
        loss_val_log = AverageMeter()
        loss_val_psnr = AverageMeter()
        loss_val_ssim = AverageMeter()
        loss_val_lpips = AverageMeter()
        start = time.time()

        model.train()
        
        # 用于gradient accumulation的变量
        accumulated_loss = 0.0
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        for it, batch in enumerate(train_dataloader, 0):
            input = batch['p0'].to(torch.float32).cuda(device)
            gt = batch['sinogram'].to(torch.float32).cuda(device)
            minmax = np.array([batch['d_min'].numpy(), batch['d_max'].numpy()]).T
            batch_size = len(input)

            output = model(input)
            
            loss_term = Loss_img(pred=output, gt=gt, epoch=epoch, minmax=minmax)

            # Calculate total loss based on loss type
            if (args.loss_type == 'mse') :
                loss = loss_term['mse_loss']
                if (it + 1) % (args.freq_print_train * accumulation_steps) == 0:
                    print('MSE-Term:{:.5f}'.format(loss_term['mse_loss'].item()))

            if (args.loss_type == 'forward') :
                loss = loss_term['mse_loss'] + loss_term['das_loss'] + loss_term['lut_loss']
                if (it + 1) % (args.freq_print_train * accumulation_steps) == 0:
                    print('MSE-Term:{:.5f} DAS-Term:{:.5f} LUT-Term:{:.5f}'.format(loss_term['mse_loss'].item(), loss_term['das_loss'].item(), loss_term['lut_loss'].item()))

            # 将loss除以accumulation_steps，这样累积的梯度就相当于大batch size的平均梯度
            loss = loss / accumulation_steps
            accumulated_loss += loss.item()

            loss.backward()
            
            # 每accumulation_steps步或者到epoch结尾时执行optimizer step
            if (it + 1) % accumulation_steps == 0 or (it + 1) == len(train_dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 打印信息，使用accumulated loss
                total_accumulated_loss = accumulated_loss * accumulation_steps  # 恢复到原始scale
                batch_time.update(time.time() - start)
                loss_log.update(total_accumulated_loss, actual_effective_batch_size)
                
                if (it + 1) % (args.freq_print_train * accumulation_steps) == 0:
                    message = 'Epoch : [{0}][{1}/{2}]  Learning rate  {learning_rate:.7f}\t' \
                              'Batch Time {batch_time.val:.3f}s ({batch_time.ave:.3f})\t' \
                              'Speed {speed:.1f} samples/s \t' \
                              'Loss_train {loss1.val:.5f} ({loss1.ave:.5f})\t' \
                              'Effective Batch Size: {effective_bs}\t'.format(
                        epoch, (it + 1) // accumulation_steps, len(train_dataloader) // accumulation_steps,
                        learning_rate=optimizer.param_groups[0]['lr'],
                        batch_time=batch_time, speed=batch_size * accumulation_steps / batch_time.val,
                        loss1=loss_log, effective_bs=args.effective_batch_size)
                    LOGGER.info(message)
                
                # 重置accumulated loss
                accumulated_loss = 0.0
                start = time.time()

            # scheduler.step()

        LOGGER.info(f'---------------Training {epoch} end---------------')
        LOGGER.info(f'--------------Validation {epoch} start--------------')

        # Validation
        validate(val_dataloader, LOGGER, device, model, das_recon_model, 
                 loss_val_log, loss_val_psnr, loss_val_ssim, loss_val_lpips, args)

        # Save best model
        if is_model_better(loss_val_log, loss_val_psnr, loss_val_ssim, 
                           loss_val_lpips, best_perf_record):
            best_perf_record = [loss_val_log.ave, loss_val_psnr.ave, loss_val_ssim.ave, loss_val_lpips.ave]
            save_state(model, optimizer, LOGGER, logdir, name='best_perf')

        if args.save_epoch == 1:
            save_state(model, optimizer, LOGGER, logdir, name='last')
    
    save_state(model, optimizer, LOGGER, logdir, name='last')

    message = 'Best metrics: RMSE_val {loss1:.5f} || PSNR {loss2:.5f} || SSIM {loss3:.5f} || DAS: {loss4:.5f}\t'.format(
        loss1=best_perf_record[0],loss2=best_perf_record[1],loss3=best_perf_record[2],loss4=best_perf_record[3])
    LOGGER.info(message)
    LOGGER.info('Finish Training')

    # Test with best model
    LOGGER.info('Start Testing the saved model')
    checkpoint_dir = os.path.join(logdir, 'checkpoints')
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_perf.tar'), weights_only=False)
    
    # Load model state dict properly for DataParallel
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    LOGGER.info(f'---------------Finishing loading models----------------')

    test(test_dataloader, LOGGER, device, model, das_recon_model, args)


if __name__ == "__main__":
    args = get_args()
    
    # 自动检测是否使用多GPU
    if torch.cuda.device_count() > 1:
        args.multi_gpu = True
        print(f"Detected {torch.cuda.device_count()} GPUs, enabling multi-GPU training")
    
    if args.mode == 'train':
        train()