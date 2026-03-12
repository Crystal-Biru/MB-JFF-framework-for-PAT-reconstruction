import torch
import torch.nn as nn
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
import os

from dataset_PACT import dataset_PACT
from models.HybridNetwork import Hybrid_Network
from models.forward_operator import Forward_Network
from models.GE_CNN import GE_CNN  

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
    parser.add_argument('--gpu', type=str, default='0', help='GPU IDs to use, comma separated')
    parser.add_argument('--trial',type=str, default='dual_input_baseline')
    parser.add_argument('--mode', type=str, default='train',choices=['train','test','test_offline'])
    parser.add_argument('--seed',type=int,default=1)

    # ==========define the task==============
    parser.add_argument('--channel_num', type=int, default=64)  # 128 for Linear-Data, 64 for MSOT-Data
    parser.add_argument('--time_step_num', type=int, default=2030)  # 4096 for Linear-Data, 2030 for MSOT-Data
    parser.add_argument('--p0_pixel', type=int, default=128)  # p0 image size

    # =========for training===========
    parser.add_argument('--train_batchsize', type=int, default=2, help='Physical batch size per GPU')
    parser.add_argument('--effective_batchsize', type=int, default=16, help='Effective batch size via gradient accumulation')
    parser.add_argument('--val_batchsize', type=int, default=2, help='Total batch size across all GPUs')
    parser.add_argument('--max_epoch', default=300, help='max training epoch', type=int)

    parser.add_argument('--lr', default=5e-4, help='learning rate', type=float) # 5e-4
    parser.add_argument('--weight_decay', default=0.0001, help='decay of learning rate', type=float)
    
    parser.add_argument('--freq_print_train', default=20, help='Printing frequency for training', type=int)
    parser.add_argument('--freq_print_val', default=20, help='Printing frequency for validation', type=int)
    parser.add_argument('--freq_print_test', default=50, help='Printing frequency for test', type=int)
    parser.add_argument('--load_model', type=str, default='')

    # ==========loss function============
    parser.add_argument('--loss_type',type=str,default='adjoint', 
                        choices=['mse','percept','ssim','adjoint'])
    parser.add_argument('--w_pxl',type=float,default=1.0)
    parser.add_argument('--w_per',type=float,default=0.4)  # 0.3
    parser.add_argument('--w_ssim',type=float,default=0.5)  # 0.5
    
    # ========for dual input model ==============
    parser.add_argument('--dual_model_type', type=str, default='Hybrid', 
                        choices=['Hybrid'])
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--encoder_blocks', type=int, default=3)
    parser.add_argument('--decoder_blocks', type=int, default=4)
    parser.add_argument('--drop_path_rate', type=float, default=0.)
    
    # ========for pretrained models paths==============
    parser.add_argument('--forward_model_path', type=str, 
                        default='',
                        help='Path to pretrained forward model')
    # Linear-Vessel: 2025-09-10-11-46-22
    # Linear-Phantom: 2025-10-29-20-07-15
    # Linear-Segment: 2025-10-31-11-49-56
    # MSOT-Abdomen: 2025-11-03-11-04-55

    # ========for saving results==============
    parser.add_argument('--save_epoch',type=int, default=1) # 1 for saving model current epoch
    parser.add_argument('--save_dir', type=str, default='./results_regulization', help='Directory to save sinogram results')

    # parse configs
    args = parser.parse_args()
    
    # Calculate gradient accumulation steps
    args.accumulation_steps = args.effective_batchsize // args.train_batchsize
     
    return args

def setup_multi_gpu(args):
    """Setup multi-GPU configuration"""
    if args.gpu:
        gpu_ids = [int(x) for x in args.gpu.split(',')]
        print(f"Using GPUs: {gpu_ids}")
        
        # Set the primary GPU
        torch.cuda.set_device(gpu_ids[0])
        
        # Check if all GPUs are available
        for gpu_id in gpu_ids:
            if gpu_id >= torch.cuda.device_count():
                raise ValueError(f"GPU {gpu_id} is not available")
        
        return gpu_ids
    else:
        return [0]

def load_pretrained_model(model, checkpoint_path, device, model_name, logger):
    """Load pretrained model weights"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle loading checkpoint for different model types
    if isinstance(model, nn.DataParallel):
        # If checkpoint was saved from DataParallel model
        if 'module.' in list(checkpoint['model_state_dict'].keys())[0]:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # If checkpoint was saved from single GPU model
            model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Remove 'module.' prefix if loading single GPU model from DataParallel checkpoint
        state_dict = checkpoint['model_state_dict']
        if 'module.' in list(state_dict.keys())[0]:
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    
    logger.info(f'Successfully loaded {model_name} from {checkpoint_path}')
    return model

def save_state(model, optimizer, LOGGER, logdir, name='best_perf'): 
    checkpoint_dir = os.path.join(logdir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    LOGGER.info('=> saving checkpoint to {}'.format(checkpoint_dir))
    states = dict()
    # Handle DataParallel model
    if isinstance(model, nn.DataParallel):
        states['model_state_dict'] = model.module.state_dict()
    else:
        states['model_state_dict'] = model.state_dict()
    states['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(states, os.path.join(checkpoint_dir, name+'.tar'))

def validate(args, val_dataloader, LOGGER, device, forward_model, dual_model, 
             loss_val_log, loss_val_psnr, loss_val_ssim, loss_val_lpips):
    loss_lpips = lpips.LPIPS(net='alex').cuda(device)
    
    # ================validation process=============================================
    with torch.no_grad():
        forward_model.eval()
        dual_model.eval()
        
        for it, batch in enumerate(val_dataloader, 0):
            sinogram = batch['sinogram'].to(torch.float32).cuda(device)  # [B, 1, channel_num, time_step_num]
            p0_gt = batch['p0'].to(torch.float32).cuda(device)  # [B, 1, 256, 256]
            batch_size = len(sinogram)
            minmax = np.array([batch['d_min'].numpy(), batch['d_max'].numpy()]).T
            
            x = dual_model.module.get_das(sinogram, norm_type='clamp')
            y_pred = forward_model(x)
            residual = sinogram - y_pred
            o, o_lut = dual_model.module.get_das_lut(residual, norm_type='abs')
            x_res = torch.cat((o, o_lut), dim=1)
            final_output, x_hat = dual_model(x, x_res, sinogram)

            for idx in range(len(final_output)):
                rmse = calc_rmse(final_output[idx, 0].cpu().numpy(), p0_gt[idx, 0].cpu().numpy(), minmax[idx])
                loss_val_log.update(rmse, 1)
                psnr = calc_psnr(final_output[idx, 0].cpu().numpy(), p0_gt[idx, 0].cpu().numpy(), minmax[idx])
                loss_val_psnr.update(psnr, 1)
                ssim = calc_ssim(final_output[idx, 0].cpu().numpy(), p0_gt[idx, 0].cpu().numpy(), minmax[idx])
                loss_val_ssim.update(ssim, 1)

            output_exp = final_output.expand(len(final_output),3,p0_gt.shape[2],p0_gt.shape[3])
            output_exp = torch.clip(output_exp,-1,1)
            gt_exp = p0_gt.expand(len(p0_gt),3,p0_gt.shape[2],p0_gt.shape[3])
            gt_exp = torch.clip(gt_exp,-1,1)
            lpips_error = loss_lpips(output_exp,gt_exp)
            loss_val_lpips.update(lpips_error.cpu().numpy().mean(),len(output_exp))

        message = 'RMSE_val {loss1.ave:.5f} || PSNR {loss2.ave:.5f} || SSIM {loss3.ave:.5f} || LPIPS: {loss4.ave:.5f}\t'.format(
            loss1=loss_val_log,loss2=loss_val_psnr,loss3=loss_val_ssim,loss4=loss_val_lpips)
        LOGGER.info(message)

def test(args, test_dataloader, LOGGER, device, forward_model, dual_model):
    rmse_log = AverageMeter()
    psnr_log = AverageMeter()
    ssim_log = AverageMeter()
    lpips_log = AverageMeter()
    timing_log = AverageMeter()  # 添加计时记录器
    loss_lpips = lpips.LPIPS(net='alex').cuda(device)

    # ================test process=============================================
    with torch.no_grad():
        forward_model.eval()
        dual_model.eval()
        sample_count = 0

        for it, batch in enumerate(test_dataloader, 0):
            sinogram = batch['sinogram'].to(torch.float32).cuda(device)  # [B, 1, channel_num, time_step_num]
            p0_gt = batch['p0'].to(torch.float32).cuda(device)  # [B, 1, 256, 256]
            batch_size = len(sinogram)
            minmax = np.array([batch['d_min'].numpy(), batch['d_max'].numpy()]).T

            x = dual_model.module.get_das(sinogram, norm_type='clamp')
            y_pred = forward_model(x)
            residual = sinogram - y_pred
            o, o_lut = dual_model.module.get_das_lut(residual, norm_type='abs')
            x_res = torch.cat((o, o_lut), dim=1)
            final_output, x_hat = dual_model(x, x_res, sinogram)

            for idx in range(len(final_output)):
                rmse = calc_rmse(final_output[idx, 0].cpu().numpy(), p0_gt[idx, 0].cpu().numpy(), minmax[idx])
                rmse_log.update(rmse, 1)
                psnr = calc_psnr(final_output[idx, 0].cpu().numpy(), p0_gt[idx, 0].cpu().numpy(), minmax[idx])
                psnr_log.update(psnr, 1) if psnr < float('inf') else None
                ssim = calc_ssim(final_output[idx, 0].cpu().numpy(), p0_gt[idx, 0].cpu().numpy(), minmax[idx])
                ssim_log.update(ssim, 1)

            output_exp = final_output.expand(len(final_output),3,p0_gt.shape[2],p0_gt.shape[3])
            output_exp = torch.clip(output_exp,-1,1)
            gt_exp = p0_gt.expand(len(p0_gt),3,p0_gt.shape[2],p0_gt.shape[3])
            gt_exp = torch.clip(gt_exp,-1,1)
            lpips_error = loss_lpips(output_exp,gt_exp)
            lpips_log.update(lpips_error.cpu().numpy().mean(),len(output_exp))
    
        message1 = 'RMSE_test: {loss1.ave:.5f} || PSNR_test: {loss2.ave:.5f} || SSIM_test: {loss3.ave:.5f} || LPIPS_test: {loss4.ave:.5f}'.format(loss1=rmse_log,loss2=psnr_log,loss3=ssim_log,loss4=lpips_log)
        message2 = 'RMSE_std: {loss1.std:.5f} || PSNR_std: {loss2.std:.5f} || SSIM_std: {loss3.std:.5f} || LPIPS_std: {loss4.std:.5f}'.format(loss1=rmse_log,loss2=psnr_log,loss3=ssim_log,loss4=lpips_log)
        message3 = 'Inference_time_avg: {timing.ave:.3f} ms || Inference_time_std: {timing.std:.3f} ms'.format(timing=timing_log)
        LOGGER.info(message1)
        LOGGER.info(message2)
        LOGGER.info(message3)
    LOGGER.info('Finish Testing')

def is_model_better(loss_val_log, loss_val_psnr, loss_val_ssim, 
                    loss_val_lpips, best_perf_record):
    if loss_val_psnr.ave == 0:
        if loss_val_log.ave < best_perf_record[0]:
            return True
        return False
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
    
    # Setup multi-GPU
    gpu_ids = setup_multi_gpu(args)
    device = torch.device(f'cuda:{gpu_ids[0]}')
    
    # ----------record args info in log file--------------
    LOGGER = ConsoleLogger('train_'+args.trial, 'train')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)
    LOGGER.info(f'Using GPUs: {gpu_ids}')
    LOGGER.info(f'Gradient accumulation steps: {args.accumulation_steps}')
    LOGGER.info(f'Effective batch size: {args.effective_batchsize}')
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
                                shuffle=False, num_workers=16, drop_last=True)
    test_set = dataset_PACT()
    test_dataloader = DataLoader(test_set, batch_size=args.val_batchsize, 
                                 shuffle=False, num_workers=16, drop_last=True)
    LOGGER.info('Initial Dataset Finished')
    
    # =========Load pretrained models=========
    LOGGER.info('Loading pretrained models...')
    
    # Load forward model  
    forward_model = Forward_Network(type='type3', unet_type='fd_unet', target_channels=1, inner_channel=32,
                                    matrix_path='photoacoustic_system_matrix_A_limited_view.mat')
    forward_model = forward_model.to(device)
    if len(gpu_ids) > 1:
        forward_model = nn.DataParallel(forward_model, device_ids=gpu_ids)
    forward_model = load_pretrained_model(forward_model, args.forward_model_path, device, 'forward_model', LOGGER)
    
    for param in forward_model.parameters():
        param.requires_grad = False
    
    LOGGER.info('Pretrained models loaded and frozen')
    
    # =========Initialize dual input model=========
   if args.dual_model_type == 'Hybrid':
        dual_model = Hybrid_Network(
            reconstruction_type='type4', unet_type='convuam', DAS_type='MSOT',
            sino_height=args.channel_num, sino_width=args.time_step_num, 
            target_size=args.p0_pixel, inner_channel=32, 
            encoder_blocks=args.encoder_blocks, decoder_blocks=args.decoder_blocks
        )
    else:
        raise ValueError(f"Dual model type {args.dual_model_type} is not supported.")
    
    # Move model to device
    dual_model = dual_model.to(device)
    
    # Wrap model with DataParallel for multi-GPU
    if len(gpu_ids) > 1:
        dual_model = nn.DataParallel(dual_model, device_ids=gpu_ids)
        LOGGER.info(f'Dual input model wrapped with DataParallel on GPUs: {gpu_ids}')
    
    LOGGER.info(f'Using dual input model type: {args.dual_model_type}')
    
    if args.loss_type == 'adjoint':
        Loss_mse = mse(args=args).to(device)
        Loss_ssim = adjoint(args=args).to(device)
        epoch_threshold = args.max_epoch // 2  # Threshold epoch for switching loss type
    elif args.loss_type == 'ssim':
        Loss_img = adjoint(args=args).to(device)
    else:
        Loss_img = eval(args.loss_type + '(args=args)').to(device)
    best_perf_record = [100.0, 0.0, -2.0, 2.0]

    if args.load_model:
        dual_model = load_pretrained_model(dual_model, args.load_model, device, 'dual_input_model', LOGGER)
     
    optimizer = torch.optim.AdamW(params=dual_model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    # Adjust scheduler step size based on gradient accumulation
    total_steps = len(train_dataloader) * args.max_epoch // args.accumulation_steps
    LOGGER.info(f'Total training steps: {total_steps}')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-7
    )
    
    # ================train process=============================================
    for epoch in range(args.max_epoch):
        LOGGER.info(f'---------------Training epoch : {epoch}-----------------')
        batch_time = AverageMeter()
        loss_log = AverageMeter()
        loss_val_log = AverageMeter()
        loss_val_psnr = AverageMeter()
        loss_val_ssim = AverageMeter()
        loss_val_lpips = AverageMeter()
        start = time.time()

        dual_model.train()
        forward_model.eval()
        
        # Initialize gradient accumulation variables
        accumulated_loss = 0.0
        optimizer.zero_grad()

        if args.loss_type == 'adjoint':
            Loss_img = Loss_mse if epoch < epoch_threshold else Loss_ssim
        
        for it, batch in enumerate(train_dataloader, 0):  # iterate through the dataloader
            # move data to device (cpu/cuda)
            sinogram = batch['sinogram'].to(torch.float32).cuda(device)
            p0_gt = batch['p0'].to(torch.float32).cuda(device)
            minmax = np.array([batch['d_min'].numpy(), batch['d_max'].numpy()]).T
            batch_size = len(sinogram)
            
            with torch.no_grad():
                x = dual_model.module.get_das(sinogram, norm_type='clamp')
                y_pred = forward_model(x)
                residual = sinogram - y_pred
                o, o_lut = dual_model.module.get_das_lut(residual, norm_type='abs')
                x_res = torch.cat((o, o_lut), dim=1)
            final_output, x_hat = dual_model(x, x_res, sinogram)

            loss_term1 = Loss_img(pred=final_output, gt=p0_gt, minmax=minmax)
            loss_term2 = Loss_img(pred=x_hat, gt=p0_gt, minmax=minmax)
            loss_term = {}
            for key in loss_term1.keys():
                loss_term[key] = loss_term1[key] * 0.6 + loss_term2[key] * 0.4

            if (args.loss_type == 'mse') :
                loss = loss_term['mse_loss']
                if (it + 1) % (args.freq_print_train * args.accumulation_steps) == 0:
                    print('MSE-Term:{:.5f}'.format(loss_term['mse_loss'].item()))

            elif (args.loss_type == 'ssim'):
                loss = args.w_pxl * loss_term['mse_loss'] + args.w_ssim * loss_term['ssim_loss']
                if (it + 1) % (args.freq_print_train * args.accumulation_steps) == 0:
                    print('MSE-Term:{:.5f} SSIM-Term:{:.5f}'.format(loss_term['mse_loss'].item(), loss_term['ssim_loss'].item()))

            if (args.loss_type == 'adjoint'):
                # 动态 loss_type 处理
                if epoch < epoch_threshold:
                    loss = loss_term['mse_loss']
                    if (it + 1) % (args.freq_print_train * args.accumulation_steps) == 0:
                        print('MSE-Term:{:.5f}'.format(loss_term['mse_loss'].item()))
                else:
                    loss = args.w_pxl * loss_term['mse_loss'] + args.w_ssim * loss_term['ssim_loss']
                    if (it + 1) % (args.freq_print_train * args.accumulation_steps) == 0:
                        print('MSE-Term:{:.5f} SSIM-Term:{:.5f}'.format(loss_term['mse_loss'].item(), loss_term['ssim_loss'].item()))

            loss = loss / args.accumulation_steps
            accumulated_loss += loss.item()
            
            loss.backward()
            
            # Update parameters every accumulation_steps
            if (it + 1) % args.accumulation_steps == 0 or (it + 1) == len(train_dataloader):

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # ===============Logging the info================
                batch_time.update(time.time() - start)
                # Use accumulated loss for logging
                loss_log.update(accumulated_loss, batch_size * args.accumulation_steps)
                
                if (it + 1) % (args.freq_print_train * args.accumulation_steps) == 0:
                    message = 'Epoch : [{0}][{1}/{2}]  Learning rate  {learning_rate:.7f}\t' \
                              'Batch Time {batch_time.val:.3f}s ({batch_time.ave:.3f})\t' \
                              'Speed {speed:.1f} samples/s \t' \
                              'Loss_train {loss1.val:.5f} ({loss1.ave:.5f})\t' \
                              'Effective Batch Size: {effective_bs}\t'.format(
                        epoch, (it + 1) // args.accumulation_steps, len(train_dataloader) // args.accumulation_steps,
                        learning_rate=optimizer.param_groups[0]['lr'],
                        batch_time=batch_time, speed=batch_size * args.accumulation_steps / batch_time.val,
                        loss1=loss_log, effective_bs=args.effective_batchsize)
                    LOGGER.info(message)
                
                accumulated_loss = 0.0
                start = time.time()

        LOGGER.info(f'---------------Training {epoch} end---------------')
        LOGGER.info(f'--------------Validation {epoch} end--------------')

        # ================validation process=============================================
        validate(args, val_dataloader, LOGGER, device, forward_model, dual_model, 
                 loss_val_log, loss_val_psnr, loss_val_ssim, loss_val_lpips)

        if is_model_better(loss_val_log, loss_val_psnr, loss_val_ssim,
                           loss_val_lpips, best_perf_record):
            best_perf_record = [loss_val_log.ave, loss_val_psnr.ave, loss_val_ssim.ave, loss_val_lpips.ave]
            save_state(dual_model, optimizer, LOGGER, logdir, name='best_perf')

        if args.save_epoch == 1:
            save_state(dual_model, optimizer, LOGGER, logdir, name='last')
    
    save_state(dual_model, optimizer, LOGGER, logdir, name='last')

    message = 'Best metrics: RMSE_val {loss1:.5f} || PSNR {loss2:.5f} || SSIM {loss3:.5f} || LPIPS: {loss4:.5f}\t'.format(
        loss1=best_perf_record[0],loss2=best_perf_record[1],loss3=best_perf_record[2],loss4=best_perf_record[3])
    LOGGER.info(message)
    LOGGER.info('Finish Training')

    LOGGER.info('Start Testing the saved model')
    checkpoint_dir = os.path.join(logdir, 'checkpoints')
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_perf.tar'), weights_only=False)
    
    if isinstance(dual_model, nn.DataParallel):
        dual_model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        dual_model.load_state_dict(checkpoint['model_state_dict'])
    LOGGER.info(f'---------------Finishing loading models----------------')

    # ================test process=============================================
    test(args, test_dataloader, LOGGER, device, forward_model, dual_model)


if __name__ == "__main__":
    args = get_args()

    if args.mode == 'train':
        train()