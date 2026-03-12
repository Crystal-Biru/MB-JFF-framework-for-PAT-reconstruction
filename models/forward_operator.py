import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import h5py
from scipy.sparse import csc_matrix, csr_matrix
import warnings
from models.FD_UNet import Conv2dBatchNorm, FDBlock, CBAM, ASPP, FD_UNet, U_Net
from models.ConvNext import ConvNeXt
from models.ConvUAM import ConvUAM
from models.MobileVit import MobileViT_SR

# 新增：系统矩阵A处理类
class SystemMatrixOperator(nn.Module):
    """系统矩阵A的加载和前向投影操作
    
    处理MATLAB保存的稀疏矩阵 A (524288×65536)
    - 524288 = M * L (传感器数量 × 时间步数)
    - 65536 = H * W (图像高度 × 宽度)
    - 支持GPU稀疏矩阵运算以节省显存
    """
    def __init__(self, matrix_path='~/PA_recon/photoacoustic_system_matrix_A.mat', 
                 device='cuda', M=128, L=4096, H=256, W=256, 
                 sparse_format='coo', chunk_size=None):
        super().__init__()
        self.matrix_path = matrix_path
        self.device = device
        self.M = M  # 传感器数量
        self.L = L  # 时间步数
        self.H = H  # 图像高度
        self.W = W  # 图像宽度
        self.sparse_format = sparse_format  # 'coo', 'csr', 'csc'
        self.chunk_size = chunk_size or (8 if device == 'cuda' else None)  # GPU批处理大小
        
        # 验证维度匹配
        expected_rows = M * L
        expected_cols = H * W
        print(f"期望矩阵维度: {expected_rows} × {expected_cols}")
        
        self.A_sparse_torch = None
        self.A_sparse_scipy = None
        self.load_matrix()
    
    def load_matrix(self):
        """加载系统矩阵A为PyTorch稀疏张量"""
        try:
            print(f"正在加载系统矩阵: {self.matrix_path}")
            
            A_sparse_scipy = self._load_scipy_sparse()
            self._convert_to_torch_sparse(A_sparse_scipy)
            
        except Exception as e:
            print(f"加载系统矩阵失败: {e}")
            raise e
    
    def _load_scipy_sparse(self):
        """加载scipy稀疏矩阵"""
        # 首先尝试使用h5py加载-v7.3格式
        try:
            with h5py.File(self.matrix_path, 'r') as f:
                print("检测到-v7.3格式的.mat文件")
                if 'A' in f:
                    # 读取稀疏矩阵的各个组件
                    if 'data' in f['A'] and 'ir' in f['A'] and 'jc' in f['A']:
                        # CSC格式稀疏矩阵
                        data = f['A']['data'][:]
                        ir = f['A']['ir'][:]  # row indices
                        jc = f['A']['jc'][:]  # column pointers
                        
                        # 获取矩阵维度
                        if 'm' in f['A'] and 'n' in f['A']:
                            m = int(f['A']['m'][0, 0])
                            n = int(f['A']['n'][0, 0])
                        else:
                            m = self.M * self.L
                            n = self.H * self.W
                        
                        print(f"稀疏矩阵维度: {m} × {n}")
                        print(f"非零元素数量: {len(data)}")
                        print(f"稀疏度: {len(data) / (m * n) * 100:.4f}%")
                        
                        # 创建scipy稀疏矩阵
                        A_sparse_scipy = csc_matrix((data.flatten(), ir.flatten(), jc.flatten()), shape=(m, n))
                        
                    else:
                        # 尝试读取密集矩阵
                        A_dense = f['A'][:]
                        print(f"密集矩阵维度: {A_dense.shape}")
                        A_sparse_scipy = csc_matrix(A_dense)
                else:
                    raise KeyError("矩阵A未在HDF5文件中找到")
                    
        except (OSError, KeyError) as e:
            print(f"尝试h5py加载失败: {e}")
        
        # 验证矩阵维度
        expected_shape = (self.M * self.L, self.H * self.W)
        if A_sparse_scipy.shape != expected_shape:
            print(f"警告: 矩阵维度 {A_sparse_scipy.shape} 与期望维度 {expected_shape} 不匹配")
        
        return A_sparse_scipy
    
    def _convert_to_torch_sparse(self, A_sparse_scipy):
        """将scipy稀疏矩阵转换为PyTorch稀疏张量"""
        print(f"转换为PyTorch {self.sparse_format.upper()}稀疏张量...")
        
        # 根据格式选择转换方式
        if self.sparse_format == 'coo':
            A_coo = A_sparse_scipy.tocoo()
            indices = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
            values = torch.from_numpy(A_coo.data).float()
            size = A_coo.shape
            
            self.A_sparse_torch = torch.sparse_coo_tensor(
                indices, values, size, 
                device=self.device, 
                dtype=torch.float32
            ).coalesce()
            
        elif self.sparse_format == 'csr':
            A_csr = A_sparse_scipy.tocsr()
            # PyTorch的CSR格式
            crow_indices = torch.from_numpy(A_csr.indptr).long()
            col_indices = torch.from_numpy(A_csr.indices).long()
            values = torch.from_numpy(A_csr.data).float()
            
            self.A_sparse_torch = torch.sparse_csr_tensor(
                crow_indices, col_indices, values, 
                size=A_csr.shape,
                device=self.device,
                dtype=torch.float32
            )
            
        elif self.sparse_format == 'csc':
            # 先转为COO，因为PyTorch的CSC支持有限
            A_coo = A_sparse_scipy.tocoo()
            indices = torch.from_numpy(np.vstack([A_coo.row, A_coo.col])).long()
            values = torch.from_numpy(A_coo.data).float()
            size = A_coo.shape
            
            self.A_sparse_torch = torch.sparse_coo_tensor(
                indices, values, size, 
                device=self.device, 
                dtype=torch.float32
            ).coalesce()
        
        # 保存scipy版本作为备用（用于CPU计算）
        if self.device == 'cpu':
            self.A_sparse_scipy = A_sparse_scipy
        
        print(f"PyTorch稀疏张量创建完成，设备: {self.device}")
    
    def _print_memory_info(self):
        """打印内存使用信息"""
        if self.A_sparse_torch is not None:
            nnz = self.A_sparse_torch._nnz()
            total_elements = self.A_sparse_torch.shape[0] * self.A_sparse_torch.shape[1]
            sparsity = nnz / total_elements
            
            # 估算内存使用（索引 + 数值）
            if self.sparse_format == 'coo':
                # COO: 2个索引数组 + 1个值数组
                memory_bytes = nnz * (2 * 8 + 4)  # 2*int64 + 1*float32
            elif self.sparse_format == 'csr':
                # CSR: 行指针 + 列索引 + 值
                memory_bytes = (self.A_sparse_torch.shape[0] + 1) * 8 + nnz * (8 + 4)
            else:  # COO as fallback
                memory_bytes = nnz * (2 * 8 + 4)
            
            print(f"稀疏张量信息:")
            print(f"  - 形状: {self.A_sparse_torch.shape}")
            print(f"  - 非零元素: {nnz:,}")
            print(f"  - 稀疏度: {sparsity:.6f} ({sparsity*100:.4f}%)")
            print(f"  - 估算内存: {memory_bytes / (1024**2):.2f} MB")
            if self.device != 'cpu':
                print(f"  - GPU设备: {self.device}")
    
    def forward_projection(self, p0):
        """
        前向投影: y = A * p0 (GPU稀疏矩阵运算)
        Args:
            p0: [B, 1, H, W] 初始压力分布
        Returns:
            sinogram: [B, 1, M, L] 重塑后的sinogram
        """
        if self.A_sparse_torch is None:
            raise RuntimeError("系统矩阵A未正确加载")
        
        batch_size, channels, H, W = p0.shape
        if p0.device != self.A_sparse_torch.device:
            p0_device = p0.device
            A_sparse_torch = self.A_sparse_torch.to(p0_device)
        else:
            A_sparse_torch = self.A_sparse_torch

        if H != self.H or W != self.W:
            print(f"警告: 输入图像尺寸 {H}×{W} 与期望尺寸 {self.H}×{self.W} 不匹配")
        
        p_img = p0.squeeze(1)  # [B, H, W] 移除通道维度
        p_transposed = p_img.transpose(-2, -1)  # [B, W, H] - 转置最后两个维度
        p_vec = p_transposed.contiguous().view(batch_size, -1)  # [B, H*W] 按MATLAB顺序展平
        
        # 前向投影计算
        with torch.no_grad():
            if self.chunk_size and batch_size > self.chunk_size:
                y_chunks = []
                for i in range(0, batch_size, self.chunk_size):
                    end_idx = min(i + self.chunk_size, batch_size)
                    p_chunk = p_vec[i:end_idx]  # [chunk_size, H*W]
                    
                    # 稀疏矩阵乘法: A @ p_chunk^T
                    y_chunk = torch.sparse.mm(A_sparse_torch, p_chunk.t())  # [M*L, chunk_size]
                    y_chunk = y_chunk.t()  # [chunk_size, M*L]
                    y_chunks.append(y_chunk)
                
                y_vec = torch.cat(y_chunks, dim=0)  # [B, M*L]
            else:
                # 直接计算小batch或单batch
                # 稀疏矩阵乘法: A @ p_vec^T
                y_vec = torch.sparse.mm(A_sparse_torch, p_vec.t())  # [M*L, B]
                y_vec = y_vec.t()  # [B, M*L]
        
        sinogram = y_vec.view(batch_size, 1, self.M, self.L)

        max_vals = torch.max(torch.max(sinogram.view(batch_size, -1), dim=1)[0],
                           torch.tensor(1e-8, device=sinogram.device))  # 防止除零
        max_vals = max_vals.view(batch_size, 1, 1, 1)  # 广播维度
        sinogram_normalized = sinogram / max_vals
        
        return sinogram_normalized


# 增强版本1：多尺度特征提取 + 残差连接
class SinogramAdapter_CNN_Enhanced_V1(nn.Module):
    """增强版本1: 多尺度特征提取"""
    def __init__(self, sino_height=128, sino_width=4096, target_size=256, target_channels=64):
        super().__init__()
        self.sino_height = sino_height
        self.sino_width = sino_width
        self.target_size = target_size
        self.target_channels = target_channels
        
        # 多尺度特征提取分支
        self.scale1_encoder = nn.Sequential(
            Conv2dBatchNorm(1, 32, kernel_size=3, padding=1, activation='relu'),
            Conv2dBatchNorm(32, 64, kernel_size=3, padding=1, activation='relu'),
        )
        
        self.scale2_encoder = nn.Sequential(
            Conv2dBatchNorm(1, 32, kernel_size=5, padding=2, activation='relu'),
            Conv2dBatchNorm(32, 64, kernel_size=5, padding=2, activation='relu'),
        )
        
        self.scale3_encoder = nn.Sequential(
            Conv2dBatchNorm(1, 32, kernel_size=7, padding=3, activation='relu'),
            Conv2dBatchNorm(32, 64, kernel_size=7, padding=3, activation='relu'),
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            Conv2dBatchNorm(192, 128, kernel_size=1, padding=0, activation='relu'),
            Conv2dBatchNorm(128, 64, kernel_size=3, padding=1, activation='relu'),
        )
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((sino_height, sino_width))

        # 注意力机制
        self.attention = CBAM(64)
        
        # 残差细化模块
        self.residual_refine = nn.Sequential(
            Conv2dBatchNorm(64, 128, kernel_size=3, padding=1, activation='relu'),
            Conv2dBatchNorm(128, 256, kernel_size=3, padding=1, activation='relu'),
            Conv2dBatchNorm(256, 128, kernel_size=3, padding=1, activation='relu'),
            Conv2dBatchNorm(128, target_channels, kernel_size=1, padding=0, activation='relu'),
        )
        
        # 残差连接
        self.residual_conv = nn.Conv2d(64, target_channels, kernel_size=1)

        self.final = Conv2dBatchNorm(target_channels, target_channels, kernel_size=3, padding=1, activation='relu')
        
    def forward(self, x):
        # 多尺度特征提取
        scale1 = self.scale1_encoder(x)  # [B, 64, M, L]
        scale2 = self.scale2_encoder(x)  # [B, 64, M, L]
        scale3 = self.scale3_encoder(x)  # [B, 64, M, L]
        
        # 特征融合
        fused = torch.cat([scale1, scale2, scale3], dim=1)  # [B, 192, M, L]
        fused = self.feature_fusion(fused)  # [B, 64, M, L]
        
        # 自适应池化
        pooled = self.attention(self.adaptive_pool(fused))  # [B, 64, 256, 256]
        
        # 残差细化
        refined = self.residual_refine(pooled)  # [B, target_channels, 256, 256]
        residual = self.residual_conv(pooled)   # [B, target_channels, 256, 256]
        
        # 残差连接
        x = refined + residual  # [B, target_channels, 256, 256]
        output = self.final(x)  # [B, target_channels, 256, 256]

        return x + output  # [B, target_channels, 256, 256]

# 完整的PAT重建网络
class Forward_Network(nn.Module):
    """完整的光声图像重建网络"""
    def __init__(self, type='type2', adapter_type='cnn_v1', unet_type='fd_unet', 
                 sino_height=128, sino_width=4096, target_size=256, target_channels=64, 
                 inner_channel=32, matrix_path='photoacoustic_system_matrix_A.mat',
                 device='cuda'):
        super().__init__()
        
        self.type = type
        self.target_channels = target_channels
        
        # 根据type决定组件配置
        if type == 'type1':
            # 只有adapter，无initial
            self.with_initial = False
            self.with_adapter = True
        elif type == 'type2':
            # 既有adapter又有initial
            self.with_initial = True
            self.with_adapter = True
        elif type == 'type3':
            # 只有initial，无adapter
            self.with_initial = True
            self.with_adapter = False
        else:
            raise ValueError(f"Unknown type: {type}. Must be one of ['type1', 'type2', 'type3']")

        # 如果需要initial，初始化系统矩阵操作器
        if self.with_initial:
            self.system_matrix = SystemMatrixOperator(
                matrix_path=matrix_path, 
                device=device, 
                M=sino_height, 
                L=sino_width,
                H=target_size, 
                W=target_size
            )
        else:
            self.system_matrix = None

        # 如果需要adapter，初始化适配器
        if self.with_adapter:
            if adapter_type == 'cnn_v1':
                self.adapter = SinogramAdapter_CNN_Enhanced_V1(sino_height, sino_width, target_size, target_channels)
            else:
                raise ValueError(f"Unknown adapter type: {adapter_type}")
        else:
            self.adapter = None

        # 确定UNet输入通道数
        if self.with_adapter and self.with_initial:
            unet_in_ch = target_channels + 1  # adapter特征 + initial sinogram
        elif self.with_adapter and not self.with_initial:
            unet_in_ch = target_channels  # 只有adapter特征
        elif not self.with_adapter and self.with_initial:
            unet_in_ch = 1  # 只有initial sinogram
        else:
            raise ValueError("At least one of adapter or initial must be enabled")

        # 初始化UNet
        if unet_type == 'fd_unet':
            self.unet = FD_UNet(in_channel=unet_in_ch, out_channel=1, inner_channel=inner_channel)
        elif unet_type == 'unet':
            self.unet = U_Net(in_channel=unet_in_ch, out_channel=1, inner_channel=inner_channel)
        else:
            raise ValueError(f"Unknown UNet type: {unet_type}")

    def forward(self, p0):
        """
        前向传播
        Args:
            p0: [B, 1, H, W] 初始压力分布图像
        Returns:
            output: [B, 1, H, W] 重建结果
        """
        features_list = []
        
        # 处理adapter部分
        if self.with_adapter:
            adapted = self.adapter(p0)  # [B, target_channels, H, W]
            features_list.append(adapted)
        
        # 处理initial部分
        if self.with_initial and self.system_matrix is not None:
            initial_sinogram = self.system_matrix.forward_projection(p0)  # [B, 1, M, L]
            features_list.append(initial_sinogram)

        # 合并特征
        if len(features_list) == 1:
            feature = features_list[0]
        else:
            feature = torch.cat(features_list, dim=1)  # 沿通道维度拼接

        # UNet处理
        original_width = feature.shape[-1]
        if original_width == 2030:
            pad_total = 2048 - 2030
            feature = F.pad(feature, (0, pad_total, 0, 0), mode='constant', value=0)
        output = self.unet(feature)  # [B, 1, H, W]
        if original_width == 2030:
            output = output[:, :, :, :original_width]  # 裁剪回W=2030

        return torch.tanh(output)  # 输出范围在[-1, 1]之间
    
    def get_initial_sino(self, p0):
        """获取初始sinogram（如果可用）"""
        if self.with_initial and self.system_matrix is not None:
            initial_sinogram = self.system_matrix.forward_projection(p0)  # [B, 1, M, L]
            return initial_sinogram
        return None