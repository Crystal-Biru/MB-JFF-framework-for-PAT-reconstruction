import torch
import torch.nn as nn
import numpy as np


class DASAndPixelInterpolator_MSOT(nn.Module):
    """
    DAS reconstruction for limited-view MSOT with circular array
    Matches the MATLAB configuration:
    - 270° circular array with 40.6mm inner radius
    - Limited-view: 64 elements (indices 31:95 from full 128 elements)
    - Output: 128×128 ROI at center
    """

    def __init__(self,
                 num_elements_full=128,  # 全视角传感器总数
                 limited_view_range=(31, 95),  # Limited-view范围（Python索引，对应MATLAB 32:95）
                 Nx=408,  # 全网格x方向点数
                 Ny=408,  # 全网格y方向点数
                 roi_size=128,  # ROI区域尺寸
                 dx=0.2e-3,  # 网格间距(m)
                 dy=0.2e-3,  # 网格间距(m)
                 c0=1500,  # 声速(m/s)
                 fs=40e6,  # 采样频率(Hz)
                 inner_radius=40.6e-3,  # 环形阵列内径(m)
                 arc_angle=270,  # 弧度角度(度)
                 arc_start_angle=135,  # 起始角度(度)
                 time_samples=2030,  # 时间采样点数
                 device='cuda'):
        super().__init__()

        self.num_elements_full = num_elements_full
        self.limited_view_start = limited_view_range[0]
        self.limited_view_end = limited_view_range[1]
        self.num_elements = limited_view_range[1] - limited_view_range[0]

        self.Nx = Nx
        self.Ny = Ny
        self.roi_size = roi_size
        self.dx = dx
        self.dy = dy
        self.c0 = c0
        self.fs = fs
        self.dt = 1 / fs
        self.inner_radius = inner_radius
        self.arc_angle = arc_angle
        self.arc_start_angle = arc_start_angle
        self.time_samples = time_samples
        self.device = device if torch.cuda.is_available() else 'cpu'

        # 计算ROI区域范围
        self.roi_start_x = round((self.Nx - self.roi_size) / 2)
        self.roi_end_x = self.roi_start_x + self.roi_size
        self.roi_start_y = round((self.Ny - self.roi_size) / 2)
        self.roi_end_y = self.roi_start_y + self.roi_size

        # 计算网格中心（笛卡尔坐标）
        self.center_x_m = (self.Nx / 2 - 0.5) * self.dx  # MATLAB: (center_x - 1) * dx
        self.center_y_m = (self.Ny / 2 - 0.5) * self.dy

        # 预计算几何信息
        self._precompute_geometry()
        self._precompute_time_indices_and_weights()

    def _precompute_geometry(self):
        """预计算传感器位置（仅limited-view部分）"""
        # 计算所有128个传感器的角度
        angle_spacing = self.arc_angle / (self.num_elements_full - 1)
        all_sensor_angles = self.arc_start_angle + np.arange(self.num_elements_full) * angle_spacing

        # 提取limited-view部分的角度
        sensor_angles = all_sensor_angles[self.limited_view_start:self.limited_view_end]

        # 转换为弧度并计算笛卡尔坐标
        sensor_angles_rad = sensor_angles * np.pi / 180

        # 传感器在相对于中心的笛卡尔坐标（米）
        sensor_x_m = self.inner_radius * np.cos(sensor_angles_rad)
        sensor_y_m = self.inner_radius * np.sin(sensor_angles_rad)

        # 传感器的绝对笛卡尔坐标（米）
        sensor_x_abs = self.center_x_m + sensor_x_m
        sensor_y_abs = self.center_y_m + sensor_y_m

        # 保存传感器信息
        self.sensor_positions_cart = torch.tensor(
            np.column_stack([sensor_x_abs, sensor_y_abs, sensor_angles_rad]),
            dtype=torch.float32, device=self.device
        )

        print(f"Limited-view传感器配置:")
        print(f"  - 传感器数量: {self.num_elements}")
        print(f"  - 索引范围: {self.limited_view_start}-{self.limited_view_end}")
        print(f"  - 角度范围: {sensor_angles[0]:.1f}° - {sensor_angles[-1]:.1f}°")
        print(f"  - ROI区域: {self.roi_size}×{self.roi_size} (中心区域)")

    def _precompute_time_indices_and_weights(self):
        """预计算ROI区域内每个像素与每个传感器的时间索引和权重"""
        # 只计算ROI区域的像素坐标
        ix = torch.arange(self.roi_start_x, self.roi_end_x, dtype=torch.float32, device=self.device)
        iy = torch.arange(self.roi_start_y, self.roi_end_y, dtype=torch.float32, device=self.device)
        x_pos = ix * self.dx  # 绝对笛卡尔坐标(m)
        y_pos = iy * self.dy
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')  # [roi_size, roi_size]

        # 传感器坐标
        sensor_x_abs = self.sensor_positions_cart[:, 0]  # [num_elements]
        sensor_y_abs = self.sensor_positions_cart[:, 1]  # [num_elements]
        sensor_theta = self.sensor_positions_cart[:, 2]  # [num_elements]

        # 计算距离 [roi_size, roi_size, num_elements]
        x_diff = x_grid.unsqueeze(-1) - sensor_x_abs
        y_diff = y_grid.unsqueeze(-1) - sensor_y_abs
        distances = torch.sqrt(x_diff ** 2 + y_diff ** 2)

        # 时间索引（MATLAB: round(distance / c0 / dt) + 1，转为Python 0-based索引）
        time_indices = torch.round(distances / self.c0 / self.dt).long()

        # 计算方向权重（匹配MATLAB逻辑）
        # sensor_normal = sensor_theta + pi (传感器朝向中心的法向量)
        sensor_normal = sensor_theta + np.pi  # [num_elements]

        # 从传感器指向像素的角度
        angle_to_point = torch.atan2(y_diff, x_diff)  # [roi_size, roi_size, num_elements]

        # 角度差
        angle_diff = angle_to_point - sensor_normal.unsqueeze(0).unsqueeze(0)

        # 归一化到[-pi, pi]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))

        # 权重：max(0, cos(angle_diff))
        weights = torch.clamp(torch.cos(angle_diff), min=0.0)

        # 创建有效掩码（时间索引在有效范围内）
        valid_mask = (time_indices >= 0) & (time_indices < self.time_samples)

        self.register_buffer('time_indices', time_indices)
        self.register_buffer('weights', weights)
        self.register_buffer('valid_mask', valid_mask)

    def forward(self, sinogram, output_type='both', norm_type='clamp'):
        """
        前向传播

        Args:
            sinogram: [B, 1, num_elements, time_samples] - limited-view正弦图
            output_type: 'das_only', 'pixel_only', or 'both'
            norm_type: 'clamp' or 'abs'

        Returns:
            - if 'das_only': das_reconstructed [B, 1, roi_size, roi_size]
            - if 'pixel_only': pixel_interp [B, num_elements, roi_size, roi_size]
            - if 'both': (das_reconstructed, pixel_interp)
        """
        B = sinogram.shape[0]
        device = sinogram.device

        # 确保预计算张量在正确设备上
        time_indices = self.time_indices.to(device)
        weights = self.weights.to(device)
        valid_mask = self.valid_mask.to(device)

        # 初始化输出
        das_reconstructed = torch.zeros(B, 1, self.roi_size, self.roi_size, device=device)
        if output_type in ['pixel_only', 'both']:
            pixel_interp = torch.zeros(B, self.num_elements, self.roi_size, self.roi_size, device=device)

        # 提取正弦图数据 [B, num_elements, time_samples]
        valid_sinogram = sinogram[:, 0, :, :]

        # 创建索引用于高级索引
        batch_indices = torch.arange(B, device=device)[:, None, None, None]
        sensor_indices = torch.arange(self.num_elements, device=device)[None, None, None, :]

        # 扩展维度
        batch_indices = batch_indices.expand(B, self.roi_size, self.roi_size, self.num_elements)
        sensor_indices = sensor_indices.expand(B, self.roi_size, self.roi_size, self.num_elements)
        time_indices_exp = time_indices.unsqueeze(0).expand(B, -1, -1, -1)

        # 使用掩码防止越界访问
        time_indices_clamped = torch.clamp(time_indices_exp, 0, self.time_samples - 1)

        # 矢量化采样 [B, roi_size, roi_size, num_elements]
        sampled_values = valid_sinogram[batch_indices, sensor_indices, time_indices_clamped]

        # 应用有效性掩码（时间索引超出范围的设为0）
        sampled_values = sampled_values * valid_mask.unsqueeze(0)

        # 应用方向权重
        weighted_values = sampled_values * weights.unsqueeze(0)

        # DAS重建：沿传感器维度求和
        if output_type in ['das_only', 'both']:
            das_result = weighted_values.sum(dim=3)  # [B, roi_size, roi_size]
            das_reconstructed[:, 0, :, :] = das_result

        # 像素插值
        if output_type in ['pixel_only', 'both']:
            pixel_interp = weighted_values.permute(0, 3, 1, 2)  # [B, num_elements, roi_size, roi_size]

        # 归一化DAS结果
        if output_type in ['das_only', 'both']:
            das_reconstructed = self._normalize_reconstruction(das_reconstructed, norm_type)

        # 返回结果
        if output_type == 'das_only':
            return das_reconstructed
        elif output_type == 'pixel_only':
            return pixel_interp
        elif output_type == 'both':
            return das_reconstructed, pixel_interp
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

    def _normalize_reconstruction(self, reconstruction, norm_type='clamp'):
        """归一化重建结果"""
        if norm_type == 'clamp':
            reconstruction = torch.clamp(reconstruction, min=0)
        elif norm_type == 'abs':
            reconstruction = torch.abs(reconstruction)

        # 按batch归一化
        B = reconstruction.shape[0]
        flat_recon = reconstruction.view(B, -1)
        max_vals, _ = flat_recon.max(dim=1, keepdim=True)

        # 避免除零
        epsilon = 1e-8
        max_vals = torch.where(max_vals > epsilon, max_vals, torch.ones_like(max_vals))
        normalized = flat_recon / max_vals

        return normalized.view_as(reconstruction)

class DASAndPixelInterpolator(nn.Module):
    """Optimized module for DAS reconstruction and pixel interpolation"""
    def __init__(self, num_elements=128, Nx=256, Ny=256, dx=0.15e-3, dy=0.15e-3, 
                 c0=1500, fs=80e6, element_pitch=0.3e-3, time_samples=4096, device='cuda'):
        super().__init__()
        self.num_elements = num_elements
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.c0 = c0
        self.fs = fs
        self.dt = 1 / fs
        self.element_pitch = element_pitch
        self.time_samples = time_samples
        self.device = device if torch.cuda.is_available() else 'cpu'

        # Precompute geometry, time indices, and weights
        self._precompute_geometry()
        self._precompute_time_indices_and_weights()

    def _precompute_geometry(self):
        """Precompute sensor positions on device"""
        element_spacing_grid = round(self.element_pitch / self.dx)
        sensor_y_position = 0  # 0-based index for top row
        array_width_grid = (self.num_elements - 1) * element_spacing_grid
        sensor_x_start = round((self.Nx - array_width_grid) / 2)

        sensor_positions = []
        for i in range(self.num_elements):
            x_pos = sensor_x_start + i * element_spacing_grid
            if 0 <= x_pos < self.Nx:
                sensor_positions.append([x_pos, sensor_y_position])

        self.sensor_positions = torch.tensor(sensor_positions, dtype=torch.float32, device=self.device)

    def _precompute_time_indices_and_weights(self):
        """Precompute time indices and weights for each pixel-sensor pair using vectorization"""
        # Create grid of pixel coordinates
        ix = torch.arange(self.Nx, dtype=torch.float32, device=self.device)
        iy = torch.arange(self.Ny, dtype=torch.float32, device=self.device)
        x_pos = ix * self.dx  # [Nx]
        y_pos = iy * self.dy  # [Ny]
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')  # [Ny, Nx]

        # Sensor coordinates
        sensor_x = self.sensor_positions[:, 0] * self.dx  # [num_elements]
        sensor_y = self.sensor_positions[:, 1] * self.dy  # [num_elements]

        # Broadcast to compute distances for all pixels and sensors
        x_diff = x_grid.unsqueeze(-1) - sensor_x  # [Ny, Nx, num_elements]
        y_diff = y_grid.unsqueeze(-1) - sensor_y  # [Ny, Nx, num_elements]
        distances = torch.sqrt(x_diff**2 + y_diff**2)  # [Ny, Nx, num_elements]
        time_indices = torch.round(distances / self.c0 / self.dt).long()  # [Ny, Nx, num_elements]
        weights = torch.clamp(torch.cos(torch.atan2(x_diff, y_diff)), min=0.0)  # [Ny, Nx, num_elements]
        
        self.register_buffer('time_indices', time_indices)
        self.register_buffer('weights', weights)
        
    def forward(self, sinogram, output_type='both', norm_type='clamp'):
        """
        sinogram: [B, 1, num_elements, time_samples]
        output_type: 'das_only' or 'both'
        returns:
            - if output_type='das_only': das_reconstructed [B, 1, Ny, Nx]
            - if output_type='both': (das_reconstructed [B, 1, Ny, Nx], pixel_interp [B, num_elements, Ny, Nx])
        """
        B, _, num_elements, time_samples = sinogram.shape
        device = sinogram.device

        # Ensure precomputed tensors are on the correct device
        time_indices = self.time_indices.to(device)
        weights = self.weights.to(device)
        # valid_mask = self.valid_mask.to(device)

        # Initialize outputs
        das_reconstructed = torch.zeros(B, 1, self.Ny, self.Nx, device=device)
        if output_type == 'both':
            pixel_interp = torch.zeros(B, num_elements, self.Ny, self.Nx, device=device)

        # Vectorize over sensors
        valid_sinogram = sinogram[:, 0, :, :]  # [B, num_elements, time_samples]
        
        # 创建批次索引用于高级索引
        batch_indices = torch.arange(B, device=device)[:, None, None, None]  # [B, 1, 1, 1]
        sensor_indices = torch.arange(num_elements, device=device)[None, None, None, :]  # [1, 1, 1, num_elements]

        # 扩展索引以匹配维度
        batch_indices = batch_indices.expand(B, self.Ny, self.Nx, num_elements)
        sensor_indices = sensor_indices.expand(B, self.Ny, self.Nx, num_elements)
        time_indices = time_indices.unsqueeze(0).expand(B, -1, -1, -1)
        
        # 矢量化采样 - 使用高级索引
        sampled_values = valid_sinogram[batch_indices, sensor_indices, time_indices]  # [B, Ny, Nx, num_valid_sensors]
        
        # 应用权重和掩码
        weighted_values = sampled_values * weights.unsqueeze(0)
        
        # DAS重建：沿传感器维度求和
        if output_type in ['das_only', 'both']:
            das_result = weighted_values.sum(dim=3)  # [B, Ny, Nx]
            das_reconstructed[:, 0, :, :] = das_result
        
        # 像素插值
        if output_type in ['pixel_only', 'both']:
            # 将有效传感器的结果映射回原始传感器索引
            for orig_sensor_idx in range(num_elements):
                pixel_interp[:, orig_sensor_idx, :, :] = weighted_values[:, :, :, orig_sensor_idx]
        
        # 归一化DAS结果
        if output_type in ['das_only', 'both']:
            das_reconstructed = self._normalize_reconstruction(das_reconstructed, norm_type)
        
        # 返回结果
        if output_type == 'das_only':
            return das_reconstructed
        elif output_type == 'pixel_only':
            return pixel_interp
        elif output_type == 'both':
            return das_reconstructed, pixel_interp
        else:
            raise ValueError(f"Unknown output_type: {output_type}. Choose from ['das_only', 'pixel_only', 'both']")

    def _normalize_reconstruction(self, reconstruction, norm_type='clamp'):
        """Efficient normalization with numerical stability"""
        if norm_type == 'clamp':
            reconstruction = torch.clamp(reconstruction, min=0)
        elif norm_type == 'abs':
            reconstruction = torch.abs(reconstruction)

        # 矢量化归一化
        B = reconstruction.shape[0]
        flat_recon = reconstruction.view(B, -1)  # [B, H*W]
        max_vals, _ = flat_recon.max(dim=1, keepdim=True)  # [B, 1]
        
        # 避免除零，添加小的epsilon
        epsilon = 1e-8
        max_vals = torch.where(max_vals > epsilon, max_vals, torch.ones_like(max_vals))
        normalized = flat_recon / max_vals
        
        return normalized.view_as(reconstruction)

# 使用示例
if __name__ == "__main__":
    # 创建模型
    model = DASAndPixelInterpolator_MSOT(
        num_elements_full=128,
        limited_view_range=(31, 95),  # 对应MATLAB的32:95
        Nx=408, Ny=408, roi_size=128,
        dx=0.2e-3, dy=0.2e-3, c0=1500, fs=40e6,
        inner_radius=40.6e-3,
        arc_angle=270, arc_start_angle=135,
        time_samples=2030,
        device='cuda'
    )

    # 测试输入：[B, 1, 64, 2030]
    batch_size = 2
    sinogram = torch.randn(batch_size, 1, 64, 2030, device='cuda')

    # 前向传播
    das_recon, pixel_interp = model(sinogram, output_type='both')

    print(f"\n输出形状:")
    print(f"DAS重建: {das_recon.shape}")  # [2, 1, 128, 128]
    print(f"像素插值: {pixel_interp.shape}")  # [2, 64, 128, 128]