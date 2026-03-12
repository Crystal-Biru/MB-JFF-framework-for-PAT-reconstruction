# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath  
from timm.models import register_model

class Block(nn.Module):
    r""" ConvNeXtV2 Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, with_cbam=False):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if with_cbam:
            self.cbam = CBAM(dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        # x = self.grn(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)

        if hasattr(self, 'cbam'):
            x = self.cbam(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x
    

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_stem=False, with_cbam=False):
        super(DownBlock, self).__init__()
        self.is_stem = is_stem
        if self.is_stem:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2),
                LayerNorm(out_ch, eps=1e-6, data_format="channels_first")
            )
        else:
            self.norm = LayerNorm(in_ch, eps=1e-6, data_format="channels_first")
            self.down = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)
        if with_cbam:
            self.cbam = CBAM(out_ch)

    def forward(self, x):
        if self.is_stem:
            x_final = self.down(x)
        else:
            x = self.norm(x)
            x_final = self.down(x)
        if hasattr(self, 'cbam'):
            x_final = self.cbam(x_final)
        return x_final, x

class UpBlock(nn.Module):
    """FDUNet专用上采样块"""
    def __init__(self, in_ch, out_ch, with_cbam=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        self.norm = LayerNorm(in_ch//2, eps=1e-6, data_format="channels_first")
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        if with_cbam:
            self.cbam = CBAM(out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.norm(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        if hasattr(self, 'cbam'):
            x = self.cbam(x)
        return x


class ConvUAM(nn.Module):
    r""" ConvUAM (U-Net + ConvNeXt + CBAM)
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, out_chans=1, depths=[3, 3, 9, 3], 
                 dims=[48, 96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 with_cbam = True):
        super().__init__()

        cbam_flag = with_cbam

        self.initial = nn.Conv2d(in_chans, dims[0], kernel_size=1, padding=0)

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = DownBlock(dims[0], dims[1], is_stem=False, with_cbam=cbam_flag)
        self.downsample_layers.append(stem)
        for i in range(1, 4):
            self.downsample_layers.append(DownBlock(dims[i], dims[i+1], with_cbam=cbam_flag))

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i+1], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.upsample_layers = nn.ModuleList()
        for i in range(4, 0, -1):
            self.upsample_layers.append(UpBlock(dims[i], dims[i-1], with_cbam=cbam_flag))

        self.stages2 = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        depths2 = [depths[0], depths[1], depths[3], depths[2]]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths2))] 
        cur = 0
        for i in range(3, -1, -1):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths2[i])]
            )
            self.stages2.append(stage)
            cur += depths2[i]

        self.norm = LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        # nn.LayerNorm(dims[0], eps=1e-6) # final norm layer
        self.head = nn.Conv2d(dims[0], out_chans, kernel_size=1, padding=0)# nn.Linear(dims[-1], num_classes)

        self.adjust_ch = Adust_Block(out_ch=8)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x, out_with_feature=False):
        x = self.initial(x)
        s_list = []
        f_list = [] if out_with_feature else None
        for i in range(4):
            x, s = self.downsample_layers[i](x)
            x = self.stages[i](x)
            s_list.append(s)
        for i in range(4):
            x = self.upsample_layers[i](x, s_list[3-i])
            x = self.stages2[i](x)

            if out_with_feature:
                f = self.adjust_ch(x)
                f_list.append(f)

        return self.norm(x) if not out_with_feature else (self.norm(x), f_list)

    def forward(self, x, out_with_feature=False):
        # input = x
        if out_with_feature:
            x, f_list = self.forward_features(x, out_with_feature=True)
            x = self.head(x)
            return x, f_list  # 返回特征图列表
        else:
            x = self.forward_features(x, out_with_feature=False)
            x = self.head(x)
            return x # + input


class Adust_Block(nn.Module):
    def __init__(self, out_ch=8):
        super(Adust_Block, self).__init__()
        self.out_channels = out_ch
        self.conv = None  # 延迟初始化

    def forward(self, x):
        # 如果第一次 forward，还没定义 conv，则根据输入 x 的 channel 初始化
        in_channels = x.shape[1]  # 读取输入的通道数
        self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, padding=0)
        # 把 conv 移到 x 所在设备
        self.conv.to(x.device)
        return self.conv(x)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(32, 1, 128, 128)
    
    vit = ConvUAM(in_chans=1)
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))