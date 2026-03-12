import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.FD_UNet import Conv2dBatchNorm, FD_UNet, U_Net
from models.ConvUAM import ConvUAM, CBAM, Block, LayerNorm
from models.DASandLUT import DASAndPixelInterpolator_MSOT, DASAndPixelInterpolator
from timm.layers import trunc_normal_


class Sinogram_Adapter(nn.Module):
    def __init__(self, sino_height=128, sino_width=4096,
                 target_size=256, target_channels=64,
                 adapter_type='convnext', block_depth=2):
        super().__init__()
        self.sino_height = sino_height
        self.sino_width = sino_width
        self.target_size = target_size
        self.target_channels = target_channels
        self.adapter_type = adapter_type

        if adapter_type == 'convnext':
            self.scale1_proj = nn.Conv2d(1, 32, kernel_size=3, padding=1)
            self.scale1_blocks = nn.Sequential(*[Block(32) for _ in range(block_depth)])

            self.scale2_proj = nn.Conv2d(1, 32, kernel_size=5, padding=2)
            self.scale2_blocks = nn.Sequential(*[Block(32) for _ in range(block_depth)])

            self.scale3_proj = nn.Conv2d(1, 32, kernel_size=7, padding=3)
            self.scale3_blocks = nn.Sequential(*[Block(32) for _ in range(block_depth)])

            # 特征融合
            self.fusion_conv = nn.Conv2d(32 * 3, 64, kernel_size=1)
            self.fusion_blocks = nn.Sequential(*[Block(64) for _ in range(block_depth)])

            # 自适应池化
            self.adaptive_pool = nn.AdaptiveAvgPool2d((target_size, target_size))

            # 通道调整和细化
            self.refine_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.refine_block1 = Block(128)
            self.attention = CBAM(128)
            self.refine_conv2 = nn.Conv2d(128, target_channels, kernel_size=1)
            self.refine_block2 = Block(target_channels)

        elif adapter_type == 'cnn_enhanced':
            # CNN增强版本：多尺度特征提取 + 残差连接
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
            self.adaptive_pool = nn.AdaptiveAvgPool2d((target_size, target_size))

            # 残差细化模块
            self.residual_refine = nn.Sequential(
                Conv2dBatchNorm(64, 128, kernel_size=3, padding=1, activation='relu'),
                Conv2dBatchNorm(128, 256, kernel_size=3, padding=1, activation='relu'),
                Conv2dBatchNorm(256, 128, kernel_size=3, padding=1, activation='relu'),
                Conv2dBatchNorm(128, target_channels, kernel_size=1, padding=0, activation='relu'),
            )

            # 残差连接
            self.attention = CBAM(64)
            self.residual_conv = nn.Conv2d(64, target_channels, kernel_size=1)

    def forward(self, x):
        if self.adapter_type == 'convnext':
            # ConvNeXt多尺度分支
            s1 = self.scale1_proj(x)
            s1 = self.scale1_blocks(s1)
            s2 = self.scale2_proj(x)
            s2 = self.scale2_blocks(s2)
            s3 = self.scale3_proj(x)
            s3 = self.scale3_blocks(s3)

            # 拼接融合
            fused = torch.cat([s1, s2, s3], dim=1)  # [B, 96, H, W]
            fused = self.fusion_conv(fused)  # [B, 64, H, W]
            fused = self.fusion_blocks(fused)

            # 池化到目标尺寸
            pooled = self.adaptive_pool(fused)  # [B, 64, target_size, target_size]

            # 通道细化
            refined = self.refine_conv1(pooled)
            refined = self.refine_block1(refined)
            refined = self.attention(refined)  # 应用注意力机制
            refined = self.refine_conv2(refined)
            output = self.refine_block2(refined)

        elif self.adapter_type == 'cnn_enhanced':
            # CNN增强版本：多尺度特征提取
            scale1 = self.scale1_encoder(x)  # [B, 64, H, W]
            scale2 = self.scale2_encoder(x)  # [B, 64, H, W]
            scale3 = self.scale3_encoder(x)  # [B, 64, H, W]

            # 特征融合
            fused = torch.cat([scale1, scale2, scale3], dim=1)  # [B, 192, H, W]
            fused = self.feature_fusion(fused)  # [B, 64, H, W]

            # 自适应池化
            pooled = self.adaptive_pool(fused)  # [B, 64, target_size, target_size]

            # 残差细化
            refined = self.residual_refine(pooled)  # [B, target_channels, target_size, target_size]
            residual = self.residual_conv(self.attention(pooled))  # [B, target_channels, target_size, target_size]
            # 残差连接
            output = refined + residual  # [B, target_channels, target_size, target_size]

        return output


class Adjoint_Network(nn.Module):
    """增强的PAT重建网络, 支持三种模式
    1. type1: 仅使用sinogram特征
    2. type2: 使用sinogram特征和DAS重建
    3. type3: 使用sinogram特征、DAS重建和LUT
    4. type4: 使用DAS和LUT
    5. type5: 仅DAS
    """

    def __init__(self, reconstruction_type='type4', adapter_type='convnext', unet_type='convuam',
                 DAS_type='linear',
                 sino_height=64, sino_width=2030, target_size=128,
                 inner_channel=64):
        super().__init__()

        self.reconstruction_type = reconstruction_type
        self.num_elements = sino_height  # 探测器数量

        # Sinogram编码器
        if reconstruction_type not in ['type4', 'type5']:
            target_channels = 1 if self.reconstruction_type == 'type3' else self.num_elements
            self.sinogram_adapter = Sinogram_Adapter(
                sino_height, sino_width, target_size, target_channels, adapter_type
            )

        # DAS重建模块（在forward中计算）
        # if reconstruction_type != 'type1':
        if DAS_type == 'linear':
            self.das_and_pixel = DASAndPixelInterpolator()
        elif DAS_type == 'MSOT':
            self.das_and_pixel = DASAndPixelInterpolator_MSOT()

        # 确定U-Net输入通道数
        if reconstruction_type == 'type1':
            unet_in_ch = target_channels  # 只有sinogram特征
        elif reconstruction_type == 'type2':
            unet_in_ch = target_channels + 1  # sinogram特征 + DAS
        elif reconstruction_type == 'type3':
            unet_in_ch = target_channels + 1 + self.num_elements  # sinogram特征 + DAS + pixel interpolation
        elif reconstruction_type == 'type4':
            unet_in_ch = 1 + self.num_elements  # DAS + pixel_interpolation
        elif reconstruction_type == 'type5':
            unet_in_ch = 1  # 仅DAS
        else:
            raise ValueError(f"Unknown reconstruction type: {reconstruction_type}")

        self.unet = ConvUAM(in_chans=unet_in_ch, out_chans=1, with_cbam=True)

        self.sigmoid = nn.Sigmoid()


    def forward(self, sinogram, out_with_das=False, out_with_feature=False):
        """
        sinogram: [B, 1, num_elements, time_samples]
        """
        # 1. Sinogram编码
        if self.reconstruction_type not in ['type4', 'type5']:
            sino_features = self.sinogram_adapter(sinogram)  # [B, target_channels, 256, 256]
            features_to_concat = [sino_features]
        else:
            features_to_concat = []

        if self.reconstruction_type == 'type2' or self.reconstruction_type == 'type5':
            das_recon = self.das_and_pixel(sinogram, output_type='das_only')
            features_to_concat.append(das_recon)
        elif self.reconstruction_type == 'type3' or self.reconstruction_type == 'type4':
            das_recon, pixel_interp = self.das_and_pixel(sinogram, output_type='both')
            features_to_concat.append(das_recon)
            features_to_concat.append(pixel_interp)

        # 4. 特征拼接
        if len(features_to_concat) > 1:
            combined_features = torch.cat(features_to_concat, dim=1)
        else:
            combined_features = features_to_concat[0]

        # 5. U-Net重建
        if out_with_feature:
            reconstructed, f_list = self.unet(combined_features, out_with_feature=True)  # [B, 1, 256, 256]
        else:
            reconstructed = self.unet(combined_features)

        outputs = [self.sigmoid(reconstructed)]
        if self.reconstruction_type != 'type1' and out_with_das:
            outputs.append(das_recon)
        if out_with_feature:
            outputs.append(f_list)
        return tuple(outputs)

    def get_intermediate_results(self, sinogram):
        """获取中间结果用于分析"""
        results = {}

        if self.reconstruction_type not in ['type4', 'type5']:
            # Sinogram特征
            sino_features = self.sinogram_adapter(sinogram)
            results['sinogram_features'] = sino_features

        if self.reconstruction_type == 'type2' or self.reconstruction_type == 'type5':
            das_recon = self.das_and_pixel(sinogram, output_type='das_only')
            results['das_reconstruction'] = das_recon
        elif self.reconstruction_type == 'type3' or self.reconstruction_type == 'type4':
            das_recon, pixel_interp = self.das_and_pixel(sinogram, output_type='both')
            results['das_reconstruction'] = das_recon
            results['pixel_interpolation'] = pixel_interp

        return results

    def get_das(self, sinogram, norm_type='clamp'):
        das_recon = self.das_and_pixel(sinogram, output_type='das_only', norm_type=norm_type)

        return das_recon

    def get_das_lut(self, sinogram, norm_type='clamp'):
        das_recon, lut_rst = self.das_and_pixel(sinogram, output_type='both', norm_type=norm_type)

        return das_recon, lut_rst


class ConvNextEncoder(nn.Module):
    """ConvNext风格的编码器，将输入编码到64通道"""

    def __init__(self, in_channels, out_channels=64, num_blocks=3, drop_path_rate=0.1):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        # 创建ConvNext blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]
        self.blocks = nn.Sequential(*[
            Block(dim=out_channels, drop_path=dp_rates[i], layer_scale_init_value=1e-6)
            for i in range(num_blocks)
        ])

        self.norm = LayerNorm(out_channels, eps=1e-6, data_format="channels_first")

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class ConvNextDecoder(nn.Module):
    """ConvNext风格的解码器，从192通道解码到1通道"""

    def __init__(self, in_channels=192, num_blocks=4, drop_path_rate=0.1, with_FF=True):
        super().__init__()
        # 逐步减少通道数
        self.with_FF = with_FF
        channels = [192, 96, 48, 24, 1]

        self.layers = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]

        for i in range(len(channels) - 1):
            if i < num_blocks:
                # ConvNext blocks
                if with_FF:
                    cur_ch = channels[i] + 8
                else:
                    cur_ch = channels[i]
                layer = nn.Sequential(
                    Block(dim=(cur_ch), drop_path=dp_rates[i], layer_scale_init_value=1e-6),
                    nn.Conv2d((cur_ch), channels[i + 1], kernel_size=1, padding=0),
                    LayerNorm(channels[i + 1], eps=1e-6, data_format="channels_first") if i < len(
                        channels) - 2 else nn.Identity()
                )
            else:
                # 最后的卷积层
                layer = nn.Conv2d((channels[i] + 8), channels[i + 1], kernel_size=1, padding=0)

            self.layers.append(layer)

    def forward(self, x, x_feature):
        idx = 0
        for layer in self.layers:
            if x_feature != None and self.with_FF:
                ff = F.interpolate(x_feature[idx], size=x.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, ff], dim=1)  # 融合特征
            x = layer(x)
            idx = idx + 1
        return x


class Hybrid_Network(nn.Module):
    """使用ConvNext Block的双输入模型"""

    def __init__(self, reconstruction_type='type4', adapter_type='convnext', unet_type='convuam',
                 DAS_type='linear', with_FF = True, 
                 sino_height=64, sino_width=2030, target_size=128,
                 inner_channel=32,
                 encoder_blocks=3, decoder_blocks=4, drop_path_rate=0.):
        super().__init__()
        self.with_FF = with_FF

        # ConvUAM -----------------------------------------------
        self.adjoint_network = Adjoint_Network(reconstruction_type=reconstruction_type,
                                               adapter_type=adapter_type, unet_type=unet_type,
                                               DAS_type=DAS_type,
                                               sino_height=sino_height, sino_width=sino_width,
                                               target_size=target_size, inner_channel=inner_channel)

        # regCNN -----------------------------------------------
        # 两个编码器分别处理x_res和x_hat
        self.encoder_x1 = ConvNextEncoder(1, 96, encoder_blocks, drop_path_rate)
        self.encoder_x2 = ConvNextEncoder((sino_height + 1), 96, encoder_blocks, drop_path_rate)

        # CNN解码器网络
        self.decoder = ConvNextDecoder(192, decoder_blocks, drop_path_rate, with_FF = with_FF)

        self.sigmoid = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2, sinogram):
        if self.with_FF:
            x_hat, x_feature = self.adjoint_network(sinogram, out_with_feature=True)  # [B, 1, 256, 256]

        # 编码三个输入
        feat_x1 = self.encoder_x1(x1)  # [B, 96, H, W]
        feat_x2 = self.encoder_x2(x2)  # [B, 96, H, W]

        # 连接特征
        concat_feat = torch.cat([feat_x1, feat_x2], dim=1)  # [B, 192, H, W]

        # 通过CNN网络获得1通道特征
        if not self.with_FF:
            x_feature = None
        feature = self.decoder(concat_feat, x_feature)  # [B, 1, H, W]

        # 1x1卷积 + 残差连接
        # output = self.alpha(feature) + x_hat
        output = self.sigmoid(feature)

        if not self.with_FF:
            x_hat = None

        return output, x_hat

    def get_das(self, sinogram, norm_type='clamp'):
        return self.adjoint_network.get_das(sinogram, norm_type=norm_type)

    def get_das_lut(self, sinogram, norm_type='clamp'):
        return self.adjoint_network.get_das_lut(sinogram, norm_type=norm_type)