import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------- 2D U-Net -------------------------------
class Conv2dBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation='relu', stride=1):
        super(Conv2dBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x, 0.2)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation):
        super(DownBlock, self).__init__()
        self.conv1 = Conv2dBatchNorm(in_channels, out_channels, kernel_size, padding, activation)
        self.conv2 = Conv2dBatchNorm(out_channels, out_channels, kernel_size, padding, activation)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_pooled = self.pool(x)
        return x_pooled, x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = Conv2dBatchNorm(in_channels, out_channels, kernel_size, padding, activation)
        self.conv2 = Conv2dBatchNorm(out_channels, out_channels, kernel_size, padding, activation)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class U_Net(nn.Module):
    def __init__(self, in_channel=1, out_channel=1, inner_channel=32, norm_groups=16, 
                 channel_mults=(1, 2, 4, 8), attn_res=[16, 32, 64, 128], 
                 res_blocks=1, dropout=0.1, image_size=256):
        super(U_Net, self).__init__()
        filters = inner_channel
        kernel_size = 3
        padding = 1
        activation = 'relu'
        
        self.initial = Conv2dBatchNorm(in_channel, filters, kernel_size=kernel_size, padding=padding, activation=activation)
        
        self.down1 = DownBlock(filters, filters*2, kernel_size=kernel_size, padding=padding, activation=activation)
        self.down2 = DownBlock(filters*2, filters*4, kernel_size=kernel_size, padding=padding, activation=activation)
        self.down3 = DownBlock(filters*4, filters*8, kernel_size=kernel_size, padding=padding, activation=activation)
        self.down4 = DownBlock(filters*8, filters*16, kernel_size=kernel_size, padding=padding, activation=activation)
        
        self.bridge = Conv2dBatchNorm(filters*16, filters*32, kernel_size=kernel_size, padding=padding, activation=activation)
        
        self.up1 = UpBlock(filters*32, filters*16, kernel_size=kernel_size, padding=padding, activation=activation)
        self.up2 = UpBlock(filters*16, filters*8, kernel_size=kernel_size, padding=padding, activation=activation)
        self.up3 = UpBlock(filters*8, filters*4, kernel_size=kernel_size, padding=padding, activation=activation)
        self.up4 = UpBlock(filters*4, filters*2, kernel_size=kernel_size, padding=padding, activation=activation)
        
        self.final = nn.Conv2d(filters*2, out_channel, kernel_size=1, padding=0)
        
    def forward(self, x):
        input = x
        x1 = self.initial(x)
        x2, shortcut1 = self.down1(x1)
        x3, shortcut2 = self.down2(x2)
        x4, shortcut3 = self.down3(x3)
        x5, shortcut4 = self.down4(x4)
        
        x = self.bridge(x5)
        
        x = self.up1(x, shortcut4)
        x = self.up2(x, shortcut3)
        x = self.up3(x, shortcut2)
        x = self.up4(x, shortcut1)
        
        x = self.final(x)
        return x # + input

# ------------------------------- FD U-Net -------------------------------
class FDBlock(nn.Module):
    """全密集块"""
    def __init__(self, f_in, f_out, k, kernel_size=3):
        super().__init__()
        self.layers = nn.ModuleList()
        padding = kernel_size // 2
        added_ch = f_in
        for i in range(f_in, f_out, k):
            self.layers.append(nn.Sequential(
                Conv2dBatchNorm(added_ch, f_in, kernel_size=1, padding=0, activation='relu'),
                Conv2dBatchNorm(f_in, k, kernel_size=kernel_size, padding=padding, activation='relu')
            ))
            added_ch += k  # 通道累积增长

    def forward(self, x):
        for layer in self.layers:
            shortcut = x
            out = layer(x)
            x = torch.cat([out, shortcut], dim=1)
        return x

class FD_DownBlock(nn.Module):
    """FDUNet专用下采样块"""
    def __init__(self, in_ch, out_ch, with_att=False, with_aspp=False):
        super().__init__()
        self.fd = FDBlock(in_ch, out_ch, k=in_ch//4)
        self.down = nn.Sequential(
            Conv2dBatchNorm(out_ch, out_ch, kernel_size=1, padding=0, activation='relu'),
            Conv2dBatchNorm(out_ch, out_ch, kernel_size=3, padding=1, 
                            activation='relu', stride=2)
            # nn.MaxPool2d(2)
        )
        if with_att:
            self.cbam = CBAM(out_ch)
        if with_aspp:
            self.aspp = ASPP(out_ch, out_ch)

    def forward(self, x):
        shortcut = self.fd(x)
        x = self.down(shortcut)
        if hasattr(self, 'cbam'):
            x = self.cbam(x)
        if hasattr(self, 'aspp'):
            x = self.aspp(x)
        return x, shortcut

class FD_UpBlock(nn.Module):
    """FDUNet专用上采样块"""
    def __init__(self, in_ch, out_ch, with_att=False):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        )
        self.fd = nn.Sequential(
            Conv2dBatchNorm(in_ch, out_ch, kernel_size=1, padding=0, activation='relu'),
            FDBlock(out_ch, out_ch*2, k=out_ch//4)
        )
        if with_att:
            self.cbam = CBAM(out_ch*2)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.fd(x)
        if hasattr(self, 'cbam'):
            x = self.cbam(x)
        return x

class FD_UNet(nn.Module):
    """整合后的FDUNet架构"""
    def __init__(self, in_channel=1, out_channel=1, inner_channel=32):
        super().__init__()
        filters = inner_channel

        self.adjust_channel = Conv2dBatchNorm(in_channel, out_channel, kernel_size=1, padding=0, activation='relu')

        self.initial = Conv2dBatchNorm(in_channel, filters, kernel_size=3, padding=1, activation='relu')

        # 编码器
        self.down1 = FD_DownBlock(filters, filters*2)
        self.down2 = FD_DownBlock(filters*2, filters*4)
        self.down3 = FD_DownBlock(filters*4, filters*8)
        self.down4 = FD_DownBlock(filters*8, filters*16)

        # 桥接层
        self.bridge = nn.Sequential(
            FDBlock(filters*16, filters*32, k=filters*16//4)
            # Conv2dBatchNorm(filters*16, filters*32, kernel_size=3, padding=1, activation='relu'),
            # nn.ConvTranspose2d(filters*32, filters*16, kernel_size=2, stride=2)
        )

        # 解码器
        self.up1 = FD_UpBlock(filters*32, filters*8)
        self.up2 = FD_UpBlock(filters*16, filters*4)
        self.up3 = FD_UpBlock(filters*8, filters*2)
        self.up4 = FD_UpBlock(filters*4, filters)

        # 最终输出
        self.final = nn.Sequential(
            Conv2dBatchNorm(filters*2, filters, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2d(filters, out_channel, kernel_size=1)
        )
        # self.residual = nn.Conv2d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x0 = self.initial(x)
        
        # 编码器
        x1, s1 = self.down1(x0)
        x2, s2 = self.down2(x1)
        x3, s3 = self.down3(x2)
        x4, s4 = self.down4(x3)

        # 桥接层
        b = self.bridge(x4)

        # 解码器
        d1 = self.up1(b, s4)
        d2 = self.up2(d1, s3)
        d3 = self.up3(d2, s2)
        d4 = self.up4(d3, s1)

        # 残差连接
        out = self.final(d4)
        out = out + self.adjust_channel(x)
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(32, 1, 32, 64, 64)
    
    vit = U_Net(inner_channel=32)
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))