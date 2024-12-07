
import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class _ResidualBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(_ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels_out)
        )

        if channels_in != channels_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=1, stride=1),
                nn.InstanceNorm2d(channels_out)
            )

    def forward(self, x):
        residual = x
        out = self.block(x)
        if hasattr(self, 'shortcut'):
            residual = self.shortcut(x)
        out += residual
        return out

class ResNet(nn.Module):
    '''
    ResNet Generator
    Based on the generator used in RecycleGAN.
    '''
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=6):
        super(ResNet, self).__init__()
        
        # Initial downsampling
        self.downsampling = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Second downsampling layer
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Residual blocks
        list_residual_blocks = [_ResidualBlock(256, 256) for _ in range(num_residual_blocks)]
        self.residual_blocks = nn.Sequential(*list_residual_blocks)

        # Final upsampling
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.downsampling(x)
        out = self.residual_blocks(out)
        out = self.upsampling(out)
        return out

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_features=64):
        super(UNet, self).__init__()

        # Encoder blocks
        self.enc1 = self.down_block(in_channels, base_features)
        self.enc2 = self.down_block(base_features, base_features * 2)
        self.enc3 = self.down_block(base_features * 2, base_features * 4)
        self.enc4 = self.down_block(base_features * 4, base_features * 8)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_features * 8, base_features * 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_features * 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
        )

        # Upsampling layers
        self.up1 = self.up_block(base_features * 16, base_features * 8)
        self.up2 = self.up_block(base_features * 8, base_features * 4)
        self.up3 = self.up_block(base_features * 4, base_features * 2)
        self.up4 = self.up_block(base_features * 2, base_features)

        # Decoder blocks
        self.dec1 = self.dec_block(base_features * 16, base_features * 8, dropout=True)
        self.dec2 = self.dec_block(base_features * 8, base_features * 4, dropout=True)
        self.dec3 = self.dec_block(base_features * 4, base_features * 2)
        self.dec4 = self.dec_block(base_features * 2, base_features)

        # Final output layer
        self.output = nn.Sequential(
            nn.Conv2d(base_features, out_channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)                               # [B, base_features, H, W]
        x2 = self.enc2(F.max_pool2d(x1, kernel_size=2)) # [B, base_features*2, H/2, W/2]
        x3 = self.enc3(F.max_pool2d(x2, kernel_size=2)) # [B, base_features*4, H/4, W/4]
        x4 = self.enc4(F.max_pool2d(x3, kernel_size=2)) # [B, base_features*8, H/8, W/8]

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(x4, kernel_size=2)) # [B, base_features*16, H/16, W/16]

        # Decoder
        d1 = self.up1(bottleneck)       # [B, base_features*8, H/8, W/8]
        d1 = torch.cat([d1, x4], dim=1) # Concatenate with encoder feature
        d1 = self.dec1(d1)              # [B, base_features*8, H/8, W/8]

        d2 = self.up2(d1)               # [B, base_features*4, H/4, W/4]
        d2 = torch.cat([d2, x3], dim=1)
        d2 = self.dec2(d2)              # [B, base_features*4, H/4, W/4]

        d3 = self.up3(d2)               # [B, base_features*2, H/2, W/2]
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)              # [B, base_features*2, H/2, W/2]

        d4 = self.up4(d3)               # [B, base_features, H, W]
        d4 = torch.cat([d4, x1], dim=1)
        d4 = self.dec4(d4)              # [B, base_features, H, W]

        # Output layer
        out = self.output(d4)           # [B, out_channels, H, W]

        return out

    def down_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def dec_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )