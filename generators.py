
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    '''
    Residual Block
    Currently only used in the ResNetGenerator.
    '''
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        # layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.in1 = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.in2 = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)

        # 1x1 conv for skip connection
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, padding_mode='reflect')
        else:
            self.conv3 = None

        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, x):
        identity = x

        # first conv
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu1(out)

        # second conv
        out = self.conv2(out)
        out = self.in2(out)
        
        # if channel size differs
        if self.conv3:
            identity = self.conv3(identity)
        
        out += identity
        out = self.relu2(out)

        return out

class ResNet(nn.Module):
    '''
    ResNet Generator
    Based on the generator used in RecycleGAN.
    '''
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=6):
        super(ResNet, self).__init__()
        
        # initial downsampling
        self.downsampling = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # residual blocks
        list_residual_blocks = [ResidualBlock(128, 128) for _ in range(num_residual_blocks)]
        self.residual_blocks = nn.Sequential(*list_residual_blocks)

        # final upsampling
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=7, stride=2, padding=3, output_padding=1, padding_mode='reflect'),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.downsampling(x)
        out = self.residual_blocks(out)
        return self.upsampling(out)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_features=64, num_downs=4):
        super(UNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        features = base_features

        # encoder
        self.encoder = nn.ModuleList()
        for _ in range(num_downs):
            self.encoder.append(self.down_block(in_channels, features))
            in_channels = features
            features *= 2

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(features, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(features, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # decoder
        self.decoder = nn.ModuleList()
        for _ in range(num_downs):
            features //= 2 # decrease features by a factor of 2
            self.decoder.append(self.up_block(features * 2, features))

        # final conv
        self.output = nn.Sequential(
            nn.Conv2d(features, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.Tanh(),
        )

    def forward(self, x):
        # encoder
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        
        # bottleneck
        x = self.bottleneck(x)

        # decoder
        for up, skip in zip(self.decoder, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        
        # output
        return self.output(x)

        
        
    # self.down_block definition
    def down_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # reduce dimensions by a factor of 2
        )
        return block

    # self.up_block definition
    def up_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        return block
    