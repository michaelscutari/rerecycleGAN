import torch
import torch.nn as nn
from torch.nn import functional as F

class PatchGAN(nn.Module):
    def __init__(self, in_channels=3, base_features=64, num_downs=3):
        super(PatchGAN, self).__init__()

        features = base_features

        # encoder
        self.down_blocks = nn.ModuleList()

        for i in range(num_downs):
            if i == 0:
                self.down_blocks.append(self.down_block(in_channels, features, normalize=False))
            else:
                self.down_blocks.append(self.down_block(features, features * 2))
            features *= 2

        # output layer
        self.output = nn.Sequential(
            nn.Conv2d(features, 1, kernel_size=3, stride=1, padding=1),
        ) 

    def forward(self, x):
        for down_block in self.down_blocks:
            x = down_block(x)
        return self.output(x)        
    

    # patch size for 432, 240 image calculation
    # 432 -> 216 -> 108 -> 54
    # 240 -> 120 -> 60 -> 30
        
    @staticmethod
    def down_block(in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
class MultiScale(nn.Module):
    '''
    Multi-scale discriminator.
    Implemented using multiple PatchGAN discriminators.
    '''
    def __init__(self, in_channels=3, base_features=64, num_downs=4, num_scales=3):
        super(MultiScale, self).__init__()

        self.discriminators = nn.ModuleList()
        for _ in range(num_scales):
            self.discriminators.append(PatchGAN(in_channels, base_features, num_downs))

    # forward pass using multiple discriminators and bilinear interpolation
    def forward(self, x):
        outputs = []
        input_down = x
        for discriminator in self.discriminators:
            outputs.append(discriminator(input_down))
            input_down = F.interpolate(input_down, scale_factor=0.5, mode='bilinear', align_corners=False)
        return outputs
        

        
