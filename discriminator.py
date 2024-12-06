import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchGAN(nn.Module):
    def __init__(self, in_channels=3, base_features=64, num_downs=3):
        super(PatchGAN, self).__init__()

        layers = []
        features = base_features

        # First downsampling layer (no normalization)
        layers.append(self._down_block(in_channels, features, normalize=False))

        # Subsequent downsampling layers
        for _ in range(1, num_downs):
            layers.append(self._down_block(features, features * 2))
            features *= 2

        # Final convolution layer to produce the patch output
        layers.append(nn.Conv2d(features, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _down_block(in_channels, out_channels, normalize=True):
        """Creates a downsampling block with optional normalization."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator.
    Utilizes multiple PatchGAN discriminators at different image scales.
    """
    def __init__(self, in_channels=3, base_features=64, num_downs=3, num_scales=3):
        super(MultiScaleDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList([
            PatchGAN(in_channels, base_features, num_downs) for _ in range(num_scales)
        ])

    def forward(self, x):
        outputs = []
        current_input = x

        for discriminator in self.discriminators:
            outputs.append(discriminator(current_input))
            # Downsample the input for the next discriminator
            current_input = F.interpolate(
                current_input, scale_factor=0.5, mode='bilinear', align_corners=False
            )

        # Combine the outputs into a single loss value
        combined_loss = sum([torch.mean(output) for output in outputs]) / len(outputs)
        return combined_loss