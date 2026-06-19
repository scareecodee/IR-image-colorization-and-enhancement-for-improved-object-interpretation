"""
Discriminator Networks for GAN Training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator with Spectral Normalization
    """
    def __init__(self, input_channels=4, base_channels=64, n_layers=3):
        super(PatchGANDiscriminator, self).__init__()
        
        layers = []
        
        # First layer
        layers.append(self._conv_block(
            input_channels, base_channels, 
            norm=False, activation=True
        ))
        
        # Hidden layers
        curr_channels = base_channels
        for i in range(1, n_layers):
            next_channels = min(curr_channels * 2, 512)
            layers.append(self._conv_block(
                curr_channels, next_channels,
                stride=2 if i < n_layers - 1 else 1,
                norm=True, activation=True
            ))
            curr_channels = next_channels
        
        # Output layer
        layers.append(self._conv_block(
            curr_channels, 1,
            stride=1,
            norm=False, activation=False
        ))
        
        self.model = nn.Sequential(*layers)
        
    def _conv_block(self, in_channels, out_channels, stride=2, 
                   norm=True, activation=True):
        layers = []
        
        # Convolution with spectral normalization
        conv = nn.Conv2d(in_channels, out_channels, 4, stride=stride, padding=1)
        conv = nn.utils.spectral_norm(conv)
        layers.append(conv)
        
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        if activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x, condition):
        # Concatenate RGB/Thermal as condition
        x = torch.cat([x, condition], dim=1)
        return self.model(x)


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for better quality assessment
    Evaluates at multiple resolutions
    """
    def __init__(self, input_channels=4, base_channels=64):
        super(MultiScaleDiscriminator, self).__init__()
        
        self.disc1 = PatchGANDiscriminator(input_channels, base_channels, n_layers=3)
        self.disc2 = PatchGANDiscriminator(input_channels, base_channels*2, n_layers=4)
        self.disc3 = PatchGANDiscriminator(input_channels, base_channels*4, n_layers=5)
        
    def forward(self, x, condition):
        # Downsample for different scales
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x3 = F.interpolate(x, scale_factor=0.25, mode='bilinear')
        cond2 = F.interpolate(condition, scale_factor=0.5, mode='bilinear')
        cond3 = F.interpolate(condition, scale_factor=0.25, mode='bilinear')
        
        out1 = self.disc1(x, condition)
        out2 = self.disc2(x2, cond2)
        out3 = self.disc3(x3, cond3)
        
        return out1, out2, out3