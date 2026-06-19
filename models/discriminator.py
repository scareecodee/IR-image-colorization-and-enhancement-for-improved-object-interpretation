"""
Discriminator Network with Spectral Normalization
Input: RGB (3) + Thermal (1) = 4 channels
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator with Spectral Normalization
    Input: RGB (3 channels) + Thermal (1 channel) = 4 channels
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
        # Concatenate RGB with Thermal as condition
        x = torch.cat([x, condition], dim=1)
        return self.model(x)