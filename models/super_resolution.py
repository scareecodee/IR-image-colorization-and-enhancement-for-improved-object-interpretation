"""
Super-Resolution Module for Thermal Images
Upscales thermal band (100m -> 30m)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperResolutionNetwork(nn.Module):
    """
    Simple super-resolution network for thermal band
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_channels=64,
                 num_blocks=8, scale_factor=2):
        super(SuperResolutionNetwork, self).__init__()
        
        self.scale_factor = scale_factor
        
        # Feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            self._residual_block(base_channels) for _ in range(num_blocks)
        ])
        
        # Upsampling
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
        
    def _residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x):
        residual = x
        
        feat = self.initial(x)
        for block in self.blocks:
            feat = feat + block(feat)
        
        out = self.upsample(feat)
        return out