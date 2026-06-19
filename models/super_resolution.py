"""
Super-Resolution Module with Edge Enhancement
Implements multi-scale super-resolution for thermal images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEnhancementModule(nn.Module):
    """
    Edge enhancement module for preserving structural details
    """
    def __init__(self, in_channels):
        super(EdgeEnhancementModule, self).__init__()
        
        # Sobel-like edge detection
        self.edge_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.edge_conv.weight.data = self._init_sobel_weights(in_channels)
        
        # Edge enhancement
        self.enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        
    def _init_sobel_weights(self, in_channels):
        """Initialize with Sobel kernel for edge detection"""
        sobel_kernel = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ], dtype=torch.float32)
        
        weights = torch.zeros(in_channels, in_channels, 3, 3)
        for i in range(in_channels):
            weights[i, i] = sobel_kernel
        return weights
    
    def forward(self, x):
        edges = self.edge_conv(x)
        edges = torch.abs(edges)  # Magnitude of edges
        enhanced = self.enhance(edges)
        return x + enhanced  # Add enhanced edges to original


class MultiScaleFeatureExtractor(nn.Module):
    """
    Multi-scale feature extraction for remote sensing images
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureExtractor, self).__init__()
        
        # Different scale convolutions
        self.scale1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1)
        self.scale3 = nn.Conv2d(in_channels, out_channels//4, kernel_size=3, padding=1)
        self.scale5 = nn.Conv2d(in_channels, out_channels//4, kernel_size=5, padding=2)
        self.scale7 = nn.Conv2d(in_channels, out_channels//4, kernel_size=7, padding=3)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        s1 = self.scale1(x)
        s3 = self.scale3(x)
        s5 = self.scale5(x)
        s7 = self.scale7(x)
        
        fused = torch.cat([s1, s3, s5, s7], dim=1)
        return self.fusion(fused)


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block for feature refinement
    """
    def __init__(self, channels, growth_rate=32):
        super(ResidualDenseBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2*growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3*growth_rate, growth_rate, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(channels + 4*growth_rate, channels, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.relu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.relu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5  # Residual connection


class SuperResolutionNetwork(nn.Module):
    """
    Main Super-Resolution Network with Edge Enhancement
    """
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, 
                 num_blocks=8, scale_factor=2):
        super(SuperResolutionNetwork, self).__init__()
        
        self.scale_factor = scale_factor
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale feature extraction
        self.ms_feature = MultiScaleFeatureExtractor(base_channels, base_channels)
        
        # Edge enhancement
        self.edge_enhance = EdgeEnhancementModule(base_channels)
        
        # Residual dense blocks
        self.blocks = nn.ModuleList([
            ResidualDenseBlock(base_channels) for _ in range(num_blocks)
        ])
        
        # Upsampling (pixel shuffle)
        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # Initial features
        feat = self.initial(x)
        
        # Multi-scale features
        feat = self.ms_feature(feat)
        
        # Edge enhancement
        feat = self.edge_enhance(feat)
        
        # Residual dense blocks
        for block in self.blocks:
            feat = block(feat)
        
        # Upsample
        out = self.upsample(feat)
        
        return out


class MultiStageSuperResolution(nn.Module):
    """
    Two-stage super-resolution for thermal images
    Stage 1: 1x to 2x
    Stage 2: 2x to 4x
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(MultiStageSuperResolution, self).__init__()
        
        self.stage1 = SuperResolutionNetwork(
            in_channels, out_channels, base_channels=64, 
            num_blocks=6, scale_factor=2
        )
        
        self.stage2 = SuperResolutionNetwork(
            in_channels, out_channels, base_channels=128, 
            num_blocks=8, scale_factor=2
        )
        
    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        return x1, x2