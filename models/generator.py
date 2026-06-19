"""
Generator Network for Thermal-to-RGB Colorization
Input: Single thermal band (Band 10)
Output: RGB (Bands 2, 3, 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with instance normalization"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + residual


class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return x * self.fc(x)


class AttentionResidualBlock(nn.Module):
    """Residual block with channel attention"""
    def __init__(self, channels):
        super(AttentionResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.attention = ChannelAttention(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.attention(out)
        return out + residual


class CoarseToFineGenerator(nn.Module):
    """
    Coarse-to-Fine Generator for thermal-to-RGB translation
    Input: 1 channel (thermal band 10)
    Output: 3 channels (RGB: bands 2, 3, 4)
    """
    
    def __init__(self, in_channels=1, out_channels=3, base_channels=64,
                 num_res_blocks=9, use_semantic=False):
        super(CoarseToFineGenerator, self).__init__()
        
        self.use_semantic = use_semantic
        
        # ========== COARSE SUB-NETWORK ==========
        # Downsampling
        self.coarse_down = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, 7),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.coarse_blocks = nn.ModuleList([
            ResidualBlock(base_channels*4) for _ in range(num_res_blocks)
        ])
        
        # Upsampling
        self.coarse_up = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 3, stride=2,
                              padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels*2, base_channels, 3, stride=2,
                              padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, 7),
            nn.Tanh()
        )
        
        # ========== FINE SUB-NETWORK ==========
        # Initial convolution
        self.fine_initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels + out_channels, base_channels, 7),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        # Downsampling
        self.fine_down = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks with attention
        self.fine_blocks = nn.ModuleList([
            AttentionResidualBlock(base_channels*4) for _ in range(num_res_blocks)
        ])
        
        # Upsampling
        self.fine_up = nn.Sequential(
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 3, stride=2,
                              padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(base_channels*2, base_channels, 3, stride=2,
                              padding=1, output_padding=1),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, 7),
            nn.Tanh()
        )
        
    def forward(self, x):
        # ========== COARSE PATH ==========
        coarse_feat = self.coarse_down(x)
        for block in self.coarse_blocks:
            coarse_feat = block(coarse_feat)
        coarse_out = self.coarse_up(coarse_feat)
        
        # ========== FINE PATH ==========
        # Concatenate input with coarse output
        fine_input = torch.cat([x, coarse_out], dim=1)
        fine_feat = self.fine_initial(fine_input)
        fine_feat = self.fine_down(fine_feat)
        
        # Residual blocks
        for block in self.fine_blocks:
            fine_feat = block(fine_feat)
        
        fine_out = self.fine_up(fine_feat)
        
        return {
            'coarse': coarse_out,
            'fine': fine_out
        }