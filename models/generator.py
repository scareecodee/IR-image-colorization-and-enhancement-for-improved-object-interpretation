"""
Generator Network for Thermal-to-RGB Colorization
Based on Coarse-to-Fine Architecture with Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for feature recalibration
    """
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


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for focusing on important regions
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.conv(concat))
        return x * attention


class ResidualAttentionBlock(nn.Module):
    """
    Residual block with channel and spatial attention
    """
    def __init__(self, channels):
        super(ResidualAttentionBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(channels)
        
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        
        # Apply attention
        out = self.channel_attn(out)
        out = self.spatial_attn(out)
        
        return self.relu(out + residual)


class SemanticGuidedModule(nn.Module):
    """
    Semantic guidance module for consistent color mapping
    """
    def __init__(self, in_channels, num_classes=6):
        super(SemanticGuidedModule, self).__init__()
        
        # Semantic segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, num_classes, 1)
        )
        
        # Class-specific color mapping
        self.color_embedding = nn.Embedding(num_classes, 3)
        
    def forward(self, x):
        seg_logits = self.seg_head(x)
        seg_probs = F.softmax(seg_logits, dim=1)
        
        # Class-wise weighted color mapping
        colors = self.color_embedding.weight  # (num_classes, 3)
        colors = colors.view(1, -1, 1, 1)  # (1, num_classes*3, 1, 1)
        
        # Expand probabilities to match color dimension
        seg_probs_expanded = seg_probs.unsqueeze(2)  # (B, C, 1, H, W)
        seg_probs_expanded = seg_probs_expanded.expand(-1, -1, 3, -1, -1)  # (B, C, 3, H, W)
        seg_probs_expanded = seg_probs_expanded.reshape(seg_probs.shape[0], -1, 
                                                        seg_probs.shape[2], 
                                                        seg_probs.shape[3])
        
        # Weighted sum of class colors
        semantic_colors = torch.sum(seg_probs_expanded * colors, dim=1)
        
        return seg_logits, semantic_colors


class CoarseToFineGenerator(nn.Module):
    """
    Coarse-to-Fine Generator with Semantic Guidance
    Architecture for high-quality thermal-to-RGB translation
    """
    def __init__(self, in_channels=1, out_channels=3, base_channels=64, 
                 num_res_blocks=9, use_semantic=True):
        super(CoarseToFineGenerator, self).__init__()
        
        self.use_semantic = use_semantic
        
        # ---------- Coarse Sub-network ----------
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
        
        # Coarse residual blocks
        self.coarse_blocks = nn.ModuleList([
            ResidualAttentionBlock(base_channels*4) for _ in range(num_res_blocks)
        ])
        
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
        
        # ---------- Fine Sub-network ----------
        self.fine_initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels + out_channels, base_channels, 7),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        
        self.fine_down = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*2),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels*4),
            nn.ReLU(inplace=True),
        )
        
        # Fine residual blocks with attention
        self.fine_blocks = nn.ModuleList([
            ResidualAttentionBlock(base_channels*4) for _ in range(num_res_blocks)
        ])
        
        # Semantic guidance module
        if use_semantic:
            self.semantic_module = SemanticGuidedModule(base_channels*4)
        
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
        # ---------- Coarse Path ----------
        coarse_feat = self.coarse_down(x)
        for block in self.coarse_blocks:
            coarse_feat = block(coarse_feat)
        coarse_out = self.coarse_up(coarse_feat)
        
        # ---------- Fine Path ----------
        # Concatenate input with coarse output
        fine_input = torch.cat([x, coarse_out], dim=1)
        fine_feat = self.fine_initial(fine_input)
        fine_feat = self.fine_down(fine_feat)
        
        # Semantic guidance
        seg_logits = None
        semantic_colors = None
        if self.use_semantic:
            seg_logits, semantic_colors = self.semantic_module(fine_feat)
        
        # Residual blocks
        for block in self.fine_blocks:
            fine_feat = block(fine_feat)
            
        # Apply semantic guidance if available
        if self.use_semantic and semantic_colors is not None:
            # Combine features with semantic colors
            semantic_colors_resized = F.interpolate(
                semantic_colors, size=fine_feat.shape[2:], mode='bilinear'
            )
            fine_feat = fine_feat + semantic_colors_resized
        
        fine_out = self.fine_up(fine_feat)
        
        return {
            'coarse': coarse_out,
            'fine': fine_out,
            'seg_logits': seg_logits,
            'semantic_colors': semantic_colors
        }