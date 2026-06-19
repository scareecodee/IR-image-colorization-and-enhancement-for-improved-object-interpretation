"""
Specialized Super-Resolution Training with Edge Enhancement
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import cv2


class EdgeAwareSRLoss(nn.Module):
    """
    Edge-aware loss for super-resolution
    """
    def __init__(self, edge_weight=0.1):
        super(EdgeAwareSRLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSE()
        self.edge_weight = edge_weight
        
        # Sobel filters for edge detection
        self.sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
    
    def forward(self, pred, target):
        # L1 loss
        l1_loss = self.l1_loss(pred, target)
        
        # Edge detection
        pred_edges = self._compute_edges(pred)
        target_edges = self._compute_edges(target)
        
        # Edge loss
        edge_loss = self.mse_loss(pred_edges, target_edges)
        
        return l1_loss + self.edge_weight * edge_loss
    
    def _compute_edges(self, x):
        """Compute edge magnitude"""
        # Handle multi-channel
        if x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)
        
        edge_x = torch.nn.functional.conv2d(x, self.sobel_x.to(x.device), padding=1)
        edge_y = torch.nn.functional.conv2d(x, self.sobel_y.to(x.device), padding=1)
        
        return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)


def train_super_resolution(model, dataloaders, config):
    """
    Training loop for super-resolution with detailed logging
    """
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = EdgeAwareSRLoss(edge_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-4))
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.get('lr_step', 10), gamma=0.5
    )
    
    # Training
    for epoch in range(config.get('num_epochs', 50)):
        model.train()
        total_loss = 0
        total_l1 = 0
        
        pbar = tqdm(dataloaders['train'], desc=f"SR Epoch {epoch+1}")
        for batch in pbar:
            lr = batch['thermal'].to(device)
            hr = batch['rgb'].to(device)
            
            # Forward pass
            stage1, stage2 = model(lr)
            
            # Compute losses
            loss1 = criterion(stage1, hr)
            loss2 = criterion(stage2, hr)
            loss = loss1 + loss2
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            total_l1 += torch.abs(stage2 - hr).mean().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'l1': f"{torch.abs(stage2 - hr).mean().item():.4f}"
            })
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloaders['train'])
        avg_l1 = total_l1 / len(dataloaders['train'])
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, L1 = {avg_l1:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, os.path.join(config.get('checkpoint_dir', './checkpoints_sr'),
                           f'sr_epoch_{epoch+1}.pth'))