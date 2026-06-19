"""
Complete Training Pipeline for Thermal-to-RGB Colorization
"""

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import cv2
from collections import defaultdict


class ColorizationTrainer:
    """
    Trainer for thermal-to-RGB colorization with GAN
    """
    def __init__(self, generator, discriminator, dataloaders, config):
        self.generator = generator
        self.discriminator = discriminator
        self.dataloaders = dataloaders
        
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Loss functions
        self.loss_fn = CompositeLoss(
            lambda_adv=config.get('lambda_adv', 0.03),
            lambda_perceptual=config.get('lambda_perceptual', 1.0),
            lambda_tv=config.get('lambda_tv', 1.0),
            lambda_semantic=config.get('lambda_semantic', 0.5)
        )
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            generator.parameters(), 
            lr=config.get('g_lr', 2e-4),
            betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            discriminator.parameters(),
            lr=config.get('d_lr', 2e-4),
            betas=(0.5, 0.999)
        )
        
        # Learning rate schedulers
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer_G, 
            step_size=config.get('lr_step', 5), 
            gamma=config.get('lr_gamma', 0.5)
        )
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_D,
            step_size=config.get('lr_step', 5),
            gamma=config.get('lr_gamma', 0.5)
        )
        
        # TensorBoard
        self.writer = SummaryWriter(config.get('log_dir', './logs'))
        
        # Checkpoint directory
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Move to device
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        
        # For stability
        self.scale_grad = config.get('scale_grad', 0.01)
        
    def train_step(self, batch):
        """
        Single training step
        """
        thermal = batch['thermal'].to(self.device)
        rgb = batch['rgb'].to(self.device)
        
        # ---------- Train Generator ----------
        self.optimizer_G.zero_grad()
        
        # Generate colorized images
        outputs = self.generator(thermal)
        fake_rgb = outputs['fine']
        coarse_rgb = outputs['coarse']
        
        # Discriminator predictions on fake images
        disc_fake = self.discriminator(fake_rgb, thermal)
        
        # Generator loss
        gen_losses = self.loss_fn.generator_loss(
            disc_fake, fake_rgb, rgb, coarse_rgb,
            outputs.get('seg_logits'), None
        )
        
        gen_losses['total'].backward()
        self.optimizer_G.step()
        
        # ---------- Train Discriminator ----------
        self.optimizer_D.zero_grad()
        
        # Real predictions
        disc_real = self.discriminator(rgb, thermal)
        
        # Fake predictions (detach to not update generator)
        disc_fake = self.discriminator(fake_rgb.detach(), thermal)
        
        # Discriminator loss
        d_loss = self.loss_fn.discriminator_loss(disc_real, disc_fake)
        d_loss.backward()
        self.optimizer_D.step()
        
        # Return losses
        return {
            'g_total': gen_losses['total'].item(),
            'g_content': gen_losses['content'].item(),
            'g_adv': gen_losses['adversarial'].item(),
            'g_perceptual': gen_losses['perceptual'].item(),
            'g_tv': gen_losses['tv'].item(),
            'd_loss': d_loss.item()
        }
    
    def validate(self, epoch):
        """
        Validation step
        """
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['val'], desc="Validating"):
                thermal = batch['thermal'].to(self.device)
                rgb = batch['rgb'].to(self.device)
                
                # Generate
                outputs = self.generator(thermal)
                fake_rgb = outputs['fine']
                
                # Discriminator predictions
                disc_fake = self.discriminator(fake_rgb, thermal)
                
                # Compute losses
                gen_losses = self.loss_fn.generator_loss(
                    disc_fake, fake_rgb, rgb, outputs.get('coarse')
                )
                
                # Accumulate
                for key, value in gen_losses.items():
                    if key != 'total':
                        val_losses[key] += value.item()
                
                num_batches += 1
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in val_losses.items()}
        return avg_losses
    
    def train(self, num_epochs, start_epoch=0):
        """
        Full training loop
        """
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, num_epochs):
            # Training phase
            self.generator.train()
            self.discriminator.train()
            
            epoch_losses = defaultdict(float)
            
            pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                losses = self.train_step(batch)
                
                # Accumulate losses
                for key, value in losses.items():
                    epoch_losses[key] += value
                
                # Update progress bar
                pbar.set_postfix({
                    'G': f"{losses['g_total']:.3f}",
                    'D': f"{losses['d_loss']:.3f}"
                })
            
            # Average epoch losses
            num_batches = len(self.dataloaders['train'])
            avg_epoch_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            
            # Update learning rates
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Log to TensorBoard
            for key, value in avg_epoch_losses.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            # Validation
            val_losses = self.validate(epoch)
            for key, value in val_losses.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Print summary
            print(f"\nEpoch {epoch+1}:")
            print(f"  Train Loss: G={avg_epoch_losses.get('g_total', 0):.3f}, "
                  f"D={avg_epoch_losses.get('d_loss', 0):.3f}")
            print(f"  Val Loss: {sum(val_losses.values()):.3f}")
            
            # Visualize samples
            if (epoch + 1) % 5 == 0:
                self.visualize_samples(epoch)
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
                'optimizer_G_state_dict': self.optimizer_G.state_dict(),
                'optimizer_D_state_dict': self.optimizer_D.state_dict(),
                'val_loss': sum(val_losses.values())
            }
            
            # Save best model
            val_total = sum(val_losses.values())
            if val_total < best_val_loss:
                best_val_loss = val_total
                torch.save(checkpoint, os.path.join(
                    self.checkpoint_dir, 'best_model.pth'
                ))
            
            # Save regular checkpoint
            torch.save(checkpoint, os.path.join(
                self.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'
            ))
        
        self.writer.close()
    
    def visualize_samples(self, epoch, num_samples=4):
        """
        Visualize sample predictions
        """
        self.generator.eval()
        
        # Get a batch from validation set
        val_iter = iter(self.dataloaders['val'])
        batch = next(val_iter)
        
        thermal = batch['thermal'][:num_samples].to(self.device)
        rgb = batch['rgb'][:num_samples].to(self.device)
        
        with torch.no_grad():
            outputs = self.generator(thermal)
            fake_rgb = outputs['fine']
        
        # Convert to numpy for visualization
        thermal_np = thermal.cpu().numpy()
        fake_np = fake_rgb.cpu().numpy()
        rgb_np = rgb.cpu().numpy()
        
        # Log to TensorBoard
        for i in range(min(num_samples, len(thermal))):
            # Thermal
            thermal_img = thermal_np[i, 0]
            self.writer.add_image(f'Samples/{i}/Thermal', 
                                 thermal_img, epoch, dataformats='HW')
            
            # Fake RGB
            fake_img = np.transpose(fake_np[i], (1, 2, 0))
            fake_img = np.clip(fake_img, 0, 1)
            self.writer.add_image(f'Samples/{i}/Colorized', 
                                 fake_img, epoch, dataformats='HWC')
            
            # Real RGB
            real_img = np.transpose(rgb_np[i], (1, 2, 0))
            real_img = np.clip(real_img, 0, 1)
            self.writer.add_image(f'Samples/{i}/GroundTruth', 
                                 real_img, epoch, dataformats='HWC')


class SuperResolutionTrainer:
    """
    Trainer for Super-Resolution module
    """
    def __init__(self, model, dataloaders, config):
        self.model = model
        self.dataloaders = dataloaders
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('lr', 1e-4)
        )
        
        # Checkpoint directory
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints_sr')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.model = self.model.to(self.device)
    
    def train(self, num_epochs):
        """
        Train super-resolution model
        """
        for epoch in range(num_epochs):
            self.model.train()
            
            epoch_loss = 0
            pbar = tqdm(self.dataloaders['train'], desc=f"SR Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                # Low-resolution input (thermal)
                lr = batch['thermal'].to(self.device)
                # High-resolution output (RGB or thermal upsampled)
                hr = batch['rgb'].to(self.device)
                
                # Forward
                out_stage1, out_stage2 = self.model(lr)
                
                # Loss (both stages)
                loss1 = self.l1_loss(out_stage1, hr)
                loss2 = self.l1_loss(out_stage2, hr)
                loss = loss1 + loss2
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            
            avg_loss = epoch_loss / len(self.dataloaders['train'])
            print(f"Epoch {epoch+1}: SR Loss = {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.checkpoint_dir, f'sr_epoch_{epoch+1}.pth'))