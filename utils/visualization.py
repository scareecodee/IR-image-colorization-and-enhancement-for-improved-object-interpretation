"""
Visualization Utilities for Thermal Colorization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
import torch
from PIL import Image
import cv2


class Visualizer:
    """
    Visualization utilities for monitoring training and results
    """
    
    def __init__(self, output_dir='outputs/visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_history(self, history, save_path=None):
        """
        Plot training loss history
        
        Args:
            history: Dictionary with training history
            save_path: Path to save the figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        plot_configs = [
            ('generator_loss', 'Generator Loss', 'G Loss'),
            ('discriminator_loss', 'Discriminator Loss', 'D Loss'),
            ('content_loss', 'Content Loss', 'Content Loss'),
            ('adversarial_loss', 'Adversarial Loss', 'Adv Loss'),
            ('perceptual_loss', 'Perceptual Loss', 'Perceptual Loss'),
            ('tv_loss', 'TV Loss', 'TV Loss')
        ]
        
        for idx, (key, title, label) in enumerate(plot_configs):
            if key in history:
                ax = axes[idx]
                ax.plot(history[key], label=label, linewidth=2)
                ax.set_title(title, fontsize=12)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(label)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'training_history.png'), 
                       dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def visualize_samples(self, thermal, pred, target, num_samples=4,
                          save_path=None, show_metrics=True):
        """
        Visualize colorization results with comparison
        
        Args:
            thermal: Thermal input images (B, 1, H, W)
            pred: Predicted RGB images (B, 3, H, W)
            target: Ground truth RGB images (B, 3, H, W)
            num_samples: Number of samples to show
            save_path: Path to save the figure
            show_metrics: Whether to show metrics
        """
        # Convert to numpy and denormalize
        if torch.is_tensor(thermal):
            thermal = thermal.cpu().numpy()
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
        
        # Denormalize from [-1, 1] to [0, 1]
        def denormalize(img):
            if img.max() > 1:
                return (img + 1) / 2
            return img
        
        num_samples = min(num_samples, len(thermal))
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Thermal
            thermal_img = denormalize(thermal[i, 0])
            axes[i, 0].imshow(thermal_img, cmap='gray')
            axes[i, 0].set_title(f'Input Thermal {i+1}', fontsize=12)
            axes[i, 0].axis('off')
            
            # Predicted
            pred_img = np.clip(np.transpose(denormalize(pred[i]), (1, 2, 0)), 0, 1)
            axes[i, 1].imshow(pred_img)
            axes[i, 1].set_title(f'Colorized {i+1}', fontsize=12)
            axes[i, 1].axis('off')
            
            # Ground Truth
            target_img = np.clip(np.transpose(denormalize(target[i]), (1, 2, 0)), 0, 1)
            axes[i, 2].imshow(target_img)
            axes[i, 2].set_title(f'Ground Truth {i+1}', fontsize=12)
            axes[i, 2].axis('off')
            
            # Add metrics if requested
            if show_metrics:
                from .metrics import ImageQualityMetrics
                psnr = ImageQualityMetrics.compute_psnr(pred[i], target[i])
                ssim = ImageQualityMetrics.compute_ssim(pred[i], target[i])
                
                axes[i, 1].text(0.5, -0.1, f'PSNR: {psnr:.2f} dB, SSIM: {ssim:.3f}',
                               transform=axes[i, 1].transAxes,
                               ha='center', va='top', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'colorization_samples.png'),
                       dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def compare_methods(self, thermal, results_dict, num_samples=4, save_path=None):
        """
        Compare multiple methods side by side
        
        Args:
            thermal: Thermal input images
            results_dict: Dictionary {method_name: images}
            num_samples: Number of samples to show
            save_path: Path to save the figure
        """
        if torch.is_tensor(thermal):
            thermal = thermal.cpu().numpy()
        
        num_samples = min(num_samples, len(thermal))
        num_methods = len(results_dict)
        
        fig, axes = plt.subplots(num_samples, num_methods + 1, 
                                figsize=(4*(num_methods+1), 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        def denormalize(img):
            if img.max() > 1:
                return (img + 1) / 2
            return img
        
        for i in range(num_samples):
            # Thermal
            thermal_img = denormalize(thermal[i, 0])
            axes[i, 0].imshow(thermal_img, cmap='gray')
            axes[i, 0].set_title('Input Thermal', fontsize=12)
            axes[i, 0].axis('off')
            
            # Each method
            for j, (method_name, images) in enumerate(results_dict.items()):
                img = images[i]
                if torch.is_tensor(img):
                    img = img.cpu().numpy()
                
                img_vis = np.clip(np.transpose(denormalize(img), (1, 2, 0)), 0, 1)
                axes[i, j+1].imshow(img_vis)
                axes[i, j+1].set_title(method_name, fontsize=12)
                axes[i, j+1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'method_comparison.png'),
                       dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def visualize_patches(self, thermal_patch, rgb_patch, pred_patch=None,
                          save_path=None):
        """
        Visualize individual patches with details
        """
        fig, axes = plt.subplots(1, 3 if pred_patch is not None else 2,
                                figsize=(12, 4))
        
        # Thermal
        thermal_img = thermal_patch[0] if len(thermal_patch.shape) == 3 else thermal_patch
        axes[0].imshow(thermal_img, cmap='gray')
        axes[0].set_title('Thermal Patch', fontsize=12)
        axes[0].axis('off')
        
        # RGB
        rgb_img = np.transpose(rgb_patch, (1, 2, 0))
        axes[1].imshow(np.clip(rgb_img, 0, 1))
        axes[1].set_title('RGB Patch', fontsize=12)
        axes[1].axis('off')
        
        # Predicted
        if pred_patch is not None:
            pred_img = np.transpose(pred_patch, (1, 2, 0))
            axes[2].imshow(np.clip(pred_img, 0, 1))
            axes[2].set_title('Predicted Patch', fontsize=12)
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_metric_comparison(self, metrics_dict, save_path=None):
        """
        Plot comparison of metrics across different methods
        
        Args:
            metrics_dict: Dictionary {method_name: {metric_name: value}}
            save_path: Path to save the figure
        """
        methods = list(metrics_dict.keys())
        metrics = list(metrics_dict[methods[0]].keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        
        x = np.arange(len(methods))
        width = 0.6
        
        for idx, metric in enumerate(metrics):
            values = [metrics_dict[m][metric] for m in methods]
            
            ax = axes[idx]
            bars = ax.bar(x, values, width, color=sns.color_palette("husl", len(methods)))
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'metric_comparison.png'),
                       dpi=150, bbox_inches='tight')
        
        plt.close()


def plot_training_history(history, save_path=None):
    """Wrapper function for quick plotting"""
    visualizer = Visualizer()
    visualizer.plot_training_history(history, save_path)
    return visualizer


def compare_results(thermal, pred, target, save_path=None):
    """Wrapper function for quick comparison"""
    visualizer = Visualizer()
    visualizer.visualize_samples(thermal, pred, target, 
                                num_samples=min(4, len(thermal)),
                                save_path=save_path)
    return visualizer