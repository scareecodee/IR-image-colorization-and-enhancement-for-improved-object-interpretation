"""
Evaluation Metrics for Image Quality and Task Performance
"""

import torch
import numpy as np
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from torchvision.models import inception_v3
import torch.nn.functional as F


class ImageQualityMetrics:
    """
    Image quality evaluation metrics
    """
    
    @staticmethod
    def compute_psnr(pred, target):
        """
        Peak Signal-to-Noise Ratio
        """
        # Convert to numpy if tensor
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
        
        # Ensure same shape
        if pred.shape != target.shape:
            # Resize if needed
            if len(pred.shape) == 3:
                target = cv2.resize(target.transpose(1, 2, 0), 
                                   (pred.shape[2], pred.shape[1]))
                target = target.transpose(2, 0, 1)
            else:
                target = cv2.resize(target, (pred.shape[1], pred.shape[0]))
        
        # Compute PSNR
        psnr_value = psnr(target, pred, data_range=1.0)
        return psnr_value
    
    @staticmethod
    def compute_ssim(pred, target):
        """
        Structural Similarity Index
        """
        # Convert to numpy if tensor
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
        
        # Ensure same shape
        if pred.shape != target.shape:
            if len(pred.shape) == 3:
                target = cv2.resize(target.transpose(1, 2, 0),
                                   (pred.shape[2], pred.shape[1]))
                target = target.transpose(2, 0, 1)
            else:
                target = cv2.resize(target, (pred.shape[1], pred.shape[0]))
        
        # Compute SSIM (multi-channel)
        if len(pred.shape) == 3:
            ssim_value = 0
            for c in range(pred.shape[0]):
                ssim_value += ssim(pred[c], target[c], data_range=1.0)
            ssim_value /= pred.shape[0]
        else:
            ssim_value = ssim(pred, target, data_range=1.0)
        
        return ssim_value


class FIDCalculator:
    """
    Fréchet Inception Distance for evaluating generated image quality
    """
    
    def __init__(self):
        # Load InceptionV3 for feature extraction
        self.inception = inception_v3(pretrained=True, transform_input=True)
        self.inception = self.inception.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.inception.eval()
        
    def _get_features(self, images):
        """
        Extract Inception features from images
        """
        # Resize to 299x299 (Inception input size)
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear')
        
        # Normalize to [0, 1] if needed
        if images.max() > 1:
            images = images / 255.0
        
        with torch.no_grad():
            features = self.inception(images)
        return features
    
    def compute_fid(self, pred, target):
        """
        Compute FID score between predicted and target images
        """
        # Get features
        pred_features = self._get_features(pred)
        target_features = self._get_features(target)
        
        # Calculate mean and covariance
        mu1 = pred_features.mean(0)
        mu2 = target_features.mean(0)
        
        sigma1 = torch.cov(pred_features.T)
        sigma2 = torch.cov(target_features.T)
        
        # Compute FID
        diff = mu1 - mu2
        covmean = torch.sqrt(sigma1 @ sigma2)
        
        # Calculate trace
        fid = diff @ diff + torch.trace(sigma1 + sigma2 - 2 * covmean)
        
        return fid.item()


class ObjectDetectionMetrics:
    """
    Metrics for downstream object detection tasks
    """
    
    @staticmethod
    def compute_map(pred_boxes, pred_scores, pred_labels, 
                    gt_boxes, gt_labels, iou_threshold=0.5):
        """
        Compute mAP for object detection
        """
        # This is a simplified implementation
        # In practice, use COCO evaluator or similar
        
        # Placeholder - actual implementation depends on detection framework
        return 0.0


def evaluate_model(model, dataloader, device='cuda'):
    """
    Comprehensive evaluation of the colorization model
    """
    model.eval()
    
    psnr_values = []
    ssim_values = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            thermal = batch['thermal'].to(device)
            rgb = batch['rgb'].to(device)
            
            # Generate colorized images
            outputs = model(thermal)
            fake_rgb = outputs['fine']
            
            # Compute metrics
            psnr_value = ImageQualityMetrics.compute_psnr(fake_rgb, rgb)
            ssim_value = ImageQualityMetrics.compute_ssim(fake_rgb, rgb)
            
            psnr_values.append(psnr_value)
            ssim_values.append(ssim_value)
    
    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'psnr_std': np.std(psnr_values),
        'ssim_std': np.std(ssim_values)
    }