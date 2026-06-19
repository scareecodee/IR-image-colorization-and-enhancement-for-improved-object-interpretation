"""
Evaluation Metrics for Image Quality Assessment
PSNR, SSIM, FID, and more
"""

import torch
import numpy as np
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2
from torchvision.models import inception_v3
import torch.nn.functional as F
from tqdm import tqdm


class ImageQualityMetrics:
    """
    Image quality evaluation metrics
    """
    
    @staticmethod
    def compute_psnr(pred, target, data_range=1.0):
        """
        Peak Signal-to-Noise Ratio
        
        Args:
            pred: Predicted image (tensor or numpy)
            target: Ground truth image (tensor or numpy)
            data_range: Data range (1.0 for normalized images)
        
        Returns:
            PSNR value in dB
        """
        # Convert to numpy if tensor
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
        
        # Ensure same shape
        if pred.shape != target.shape:
            if len(pred.shape) == 3:
                target = cv2.resize(
                    target.transpose(1, 2, 0),
                    (pred.shape[2], pred.shape[1])
                ).transpose(2, 0, 1)
            else:
                target = cv2.resize(target, (pred.shape[1], pred.shape[0]))
        
        # Compute PSNR
        try:
            psnr_value = psnr(target, pred, data_range=data_range)
        except:
            # Fallback manual calculation
            mse = np.mean((pred - target) ** 2)
            if mse == 0:
                psnr_value = float('inf')
            else:
                psnr_value = 20 * np.log10(data_range / np.sqrt(mse))
        
        return psnr_value
    
    @staticmethod
    def compute_ssim(pred, target, data_range=1.0, multichannel=True):
        """
        Structural Similarity Index
        
        Args:
            pred: Predicted image
            target: Ground truth image
            data_range: Data range
            multichannel: Whether image has multiple channels
        
        Returns:
            SSIM value
        """
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
        
        # Ensure same shape
        if pred.shape != target.shape:
            if len(pred.shape) == 3:
                target = cv2.resize(
                    target.transpose(1, 2, 0),
                    (pred.shape[2], pred.shape[1])
                ).transpose(2, 0, 1)
            else:
                target = cv2.resize(target, (pred.shape[1], pred.shape[0]))
        
        # Compute SSIM
        try:
            if len(pred.shape) == 3:
                # Multi-channel (RGB)
                ssim_value = 0
                for c in range(pred.shape[0]):
                    ssim_value += ssim(
                        pred[c], target[c],
                        data_range=data_range
                    )
                ssim_value /= pred.shape[0]
            else:
                ssim_value = ssim(pred, target, data_range=data_range)
        except:
            # Fallback
            if len(pred.shape) == 3:
                ssim_value = np.mean([
                    ssim(pred[c], target[c], data_range=data_range)
                    for c in range(pred.shape[0])
                ])
            else:
                ssim_value = ssim(pred, target, data_range=data_range)
        
        return ssim_value
    
    @staticmethod
    def compute_mse(pred, target):
        """Mean Squared Error"""
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
        
        return np.mean((pred - target) ** 2)
    
    @staticmethod
    def compute_mae(pred, target):
        """Mean Absolute Error"""
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
            target = target.cpu().numpy()
        
        return np.mean(np.abs(pred - target))
    
    @staticmethod
    def compute_lpips(pred, target):
        """
        LPIPS - Learned Perceptual Image Patch Similarity
        Requires: pip install lpips
        """
        try:
            import lpips
            lpips_fn = lpips.LPIPS(net='alex')
            
            if not torch.is_tensor(pred):
                pred = torch.FloatTensor(pred)
                target = torch.FloatTensor(target)
            
            # Ensure correct shape
            if len(pred.shape) == 3:
                pred = pred.unsqueeze(0)
                target = target.unsqueeze(0)
            
            with torch.no_grad():
                score = lpips_fn(pred, target)
            return score.item()
        except ImportError:
            print("LPIPS not installed. Install with: pip install lpips")
            return None


class FIDCalculator:
    """
    Fréchet Inception Distance for evaluating generated image quality
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.inception = None
        
    def _load_inception(self):
        """Load InceptionV3 model for feature extraction"""
        if self.inception is None:
            self.inception = inception_v3(
                pretrained=True,
                transform_input=True,
                aux_logits=False
            )
            self.inception = self.inception.to(self.device)
            self.inception.eval()
            
            # Remove the final pooling and classification layers
            self.inception.fc = torch.nn.Identity()
    
    def _get_features(self, images):
        """
        Extract Inception features from images
        
        Args:
            images: Tensor of images (B, C, H, W) in [0, 1] range
        
        Returns:
            Feature vectors
        """
        self._load_inception()
        
        # Ensure correct size (299x299)
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(
                images, size=(299, 299),
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize to [0, 1] if needed
        if images.max() > 1:
            images = images / 255.0
        
        images = images.to(self.device)
        
        with torch.no_grad():
            features = self.inception(images)
        
        return features.cpu().numpy()
    
    def compute_fid(self, pred, target):
        """
        Compute FID score between predicted and target images
        
        Args:
            pred: Predicted images (tensor or numpy)
            target: Target images (tensor or numpy)
        
        Returns:
            FID score
        """
        # Convert to tensor if needed
        if not torch.is_tensor(pred):
            pred = torch.FloatTensor(pred)
            target = torch.FloatTensor(target)
        
        # Ensure correct shape (B, C, H, W)
        if len(pred.shape) == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
        
        # Get features
        pred_features = self._get_features(pred)
        target_features = self._get_features(target)
        
        # Calculate statistics
        mu1 = np.mean(pred_features, axis=0)
        mu2 = np.mean(target_features, axis=0)
        
        sigma1 = np.cov(pred_features, rowvar=False)
        sigma2 = np.cov(target_features, rowvar=False)
        
        # Compute FID
        diff = mu1 - mu2
        
        # Calculate sqrt of product of covariances
        covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
        
        # Check for numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        # FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fid)


def compute_all_metrics(pred, target, data_range=1.0):
    """
    Compute all image quality metrics at once
    
    Args:
        pred: Predicted image
        target: Ground truth image
        data_range: Data range
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'psnr': ImageQualityMetrics.compute_psnr(pred, target, data_range),
        'ssim': ImageQualityMetrics.compute_ssim(pred, target, data_range),
        'mse': ImageQualityMetrics.compute_mse(pred, target),
        'mae': ImageQualityMetrics.compute_mae(pred, target)
    }
    
    return metrics


def evaluate_batch(model, dataloader, device='cuda', num_samples=None):
    """
    Evaluate model on a batch of data
    
    Args:
        model: Generator model
        dataloader: DataLoader with test data
        device: Device to run on
        num_samples: Number of samples to evaluate
    
    Returns:
        Dictionary with aggregated metrics
    """
    model.eval()
    
    all_psnr = []
    all_ssim = []
    all_mse = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if num_samples and idx >= num_samples:
                break
                
            thermal = batch['thermal'].to(device)
            rgb = batch['rgb'].to(device)
            
            outputs = model(thermal)
            pred = outputs['fine']
            
            # Compute metrics
            psnr_val = ImageQualityMetrics.compute_psnr(pred, rgb)
            ssim_val = ImageQualityMetrics.compute_ssim(pred, rgb)
            mse_val = ImageQualityMetrics.compute_mse(pred, rgb)
            
            all_psnr.append(psnr_val)
            all_ssim.append(ssim_val)
            all_mse.append(mse_val)
    
    results = {
        'psnr': {
            'mean': np.mean(all_psnr),
            'std': np.std(all_psnr),
            'min': np.min(all_psnr),
            'max': np.max(all_psnr)
        },
        'ssim': {
            'mean': np.mean(all_ssim),
            'std': np.std(all_ssim),
            'min': np.min(all_ssim),
            'max': np.max(all_ssim)
        },
        'mse': {
            'mean': np.mean(all_mse),
            'std': np.std(all_mse),
            'min': np.min(all_mse),
            'max': np.max(all_mse)
        },
        'num_samples': len(all_psnr)
    }
    
    return results