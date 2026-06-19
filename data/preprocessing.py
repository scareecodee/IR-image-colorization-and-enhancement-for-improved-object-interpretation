"""
Data Preprocessing Module for Landsat 8/9 Thermal and RGB Images
Handles: Download, co-registration, patch extraction, and augmentation
"""

import os
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import albumentations as A
from glob import glob
import warnings
warnings.filterwarnings('ignore')


class LandsatPreprocessor:
    """
    Preprocess Landsat 8/9 data for thermal IR colorization
    
    Bands:
    - Thermal Infrared (TIRS): Bands 10, 11 (100m resampled to 30m)
    - RGB: Bands 2 (Blue), 3 (Green), 4 (Red) (30m)
    - Panchromatic: Band 8 (15m) for super-resolution reference
    """
    
    def __init__(self, data_dir, output_dir, patch_size=256, stride=128):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.stride = stride
        
        # Band mappings
        self.thermal_bands = [10, 11]  # TIRS
        self.rgb_bands = [4, 3, 2]  # Red, Green, Blue
        self.panchromatic_band = 8  # For SR reference
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'thermal'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'rgb'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'pan'), exist_ok=True)
        
    def download_landsat_scene(self, scene_id, output_path):
        """
        Download Landsat scene from USGS EarthExplorer or AWS
        Note: Requires USGS credentials or use pre-downloaded data
        """
        # This is a placeholder - actual implementation would use:
        # - USGS EarthExplorer API
        # - AWS S3 (landsat-pds)
        # - Google Earth Engine
        
        # For hackathon, assume data is already downloaded
        print(f"Processing scene: {scene_id}")
        # Use existing data in data_dir
        pass
    
    def read_bands(self, scene_path, bands):
        """
        Read specific bands from Landsat scene
        """
        band_images = []
        for band in bands:
            band_file = os.path.join(scene_path, f"band{band}.tif")
            if os.path.exists(band_file):
                with rasterio.open(band_file) as src:
                    img = src.read(1)
                    band_images.append(img)
            else:
                print(f"Warning: Band {band} not found in {scene_path}")
                return None
        return np.stack(band_images, axis=0)
    
    def co_register_images(self, thermal_img, rgb_img, pan_img=None):
        """
        Co-register thermal and RGB images to ensure pixel-perfect alignment
        """
        # Since Landsat data is already co-registered, just ensure same dimensions
        h, w = thermal_img.shape[1], thermal_img.shape[2]
        
        # Resample RGB to match thermal if needed
        if rgb_img.shape[1] != h or rgb_img.shape[2] != w:
            rgb_resampled = np.zeros((3, h, w))
            for i in range(3):
                rgb_resampled[i] = cv2.resize(
                    rgb_img[i], (w, h), 
                    interpolation=cv2.INTER_CUBIC
                )
            rgb_img = rgb_resampled
            
        if pan_img is not None and (pan_img.shape[0] != h or pan_img.shape[1] != w):
            pan_resampled = cv2.resize(
                pan_img, (w, h), 
                interpolation=cv2.INTER_CUBIC
            )
            pan_img = pan_resampled
            
        return thermal_img, rgb_img, pan_img
    
    def extract_patches(self, image, patch_size=None, stride=None):
        """
        Extract overlapping patches from large satellite images
        """
        if patch_size is None:
            patch_size = self.patch_size
        if stride is None:
            stride = self.stride
            
        h, w = image.shape[1], image.shape[2] if len(image.shape) > 2 else image.shape
        patches = []
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                if len(image.shape) > 2:
                    patch = image[:, i:i+patch_size, j:j+patch_size]
                else:
                    patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
                
        return np.array(patches)
    
    def normalize_image(self, image, clip_percentile=2):
        """
        Normalize image to [0, 1] range with clipping
        """
        if len(image.shape) == 3:
            normalized = np.zeros_like(image, dtype=np.float32)
            for i in range(image.shape[0]):
                low = np.percentile(image[i], clip_percentile)
                high = np.percentile(image[i], 100 - clip_percentile)
                normalized[i] = np.clip((image[i] - low) / (high - low), 0, 1)
            return normalized
        else:
            low = np.percentile(image, clip_percentile)
            high = np.percentile(image, 100 - clip_percentile)
            return np.clip((image - low) / (high - low), 0, 1)
    
    def combine_thermal_bands(self, band10, band11):
        """
        Combine thermal bands for better thermal signature
        """
        # Simple average or use weighted combination
        combined = (band10 + band11) / 2
        return combined
    
    def process_scene(self, scene_path, scene_id):
        """
        Full preprocessing pipeline for a single Landsat scene
        """
        # Read bands
        thermal_data = self.read_bands(scene_path, self.thermal_bands)
        rgb_data = self.read_bands(scene_path, self.rgb_bands)
        
        if thermal_data is None or rgb_data is None:
            print(f"Skipping scene {scene_id}")
            return None
        
        # Combine thermal bands
        thermal_combined = self.combine_thermal_bands(
            thermal_data[0], thermal_data[1]
        )
        thermal_combined = thermal_combined[np.newaxis, :, :]
        
        # Normalize
        thermal_norm = self.normalize_image(thermal_combined)
        rgb_norm = self.normalize_image(rgb_data)
        
        # Co-register (ensuring same dimensions)
        thermal_norm, rgb_norm, _ = self.co_register_images(
            thermal_norm, rgb_norm
        )
        
        # Extract patches
        thermal_patches = self.extract_patches(thermal_norm)
        rgb_patches = self.extract_patches(rgb_norm)
        
        # Save patches
        for i in range(len(thermal_patches)):
            # Save thermal
            thermal_path = os.path.join(self.output_dir, 'thermal', 
                                       f"{scene_id}_{i:04d}.npy")
            np.save(thermal_path, thermal_patches[i])
            
            # Save RGB
            rgb_path = os.path.join(self.output_dir, 'rgb', 
                                   f"{scene_id}_{i:04d}.npy")
            np.save(rgb_path, rgb_patches[i])
            
        return len(thermal_patches)


class ThermalRgbDataset(Dataset):
    """
    PyTorch Dataset for thermal infrared to RGB translation
    """
    def __init__(self, data_dir, phase='train', transform=None, 
                 img_size=(256, 256), normalize=True):
        self.data_dir = data_dir
        self.phase = phase
        self.img_size = img_size
        self.normalize = normalize
        
        # Get all thermal and RGB pairs
        thermal_dir = os.path.join(data_dir, 'thermal')
        rgb_dir = os.path.join(data_dir, 'rgb')
        
        self.thermal_files = sorted(glob(os.path.join(thermal_dir, '*.npy')))
        self.rgb_files = sorted(glob(os.path.join(rgb_dir, '*.npy')))
        
        # Split dataset
        total = len(self.thermal_files)
        train_end = int(total * 0.8)
        val_end = int(total * 0.9)
        
        if phase == 'train':
            self.thermal_files = self.thermal_files[:train_end]
            self.rgb_files = self.rgb_files[:train_end]
        elif phase == 'val':
            self.thermal_files = self.thermal_files[train_end:val_end]
            self.rgb_files = self.rgb_files[train_end:val_end]
        else:  # test
            self.thermal_files = self.thermal_files[val_end:]
            self.rgb_files = self.rgb_files[val_end:]
        
        # Set transforms
        if transform is None:
            self.transform = self.get_default_transforms(phase)
        else:
            self.transform = transform
            
    def get_default_transforms(self, phase):
        """Get default augmentation transforms"""
        if phase == 'train':
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
            ])
        else:
            return A.Compose([])
    
    def __len__(self):
        return len(self.thermal_files)
    
    def __getitem__(self, idx):
        # Load data
        thermal = np.load(self.thermal_files[idx]).astype(np.float32)
        rgb = np.load(self.rgb_files[idx]).astype(np.float32)
        
        # Ensure proper shape: (C, H, W)
        if len(thermal.shape) == 2:
            thermal = thermal[np.newaxis, :, :]
        if len(rgb.shape) == 2:
            rgb = rgb[np.newaxis, :, :]
        
        # Resize if needed
        if thermal.shape[1] != self.img_size[0] or thermal.shape[2] != self.img_size[1]:
            thermal = cv2.resize(thermal[0], self.img_size, interpolation=cv2.INTER_CUBIC)
            thermal = thermal[np.newaxis, :, :]
            rgb = np.transpose(
                cv2.resize(np.transpose(rgb, (1, 2, 0)), self.img_size, 
                          interpolation=cv2.INTER_CUBIC),
                (2, 0, 1)
            )
        
        # Apply transforms (on HWC format for albumentations)
        thermal_hwc = np.transpose(thermal, (1, 2, 0))
        rgb_hwc = np.transpose(rgb, (1, 2, 0))
        
        # Apply augmentations
        augmented = self.transform(image=thermal_hwc, mask=rgb_hwc)
        thermal_aug = augmented['image']
        rgb_aug = augmented['mask']
        
        # Convert back to CHW
        thermal = np.transpose(thermal_aug, (2, 0, 1))
        rgb = np.transpose(rgb_aug, (2, 0, 1))
        
        # Ensure values in [0, 1]
        thermal = np.clip(thermal, 0, 1)
        rgb = np.clip(rgb, 0, 1)
        
        return {
            'thermal': torch.FloatTensor(thermal),
            'rgb': torch.FloatTensor(rgb),
            'thermal_path': self.thermal_files[idx],
            'rgb_path': self.rgb_files[idx]
        }


def create_dataloaders(data_dir, batch_size=8, img_size=(256, 256), 
                       num_workers=4, pin_memory=True):
    """
    Create train, val, and test dataloaders
    """
    datasets = {
        'train': ThermalRgbDataset(
            data_dir, 'train', img_size=img_size
        ),
        'val': ThermalRgbDataset(
            data_dir, 'val', img_size=img_size
        ),
        'test': ThermalRgbDataset(
            data_dir, 'test', img_size=img_size
        )
    }
    
    dataloaders = {}
    for phase, dataset in datasets.items():
        dataloaders[phase] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size if phase == 'train' else 1,
            shuffle=(phase == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(phase == 'train')
        )
    
    return dataloaders


if __name__ == '__main__':
    # Test preprocessing
    preprocessor = LandsatPreprocessor(
        data_dir='raw_data/',
        output_dir='processed_data/',
        patch_size=256,
        stride=128
    )
    
    # Process a sample scene
    # preprocessor.process_scene('path_to_scene', 'scene_id')
    
    # Test dataset
    dataloaders = create_dataloaders(
        data_dir='processed_data/',
        batch_size=4,
        img_size=(256, 256)
    )
    
    sample = next(iter(dataloaders['train']))
    print(f"Thermal shape: {sample['thermal'].shape}")
    print(f"RGB shape: {sample['rgb'].shape}")