"""
PyTorch Dataset for Thermal IR to RGB Translation
Uses: Thermal (Band 10) -> RGB (Bands 2, 3, 4)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from glob import glob
import cv2


class ThermalRgbDataset(Dataset):
    """
    Dataset for thermal infrared to RGB colorization
    Input: Thermal band (Band 10)
    Target: RGB (Bands 2, 3, 4)
    """
    
    def __init__(self, data_dir, phase='train', img_size=(256, 256),
                 transform=None, normalize=True):
        """
        Args:
            data_dir: Directory containing thermal/ and rgb/ subdirectories
            phase: 'train', 'val', or 'test'
            img_size: Output image size
            transform: Albumentations transforms
            normalize: Whether to normalize to [0, 1]
        """
        self.data_dir = data_dir
        self.phase = phase
        self.img_size = img_size
        self.normalize = normalize
        
        # Paths
        thermal_dir = os.path.join(data_dir, 'thermal')
        rgb_dir = os.path.join(data_dir, 'rgb')
        
        # Get files
        self.thermal_files = sorted(glob(os.path.join(thermal_dir, '*.npy')))
        self.rgb_files = sorted(glob(os.path.join(rgb_dir, '*.npy')))
        
        # Verify matching files
        assert len(self.thermal_files) == len(self.rgb_files), \
            f"Mismatched files: Thermal={len(self.thermal_files)}, RGB={len(self.rgb_files)}"
        
        # Split dataset (80% train, 10% val, 10% test)
        total = len(self.thermal_files)
        indices = np.random.permutation(total)
        
        train_end = int(0.8 * total)
        val_end = int(0.9 * total)
        
        if phase == 'train':
            self.indices = indices[:train_end]
        elif phase == 'val':
            self.indices = indices[train_end:val_end]
        else:  # test
            self.indices = indices[val_end:]
        
        print(f"📊 {phase.upper()} set: {len(self.indices)} samples")
        
        # Set transforms
        if transform is None:
            self.transform = self._get_default_transforms(phase)
        else:
            self.transform = transform
    
    def _get_default_transforms(self, phase):
        """
        Get default augmentation transforms
        """
        if phase == 'train':
            return A.Compose([
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.3
                ),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.RandomResizedCrop(
                    height=self.img_size[0],
                    width=self.img_size[1],
                    scale=(0.8, 1.0),
                    p=0.3
                ),
            ])
        else:
            return A.Compose([])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get actual index
        actual_idx = self.indices[idx]
        
        # Load data
        thermal = np.load(self.thermal_files[actual_idx]).astype(np.float32)
        rgb = np.load(self.rgb_files[actual_idx]).astype(np.float32)
        
        # Ensure proper shape: (C, H, W)
        if len(thermal.shape) == 2:
            thermal = thermal[np.newaxis, :, :]
        if len(rgb.shape) == 2:
            rgb = rgb[np.newaxis, :, :]
        
        # Resize if needed
        if thermal.shape[1] != self.img_size[0] or thermal.shape[2] != self.img_size[1]:
            thermal = cv2.resize(
                thermal[0], 
                self.img_size,
                interpolation=cv2.INTER_CUBIC
            )
            thermal = thermal[np.newaxis, :, :]
            
            rgb = np.transpose(rgb, (1, 2, 0))
            rgb = cv2.resize(rgb, self.img_size, interpolation=cv2.INTER_CUBIC)
            rgb = np.transpose(rgb, (2, 0, 1))
        
        # Convert to HWC for albumentations
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
        if self.normalize:
            thermal = np.clip(thermal, 0, 1)
            rgb = np.clip(rgb, 0, 1)
        
        return {
            'thermal': torch.FloatTensor(thermal),
            'rgb': torch.FloatTensor(rgb),
            'thermal_path': self.thermal_files[actual_idx],
            'rgb_path': self.rgb_files[actual_idx]
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
        dataloaders[phase] = DataLoader(
            dataset,
            batch_size=batch_size if phase == 'train' else 1,
            shuffle=(phase == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(phase == 'train')
        )
    
    return dataloaders