"""
Data Preprocessing Module for Landsat 8/9 Level-2 (SR) Data
Handles new naming convention: LC09_L2SP_146040_20260423_20260424_02_T1_SR_B2.TIF
Bands Used: B2 (Blue), B3 (Green), B4 (Red), B10 (Thermal)
"""

import os
import re
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
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
    Preprocess Landsat 8/9 Level-2 data for thermal IR colorization
    
    Handles both old and new naming conventions:
    - Old: LC08_L1TP_143042_20200101_20200117_01_T1_B2.TIF
    - New: LC09_L2SP_146040_20260423_20260424_02_T1_SR_B2.TIF
    
    Bands:
    - RGB: Bands 2 (Blue), 3 (Green), 4 (Red) - 30m resolution
    - Thermal: Band 10 - 100m (resampled to 30m)
    """
    
    def __init__(self, raw_dir, output_dir, patch_size=256, stride=128):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.patch_size = patch_size
        self.stride = stride
        
        # Band configuration (Level-2 SR bands)
        self.rgb_bands = [4, 3, 2]  # Red, Green, Blue (for ground truth)
        self.thermal_band = 10      # Thermal infrared (input)
        
        # Create output directories
        self.thermal_dir = os.path.join(output_dir, 'thermal')
        self.rgb_dir = os.path.join(output_dir, 'rgb')
        
        os.makedirs(self.thermal_dir, exist_ok=True)
        os.makedirs(self.rgb_dir, exist_ok=True)
        
        print(f"📂 Raw data directory: {raw_dir}")
        print(f"📂 Output directory: {output_dir}")
    
    def find_band_file(self, scene_path, band):
        """
        Find band file with flexible naming patterns
        
        Supports:
        - LC08_L1TP_143042_20200101_20200117_01_T1_B2.TIF
        - LC09_L2SP_146040_20260423_20260424_02_T1_SR_B2.TIF
        - *B2.TIF, *band2.tif, etc.
        """
        # Get base filename patterns
        base_patterns = [
            f"*B{band}.TIF",           # Modern: _B2.TIF
            f"*_band{band}.tif",       # Legacy: _band2.tif
            f"*B{band}.TIF",           # Alternative: B2.TIF
            f"*band{band}.tif",        # Alternative: band2.tif
            f"*_{band}.TIF",           # Alternative: _2.TIF
        ]
        
        # Search for the band file
        for pattern in base_patterns:
            pattern_path = os.path.join(scene_path, pattern)
            files = glob(pattern_path)
            if files:
                print(f"✅ Found band {band}: {os.path.basename(files[0])}")
                return files[0]
        
        # If still not found, try recursive search
        all_files = glob(os.path.join(scene_path, "**", "*"), recursive=True)
        for file in all_files:
            filename = os.path.basename(file)
            # Check if file contains band number
            if f"B{band}" in filename or f"band{band}" in filename or f"_{band}." in filename:
                if filename.endswith(('.TIF', '.tif', '.tiff')):
                    print(f"✅ Found band {band}: {filename}")
                    return file
        
        raise FileNotFoundError(f"Band {band} not found in {scene_path}")
    
    def read_band(self, scene_path, band):
        """
        Read a single band from Landsat scene with proper handling
        """
        try:
            band_path = self.find_band_file(scene_path, band)
            with rasterio.open(band_path) as src:
                data = src.read(1)
                meta = src.meta.copy()
                # For Level-2 SR data, values are already scaled
                # Typically 0-10000 or 0-65535 for thermal
                return data, meta
        except Exception as e:
            print(f"❌ Error reading band {band}: {e}")
            raise
    
    def read_rgb_bands(self, scene_path):
        """
        Read RGB bands (B2, B3, B4) and stack them
        """
        rgb_images = []
        for band in self.rgb_bands:
            data, _ = self.read_band(scene_path, band)
            rgb_images.append(data)
        
        rgb_stack = np.stack(rgb_images, axis=0)
        return rgb_stack
    
    def read_thermal_band(self, scene_path):
        """
        Read thermal band (B10)
        """
        data, meta = self.read_band(scene_path, self.thermal_band)
        return data, meta
    
    def scale_level2_data(self, data, band_type='rgb'):
        """
        Scale Level-2 Surface Reflectance data to [0, 1]
        
        Level-2 SR data typically:
        - RGB: 0-10000 (reflectance * 10000)
        - Thermal: 0-65535 (temperature * 100)
        """
        if band_type == 'rgb':
            # Surface reflectance: scale 0-10000 to 0-1
            data = data.astype(np.float32) / 10000.0
        else:  # thermal
            # Thermal: scale 0-65535 to 0-1
            data = data.astype(np.float32) / 65535.0
        
        # Clip to valid range
        data = np.clip(data, 0, 1)
        return data
    
    def co_register_bands(self, thermal_data, rgb_data):
        """
        Co-register thermal and RGB bands
        Thermal is 100m, RGB is 30m - resample thermal to match RGB
        """
        h_rgb, w_rgb = rgb_data.shape[1], rgb_data.shape[2]
        h_thermal, w_thermal = thermal_data.shape[0], thermal_data.shape[1]
        
        # If thermal is smaller, upsample to match RGB
        if h_thermal != h_rgb or w_thermal != w_rgb:
            thermal_resampled = cv2.resize(
                thermal_data, 
                (w_rgb, h_rgb),
                interpolation=cv2.INTER_CUBIC
            )
            thermal_resampled = thermal_resampled[np.newaxis, :, :]
        else:
            thermal_resampled = thermal_data[np.newaxis, :, :]
        
        return thermal_resampled, rgb_data
    
    def normalize_image(self, image, clip_percentile=2):
        """
        Normalize image to [0, 1] range with percentile clipping
        """
        if len(image.shape) == 3:
            normalized = np.zeros_like(image, dtype=np.float32)
            for i in range(image.shape[0]):
                low = np.percentile(image[i], clip_percentile)
                high = np.percentile(image[i], 100 - clip_percentile)
                normalized[i] = np.clip((image[i] - low) / (high - low + 1e-8), 0, 1)
            return normalized
        else:
            low = np.percentile(image, clip_percentile)
            high = np.percentile(image, 100 - clip_percentile)
            return np.clip((image - low) / (high - low + 1e-8), 0, 1)
    
    def extract_patches(self, image, patch_size=None, stride=None):
        """
        Extract overlapping patches from large satellite images
        """
        if patch_size is None:
            patch_size = self.patch_size
        if stride is None:
            stride = self.stride
        
        if len(image.shape) == 3:
            _, h, w = image.shape
        else:
            h, w = image.shape
        
        patches = []
        
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                if len(image.shape) == 3:
                    patch = image[:, i:i+patch_size, j:j+patch_size]
                else:
                    patch = image[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        
        return np.array(patches)
    
    def process_scene(self, scene_path, scene_id):
        """
        Full preprocessing pipeline for a single Landsat scene
        """
        try:
            print(f"\n🔄 Processing scene: {scene_id}")
            print(f"   Path: {scene_path}")
            
            # Read RGB bands
            print("   Reading RGB bands...")
            rgb_data = self.read_rgb_bands(scene_path)
            
            # Read thermal band
            print("   Reading thermal band...")
            thermal_data, _ = self.read_thermal_band(scene_path)
            
            # Scale Level-2 data
            print("   Scaling data...")
            thermal_data = self.scale_level2_data(thermal_data, 'thermal')
            rgb_data = self.scale_level2_data(rgb_data, 'rgb')
            
            # Co-register
            print("   Co-registering bands...")
            thermal_reg, rgb_reg = self.co_register_bands(thermal_data, rgb_data)
            
            # Normalize
            print("   Normalizing...")
            thermal_norm = self.normalize_image(thermal_reg)
            rgb_norm = self.normalize_image(rgb_reg)
            
            # Extract patches
            print("   Extracting patches...")
            thermal_patches = self.extract_patches(thermal_norm)
            rgb_patches = self.extract_patches(rgb_norm)
            
            # Verify matching number of patches
            assert len(thermal_patches) == len(rgb_patches), "Mismatched patches!"
            
            # Save patches
            print(f"   Saving {len(thermal_patches)} patches...")
            for i in range(len(thermal_patches)):
                # Thermal patch
                thermal_path = os.path.join(self.thermal_dir, f"{scene_id}_{i:04d}.npy")
                np.save(thermal_path, thermal_patches[i].astype(np.float32))
                
                # RGB patch
                rgb_path = os.path.join(self.rgb_dir, f"{scene_id}_{i:04d}.npy")
                np.save(rgb_path, rgb_patches[i].astype(np.float32))
            
            print(f"  ✅ Created {len(thermal_patches)} patches from scene: {scene_id}")
            return len(thermal_patches)
            
        except Exception as e:
            print(f"  ❌ Error processing scene {scene_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return 0


def extract_patches(image, patch_size=256, stride=128):
    """
    Utility function to extract patches
    """
    preprocessor = LandsatPreprocessor('', '')
    return preprocessor.extract_patches(image, patch_size, stride)