"""
Inference Pipeline for Thermal-to-RGB Colorization
Supports single image, batch, and full satellite scene processing
"""

import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from PIL import Image
import cv2
import os
from tqdm import tqdm
import albumentations as A


class ThermalColorizer:
    """
    Complete inference pipeline for thermal image colorization
    """
    
    def __init__(self, generator, sr_model=None, device='cuda', 
                 img_size=(256, 256)):
        self.generator = generator
        self.sr_model = sr_model
        self.device = device
        self.img_size = img_size
        
        self.generator = self.generator.to(device)
        self.generator.eval()
        
        if self.sr_model is not None:
            self.sr_model = self.sr_model.to(device)
            self.sr_model.eval()
    
    def preprocess_image(self, image):
        """
        Preprocess input image for model
        """
        # Convert to numpy if PIL image
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Resize if needed
        if image.shape[0] != self.img_size[0] or image.shape[1] != self.img_size[1]:
            image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_CUBIC)
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image / 255.0
        elif image.dtype == np.uint16:
            image = image / 65535.0
        
        # Add batch and channel dimension
        image = image[np.newaxis, np.newaxis, :, :]
        
        # Convert to tensor
        image = torch.FloatTensor(image).to(self.device)
        
        return image
    
    def postprocess_image(self, output):
        """
        Postprocess output to displayable image
        """
        # Convert to numpy
        if torch.is_tensor(output):
            output = output.cpu().numpy()
        
        # Remove batch dimension
        if len(output.shape) == 4:
            output = output[0]
        
        # Remove channel dimension if 1 channel
        if output.shape[0] == 1:
            output = output[0]
        
        # Ensure RGB format
        if len(output.shape) == 2:
            output = np.stack([output] * 3, axis=-1)
        elif output.shape[0] == 3:
            output = np.transpose(output, (1, 2, 0))
        
        # Clip and convert to uint8
        output = np.clip(output, 0, 1)
        output = (output * 255).astype(np.uint8)
        
        return output
    
    def colorize_single(self, image):
        """
        Colorize a single thermal image
        """
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Apply super-resolution if available
        if self.sr_model is not None:
            with torch.no_grad():
                _, sr_output = self.sr_model(input_tensor)
                input_tensor = sr_output
        
        # Colorize
        with torch.no_grad():
            outputs = self.generator(input_tensor)
            colorized = outputs['fine']
        
        # Postprocess
        colorized = self.postprocess_image(colorized)
        
        return colorized
    
    def colorize_tile(self, thermal_tile, tile_info=None):
        """
        Colorize a satellite tile with geospatial metadata
        """
        # Preprocess
        input_tensor = self.preprocess_image(thermal_tile)
        
        # Colorize
        with torch.no_grad():
            outputs = self.generator(input_tensor)
            colorized = outputs['fine']
        
        # Postprocess
        colorized = self.postprocess_image(colorized)
        
        # Return with metadata if provided
        if tile_info:
            return colorized, tile_info
        return colorized
    
    def colorize_scene(self, thermal_path, output_path, tile_size=256, overlap=32):
        """
        Colorize an entire satellite scene using sliding window
        """
        # Open thermal image
        with rasterio.open(thermal_path) as src:
            meta = src.meta.copy()
            height, width = src.height, src.width
            
            # Initialize output array
            output = np.zeros((height, width, 3), dtype=np.uint8)
            weight_map = np.zeros((height, width), dtype=np.float32)
            
            # Slide window
            for y in tqdm(range(0, height - tile_size + 1, tile_size - overlap)):
                for x in range(0, width - tile_size + 1, tile_size - overlap):
                    # Read tile
                    window = Window(x, y, tile_size, tile_size)
                    tile = src.read(1, window=window)
                    
                    # Colorize
                    colorized_tile = self.colorize_tile(tile)
                    
                    # Accumulate with weights (center pixels weighted more)
                    weight = self._create_weight_map(tile_size, overlap)
                    start_y, start_x = y, x
                    end_y, end_x = y + tile_size, x + tile_size
                    
                    output[start_y:end_y, start_x:end_x] = (
                        output[start_y:end_y, start_x:end_x] * 
                        weight_map[start_y:end_y, start_x:end_x, np.newaxis] +
                        colorized_tile * weight[:, :, np.newaxis]
                    ) / (weight_map[start_y:end_y, start_x:end_x, np.newaxis] + weight[:, :, np.newaxis] + 1e-6)
                    
                    weight_map[start_y:end_y, start_x:end_x] += weight
        
        # Save output
        meta.update(
            count=3,
            dtype='uint8',
            driver='GTiff'
        )
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            for i in range(3):
                dst.write(output[:, :, i], i+1)
        
        return output
    
    def _create_weight_map(self, tile_size, overlap):
        """
        Create weight map for smooth blending
        """
        weight = np.ones((tile_size, tile_size), dtype=np.float32)
        
        # Reduce weights near edges
        border = tile_size - overlap
        for i in range(tile_size):
            if i < overlap // 2:
                weight[i, :] *= i / (overlap // 2)
            elif i > tile_size - overlap // 2 - 1:
                weight[i, :] *= (tile_size - 1 - i) / (overlap // 2)
        
        for j in range(tile_size):
            if j < overlap // 2:
                weight[:, j] *= j / (overlap // 2)
            elif j > tile_size - overlap // 2 - 1:
                weight[:, j] *= (tile_size - 1 - j) / (overlap // 2)
        
        return weight


def load_model(checkpoint_path, generator, device='cuda'):
    """
    Load trained model from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    return generator