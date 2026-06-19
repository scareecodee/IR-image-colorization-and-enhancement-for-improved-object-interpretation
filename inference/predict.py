"""
Inference Pipeline for Thermal-to-RGB Colorization
Supports: Single image, batch processing, and full satellite scenes
"""

import os
import sys
import time
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
from PIL import Image
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator import CoarseToFineGenerator
from utils.visualization import Visualizer


class ThermalColorizer:
    """
    Complete inference pipeline for thermal image colorization
    """
    
    def __init__(self, generator, config=None, device='cuda', img_size=(256, 256)):
        """
        Initialize colorizer
        
        Args:
            generator: Trained generator model
            config: Configuration dictionary
            device: Device to run on
            img_size: Input image size
        """
        self.generator = generator
        self.config = config or {}
        self.device = device
        self.img_size = img_size
        
        self.generator = self.generator.to(device)
        self.generator.eval()
        
        # Create output directory
        self.output_dir = self.config.get('output_dir', './outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = Visualizer(os.path.join(self.output_dir, 'visualizations'))
    
    def preprocess_single(self, image):
        """
        Preprocess a single image for inference
        
        Args:
            image: Input image (numpy array or PIL Image)
        
        Returns:
            Preprocessed tensor
        """
        # Convert to numpy if PIL
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
    
    def postprocess_single(self, output):
        """
        Postprocess model output to displayable image
        
        Args:
            output: Model output tensor
        
        Returns:
            RGB image as numpy array (H, W, 3)
        """
        # Convert to numpy
        if torch.is_tensor(output):
            output = output.cpu().numpy()
        
        # Remove batch dimension
        if len(output.shape) == 4:
            output = output[0]
        
        # Denormalize from [-1, 1] to [0, 1]
        if output.max() > 1:
            output = (output + 1) / 2
        
        # Clip and convert to HWC
        output = np.clip(output, 0, 1)
        if output.shape[0] == 3:
            output = np.transpose(output, (1, 2, 0))
        
        # Convert to uint8
        output = (output * 255).astype(np.uint8)
        
        return output
    
    def colorize_single(self, image, save_path=None, return_tensor=False):
        """
        Colorize a single thermal image
        
        Args:
            image: Input thermal image
            save_path: Path to save the result
            return_tensor: Return tensor instead of numpy
        
        Returns:
            Colorized RGB image
        """
        # Preprocess
        input_tensor = self.preprocess_single(image)
        
        # Colorize
        with torch.no_grad():
            outputs = self.generator(input_tensor)
            colorized = outputs['fine']
        
        if return_tensor:
            return colorized
        
        # Postprocess
        colorized = self.postprocess_single(colorized)
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.fromarray(colorized).save(save_path)
        
        return colorized
    
    def colorize_batch(self, images, save_dir=None):
        """
        Colorize a batch of thermal images
        
        Args:
            images: List of input images
            save_dir: Directory to save results
        
        Returns:
            List of colorized images
        """
        results = []
        
        for idx, image in enumerate(tqdm(images, desc="Colorizing")):
            save_path = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'colorized_{idx:04d}.png')
            
            result = self.colorize_single(image, save_path)
            results.append(result)
        
        return results
    
    def colorize_tile(self, thermal_tile, meta=None, save_path=None):
        """
        Colorize a satellite tile with geospatial metadata
        
        Args:
            thermal_tile: Thermal image tile (numpy array)
            meta: Rasterio metadata
            save_path: Path to save GeoTIFF
        
        Returns:
            Colorized tile and metadata
        """
        # Preprocess
        input_tensor = self.preprocess_single(thermal_tile)
        
        # Colorize
        with torch.no_grad():
            outputs = self.generator(input_tensor)
            colorized = outputs['fine']
        
        # Postprocess
        colorized = self.postprocess_single(colorized)
        
        # Save GeoTIFF if metadata provided
        if meta and save_path:
            meta.update(
                count=3,
                dtype='uint8',
                driver='GTiff',
                compress='lzw'
            )
            with rasterio.open(save_path, 'w', **meta) as dst:
                for i in range(3):
                    dst.write(colorized[:, :, i], i+1)
        
        return colorized
    
    def colorize_scene(self, thermal_path, output_path, tile_size=256, overlap=32):
        """
        Colorize an entire satellite scene using sliding window
        
        Args:
            thermal_path: Path to thermal GeoTIFF
            output_path: Path to save colorized GeoTIFF
            tile_size: Tile size for processing
            overlap: Overlap between tiles
        
        Returns:
            Colorized scene as numpy array
        """
        print(f"🔄 Processing scene: {thermal_path}")
        
        # Open thermal image
        with rasterio.open(thermal_path) as src:
            meta = src.meta.copy()
            height, width = src.height, src.width
            
            # Initialize output array
            output = np.zeros((height, width, 3), dtype=np.float32)
            weight_map = np.zeros((height, width), dtype=np.float32)
            
            # Slide window
            step = tile_size - overlap
            
            for y in tqdm(range(0, height - tile_size + 1, step), desc="Processing tiles"):
                for x in range(0, width - tile_size + 1, step):
                    # Read tile
                    window = Window(x, y, tile_size, tile_size)
                    tile = src.read(1, window=window)
                    
                    # Colorize
                    colorized_tile = self.colorize_tile(tile) / 255.0
                    
                    # Create weight map for blending
                    weight = self._create_weight_map(tile_size, overlap)
                    
                    # Accumulate
                    output[y:y+tile_size, x:x+tile_size] += colorized_tile * weight[:, :, np.newaxis]
                    weight_map[y:y+tile_size, x:x+tile_size] += weight
            
            # Normalize by weight map
            output = np.clip(output / (weight_map[:, :, np.newaxis] + 1e-8), 0, 1)
            output = (output * 255).astype(np.uint8)
        
        # Save output
        meta.update(
            count=3,
            dtype='uint8',
            driver='GTiff',
            compress='lzw'
        )
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            for i in range(3):
                dst.write(output[:, :, i], i+1)
        
        print(f"✅ Colorized scene saved to: {output_path}")
        
        return output
    
    def _create_weight_map(self, tile_size, overlap):
        """
        Create weight map for smooth blending
        """
        weight = np.ones((tile_size, tile_size), dtype=np.float32)
        
        # Reduce weights near edges
        border = overlap // 2
        for i in range(tile_size):
            if i < border:
                weight[i, :] *= i / border
            elif i > tile_size - border - 1:
                weight[i, :] *= (tile_size - 1 - i) / border
        
        for j in range(tile_size):
            if j < border:
                weight[:, j] *= j / border
            elif j > tile_size - border - 1:
                weight[:, j] *= (tile_size - 1 - j) / border
        
        return weight
    
    def compare_and_visualize(self, thermal, rgb=None, save_path=None):
        """
        Visualize input, colorized result, and ground truth
        
        Args:
            thermal: Input thermal image
            rgb: Ground truth RGB (optional)
            save_path: Path to save visualization
        """
        # Colorize
        colorized = self.colorize_single(thermal)
        
        # Display
        fig, axes = plt.subplots(1, 3 if rgb is not None else 2, figsize=(15, 5))
        
        # Thermal
        thermal_display = thermal.copy()
        if len(thermal_display.shape) == 3:
            thermal_display = cv2.cvtColor(thermal_display, cv2.COLOR_RGB2GRAY)
        
        axes[0].imshow(thermal_display, cmap='gray')
        axes[0].set_title('Input Thermal')
        axes[0].axis('off')
        
        # Colorized
        axes[1].imshow(colorized)
        axes[1].set_title('Colorized RGB')
        axes[1].axis('off')
        
        # Ground Truth
        if rgb is not None:
            axes[2].imshow(rgb)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Visualization saved to: {save_path}")
        
        plt.close()
        
        return colorized


def load_colorizer(checkpoint_path, config=None, device='cuda'):
    """
    Load trained model and create colorizer
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dictionary
        device: Device to run on
    
    Returns:
        ThermalColorizer instance
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create generator
    generator = CoarseToFineGenerator(
        in_channels=1,
        out_channels=3,
        base_channels=64,
        num_res_blocks=9
    )
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    print(f"✅ Loaded model from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    # Create colorizer
    colorizer = ThermalColorizer(
        generator=generator,
        config=config,
        device=device
    )
    
    return colorizer


def main():
    """
    Main inference script
    """
    import yaml
    
    # Load config
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")
    
    # Load colorizer
    checkpoint_path = os.path.join(
        config['logging']['checkpoint_dir'], 
        'best_model.pth'
    )
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Please train the model first.")
        return
    
    colorizer = load_colorizer(checkpoint_path, config, device)
    
    # Colorize a test image
    test_image_path = os.path.join(config['data']['raw_dir'], 'test_thermal.tif')
    
    if os.path.exists(test_image_path):
        print(f"\n📷 Colorizing: {test_image_path}")
        
        # Colorize single image
        result = colorizer.colorize_single(
            test_image_path,
            save_path=os.path.join(config['logging']['output_dir'], 'colorized_result.png')
        )
        
        print(f"✅ Result saved to: {config['logging']['output_dir']}/colorized_result.png")
    else:
        print(f"\n⚠️ No test image found at: {test_image_path}")
        print("   Please place a test thermal image in the raw directory.")
    
    print("\n✅ Inference complete!")


if __name__ == '__main__':
    main()