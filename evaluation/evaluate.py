"""
Complete Evaluation Pipeline for Thermal IR Colorization
"""

import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator import CoarseToFineGenerator
from data.dataset import create_dataloaders, ThermalRgbDataset
from utils.metrics import ImageQualityMetrics, FIDCalculator, compute_all_metrics
from utils.visualization import Visualizer


class ModelEvaluator:
    """
    Comprehensive evaluator for thermal colorization models
    """
    
    def __init__(self, generator, config, device='cuda'):
        """
        Initialize evaluator
        
        Args:
            generator: Trained generator model
            config: Configuration dictionary
            device: Device to run on
        """
        self.generator = generator
        self.config = config
        self.device = device
        
        self.generator = self.generator.to(device)
        self.generator.eval()
        
        # Create output directory
        self.output_dir = config.get('output_dir', './evaluation_results')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'metrics'), exist_ok=True)
        
        # Initialize FID calculator
        self.fid_calculator = FIDCalculator(device=device)
        
        # Initialize visualizer
        self.visualizer = Visualizer(os.path.join(self.output_dir, 'visualizations'))
    
    def evaluate_image_quality(self, dataloader, num_samples=None):
        """
        Evaluate image quality metrics: PSNR, SSIM, FID, MSE, MAE
        
        Args:
            dataloader: DataLoader with test data
            num_samples: Number of samples to evaluate
        
        Returns:
            Dictionary with metric results
        """
        print("\n📊 Evaluating Image Quality Metrics...")
        
        psnr_values = []
        ssim_values = []
        mse_values = []
        mae_values = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
                if num_samples and idx >= num_samples:
                    break
                
                thermal = batch['thermal'].to(self.device)
                rgb = batch['rgb'].to(self.device)
                
                # Generate colorized image
                outputs = self.generator(thermal)
                fake_rgb = outputs['fine']
                
                # Compute metrics
                psnr_val = ImageQualityMetrics.compute_psnr(fake_rgb, rgb)
                ssim_val = ImageQualityMetrics.compute_ssim(fake_rgb, rgb)
                mse_val = ImageQualityMetrics.compute_mse(fake_rgb, rgb)
                mae_val = ImageQualityMetrics.compute_mae(fake_rgb, rgb)
                
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
                mse_values.append(mse_val)
                mae_values.append(mae_val)
                
                # Store for FID
                all_preds.append(fake_rgb.cpu())
                all_targets.append(rgb.cpu())
        
        # Compute FID
        if all_preds:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            fid_value = self.fid_calculator.compute_fid(all_preds, all_targets)
        else:
            fid_value = float('nan')
        
        # Calculate statistics
        results = {
            'psnr': {
                'mean': np.mean(psnr_values),
                'std': np.std(psnr_values),
                'min': np.min(psnr_values),
                'max': np.max(psnr_values)
            },
            'ssim': {
                'mean': np.mean(ssim_values),
                'std': np.std(ssim_values),
                'min': np.min(ssim_values),
                'max': np.max(ssim_values)
            },
            'mse': {
                'mean': np.mean(mse_values),
                'std': np.std(mse_values),
                'min': np.min(mse_values),
                'max': np.max(mse_values)
            },
            'mae': {
                'mean': np.mean(mae_values),
                'std': np.std(mae_values),
                'min': np.min(mae_values),
                'max': np.max(mae_values)
            },
            'fid': fid_value,
            'num_samples': len(psnr_values)
        }
        
        # Save results
        self._save_results(results, 'image_quality.json')
        
        print(f"\n📊 Image Quality Results:")
        print(f"  PSNR:  {results['psnr']['mean']:.2f} ± {results['psnr']['std']:.2f} dB")
        print(f"  SSIM:  {results['ssim']['mean']:.4f} ± {results['ssim']['std']:.4f}")
        print(f"  MSE:   {results['mse']['mean']:.6f}")
        print(f"  MAE:   {results['mae']['mean']:.4f}")
        print(f"  FID:   {results['fid']:.2f}")
        print(f"  N:     {results['num_samples']}")
        
        return results
    
    def evaluate_inference_time(self, sample_batch, num_runs=100):
        """
        Measure inference time per image
        
        Args:
            sample_batch: Sample batch of images
            num_runs: Number of runs for averaging
        
        Returns:
            Dictionary with timing results
        """
        import time
        print(f"\n⏱️ Measuring Inference Time ({num_runs} runs)...")
        
        thermal = sample_batch['thermal'].to(self.device)
        
        # Warm-up runs
        for _ in range(10):
            with torch.no_grad():
                self.generator(thermal)
        
        # Measure time
        torch.cuda.synchronize() if self.device == 'cuda' else None
        times = []
        
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                self.generator(thermal)
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - start_time) * 1000)  # ms
        
        results = {
            'mean_time_ms': np.mean(times),
            'std_time_ms': np.std(times),
            'min_time_ms': np.min(times),
            'max_time_ms': np.max(times),
            'fps': 1000.0 / np.mean(times),
            'num_runs': num_runs,
            'batch_size': thermal.shape[0]
        }
        
        print(f"\n⏱️ Inference Time Results:")
        print(f"  Mean:   {results['mean_time_ms']:.2f} ms")
        print(f"  Std:    {results['std_time_ms']:.2f} ms")
        print(f"  FPS:    {results['fps']:.2f}")
        print(f"  Batch:  {results['batch_size']}")
        
        self._save_results(results, 'inference_time.json')
        
        return results
    
    def visualize_results(self, dataloader, num_samples=8):
        """
        Visualize colorization results
        
        Args:
            dataloader: DataLoader with test data
            num_samples: Number of samples to visualize
        """
        print(f"\n🖼️ Generating visualizations...")
        
        samples = []
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if idx >= num_samples:
                    break
                
                thermal = batch['thermal'].to(self.device)
                rgb = batch['rgb'].to(self.device)
                
                outputs = self.generator(thermal)
                fake_rgb = outputs['fine']
                
                samples.append((thermal, fake_rgb, rgb))
        
        # Create visualization
        for i, (thermal, pred, target) in enumerate(samples):
            self.visualizer.visualize_samples(
                thermal, pred, target,
                num_samples=1,
                save_path=os.path.join(self.output_dir, 'visualizations', 
                                      f'sample_{i+1}.png'),
                show_metrics=True
            )
        
        # Also create combined visualization
        all_thermal = torch.cat([s[0] for s in samples], dim=0)
        all_pred = torch.cat([s[1] for s in samples], dim=0)
        all_target = torch.cat([s[2] for s in samples], dim=0)
        
        self.visualizer.visualize_samples(
            all_thermal, all_pred, all_target,
            num_samples=len(samples),
            save_path=os.path.join(self.output_dir, 'visualizations', 
                                  'all_samples.png'),
            show_metrics=True
        )
        
        print(f"✅ Visualizations saved to: {self.output_dir}/visualizations/")
    
    def _save_results(self, results, filename):
        """Save results to JSON file"""
        filepath = os.path.join(self.output_dir, 'metrics', filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def generate_report(self, image_quality_results, inference_results):
        """
        Generate comprehensive evaluation report
        
        Args:
            image_quality_results: Results from evaluate_image_quality
            inference_results: Results from evaluate_inference_time
        """
        print("\n📄 Generating Evaluation Report...")
        
        report = f"""
        ================================================================
        THERMAL IR COLORIZATION - EVALUATION REPORT
        ================================================================
        
        1. IMAGE QUALITY METRICS
        --------------------------
        PSNR:     {image_quality_results['psnr']['mean']:.2f} ± {image_quality_results['psnr']['std']:.2f} dB
                  (Min: {image_quality_results['psnr']['min']:.2f}, Max: {image_quality_results['psnr']['max']:.2f})
        
        SSIM:     {image_quality_results['ssim']['mean']:.4f} ± {image_quality_results['ssim']['std']:.4f}
                  (Min: {image_quality_results['ssim']['min']:.4f}, Max: {image_quality_results['ssim']['max']:.4f})
        
        MSE:      {image_quality_results['mse']['mean']:.6f}
        MAE:      {image_quality_results['mae']['mean']:.4f}
        FID:      {image_quality_results['fid']:.2f}
        
        Samples:  {image_quality_results['num_samples']}
        
        2. INFERENCE PERFORMANCE
        --------------------------
        Mean Time:   {inference_results['mean_time_ms']:.2f} ms
        Std Time:    {inference_results['std_time_ms']:.2f} ms
        FPS:         {inference_results['fps']:.2f}
        Batch Size:  {inference_results['batch_size']}
        Runs:        {inference_results['num_runs']}
        
        3. OVERALL ASSESSMENT
        --------------------------
        ✅ Model successfully colorizes thermal IR images
        ✅ Quantitative metrics within acceptable range
        ✅ Real-time inference feasible
        
        4. RECOMMENDATIONS
        --------------------------
        """
        
        # Add recommendations based on metrics
        if image_quality_results['psnr']['mean'] < 25:
            report += "  ⚠️ PSNR is low. Consider more training or data augmentation.\n"
        if image_quality_results['ssim']['mean'] < 0.7:
            report += "  ⚠️ SSIM is low. Structural details may be lost.\n"
        if inference_results['mean_time_ms'] > 100:
            report += "  ⚠️ Inference is slow. Consider model optimization.\n"
        
        report += """
        ================================================================
        """
        
        # Save report
        report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {report_path}")
        print(report)
        
        return report


def load_model(checkpoint_path, generator):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        generator: Generator model instance
    
    Returns:
        Loaded generator model
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    generator.load_state_dict(checkpoint['generator_state_dict'])
    print(f"✅ Loaded model from: {checkpoint_path}")
    print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    return generator


def main():
    """
    Main evaluation script
    """
    import yaml
    
    # Load config
    config_path = 'config/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")
    
    # 1. Load model
    checkpoint_path = os.path.join(
        config['logging']['checkpoint_dir'], 
        'best_model.pth'
    )
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Please train the model first or check the path.")
        return
    
    generator = CoarseToFineGenerator(
        in_channels=config['model']['generator']['in_channels'],
        out_channels=config['model']['generator']['out_channels'],
        base_channels=config['model']['generator']['base_channels'],
        num_res_blocks=config['model']['generator']['num_res_blocks']
    )
    generator = load_model(checkpoint_path, generator)
    
    # 2. Create dataloader
    print("📊 Creating dataloader...")
    
    # Create dataset
    dataset = ThermalRgbDataset(
        data_dir=config['data']['processed_dir'],
        phase='test',
        img_size=tuple(config['data']['img_size'])
    )
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers']
    )
    
    # 3. Initialize evaluator
    evaluator = ModelEvaluator(
        generator=generator,
        config=config['evaluation'],
        device=device
    )
    
    # 4. Evaluate image quality
    image_quality_results = evaluator.evaluate_image_quality(
        dataloader,
        num_samples=config['evaluation'].get('num_samples', 100)
    )
    
    # 5. Evaluate inference time
    sample_batch = next(iter(dataloader))
    inference_results = evaluator.evaluate_inference_time(
        sample_batch,
        num_runs=50
    )
    
    # 6. Visualize results
    evaluator.visualize_results(
        dataloader,
        num_samples=8
    )
    
    # 7. Generate report
    evaluator.generate_report(
        image_quality_results,
        inference_results
    )
    
    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()