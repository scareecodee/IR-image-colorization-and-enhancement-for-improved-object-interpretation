"""
Main entry point for the Thermal Colorization Pipeline
"""

import os
import sys
import argparse
import yaml
import torch
from torch.utils.tensorboard import SummaryWriter

# Import modules
from data.preprocessing import create_dataloaders, LandsatPreprocessor
from models.generator import CoarseToFineGenerator
from models.discriminator import PatchGANDiscriminator
from models.super_resolution import MultiStageSuperResolution
from training.trainer import ColorizationTrainer, SuperResolutionTrainer
from inference.predict import ThermalColorizer, load_model
from utils.metrics import evaluate_model
from utils.visualization import visualize_results


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def phase1_data_preprocessing(config):
    """
    Phase 1: Data Acquisition and Preprocessing
    """
    print("=" * 50)
    print("PHASE 1: Data Preprocessing")
    print("=" * 50)
    
    preprocessor = LandsatPreprocessor(
        data_dir=config['data']['raw_dir'],
        output_dir=config['data']['processed_dir'],
        patch_size=config['data']['patch_size'],
        stride=config['data']['stride']
    )
    
    # Process all scenes in raw directory
    scene_dirs = [d for d in os.listdir(config['data']['raw_dir']) 
                  if os.path.isdir(os.path.join(config['data']['raw_dir'], d))]
    
    for scene_dir in scene_dirs:
        scene_path = os.path.join(config['data']['raw_dir'], scene_dir)
        preprocessor.process_scene(scene_path, scene_dir)
    
    print("Data preprocessing complete!")
    return config['data']['processed_dir']


def phase2_train_super_resolution(config):
    """
    Phase 2: Train Super-Resolution Module
    """
    print("=" * 50)
    print("PHASE 2: Super-Resolution Training")
    print("=" * 50)
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir=config['data']['processed_dir'],
        batch_size=config['data']['batch_size'],
        img_size=tuple(config['data']['img_size']),
        num_workers=config['data']['num_workers']
    )
    
    # Initialize model
    sr_model = MultiStageSuperResolution(in_channels=1, out_channels=1)
    
    # Train
    sr_config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 1e-4,
        'num_epochs': 20,
        'checkpoint_dir': config['logging']['checkpoint_dir'] + '_sr'
    }
    
    trainer = SuperResolutionTrainer(sr_model, dataloaders, sr_config)
    trainer.train(sr_config['num_epochs'])
    
    print("Super-resolution training complete!")
    return sr_model


def phase3_train_colorization(config, sr_model=None):
    """
    Phase 3: Train Thermal-to-RGB Colorization
    """
    print("=" * 50)
    print("PHASE 3: Colorization Training")
    print("=" * 50)
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir=config['data']['processed_dir'],
        batch_size=config['data']['batch_size'],
        img_size=tuple(config['data']['img_size']),
        num_workers=config['data']['num_workers']
    )
    
    # Initialize models
    generator = CoarseToFineGenerator(
        in_channels=1,
        out_channels=3,
        base_channels=config['model']['generator']['base_channels'],
        num_res_blocks=config['model']['generator']['num_res_blocks'],
        use_semantic=config['model']['generator']['use_semantic']
    )
    
    discriminator = PatchGANDiscriminator(
        input_channels=4,
        base_channels=config['model']['discriminator']['base_channels'],
        n_layers=config['model']['discriminator']['n_layers']
    )
    
    # Train
    train_config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'g_lr': config['training']['g_lr'],
        'd_lr': config['training']['d_lr'],
        'lambda_adv': config['training']['lambda_adv'],
        'lambda_perceptual': config['training']['lambda_perceptual'],
        'lambda_tv': config['training']['lambda_tv'],
        'lambda_semantic': config['training']['lambda_semantic'],
        'lr_step': config['training']['lr_step'],
        'lr_gamma': config['training']['lr_gamma'],
        'log_dir': config['logging']['log_dir'],
        'checkpoint_dir': config['logging']['checkpoint_dir']
    }
    
    trainer = ColorizationTrainer(generator, discriminator, dataloaders, train_config)
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    print("Colorization training complete!")
    return generator, discriminator


def phase4_evaluate(config, generator):
    """
    Phase 4: Evaluation
    """
    print("=" * 50)
    print("PHASE 4: Evaluation")
    print("=" * 50)
    
    # Create test dataloader
    dataloaders = create_dataloaders(
        data_dir=config['data']['processed_dir'],
        batch_size=1,
        img_size=tuple(config['data']['img_size']),
        num_workers=1
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    generator = generator.to(device)
    generator.eval()
    
    # Evaluate
    results = evaluate_model(generator, dataloaders['test'], device)
    
    print("\nEvaluation Results:")
    print(f"  PSNR: {results['psnr']:.2f} dB")
    print(f"  SSIM: {results['ssim']:.4f}")
    print(f"  PSNR Std: {results['psnr_std']:.2f}")
    print(f"  SSIM Std: {results['ssim_std']:.4f}")
    
    return results


def phase5_inference(config, generator, sr_model=None):
    """
    Phase 5: Inference
    """
    print("=" * 50)
    print("PHASE 5: Inference")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize colorizer
    colorizer = ThermalColorizer(
        generator=generator,
        sr_model=sr_model,
        device=device,
        img_size=tuple(config['data']['img_size'])
    )
    
    # Process single image
    test_image_path = os.path.join(config['data']['raw_dir'], 'test_thermal.tif')
    if os.path.exists(test_image_path):
        print(f"Processing image: {test_image_path}")
        result = colorizer.colorize_single(test_image_path)
        
        # Save result
        output_path = 'output/colorized_result.png'
        os.makedirs('output', exist_ok=True)
        cv2.imwrite(output_path, result)
        print(f"Result saved to: {output_path}")
    
    # Process full scene
    scene_path = os.path.join(config['data']['raw_dir'], 'test_scene.tif')
    if os.path.exists(scene_path):
        print(f"Processing scene: {scene_path}")
        result_scene = colorizer.colorize_scene(
            scene_path,
            'output/colorized_scene.tif',
            tile_size=config['inference']['tile_size'],
            overlap=config['inference']['overlap']
        )
        print(f"Scene saved to: output/colorized_scene.tif")
    
    print("Inference complete!")


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Thermal IR Colorization Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['all', 'preprocess', 'train_sr', 'train_colorization', 
                               'evaluate', 'inference'],
                       help='Phase to run')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for inference')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run specified phase
    if args.phase in ['all', 'preprocess']:
        processed_dir = phase1_data_preprocessing(config)
    
    if args.phase in ['all', 'train_sr']:
        sr_model = phase2_train_super_resolution(config)
    else:
        sr_model = None
    
    if args.phase in ['all', 'train_colorization']:
        generator, discriminator = phase3_train_colorization(config, sr_model)
    elif args.checkpoint:
        # Load from checkpoint
        generator = CoarseToFineGenerator()
        generator = load_model(args.checkpoint, generator)
    else:
        # Load best model from checkpoint directory
        checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model.pth')
        if os.path.exists(checkpoint_path):
            generator = CoarseToFineGenerator()
            generator = load_model(checkpoint_path, generator)
        else:
            raise FileNotFoundError("No checkpoint found. Please train or specify checkpoint.")
    
    if args.phase in ['all', 'evaluate']:
        phase4_evaluate(config, generator)
    
    if args.phase in ['all', 'inference']:
        phase5_inference(config, generator, sr_model)


if __name__ == '__main__':
    main()