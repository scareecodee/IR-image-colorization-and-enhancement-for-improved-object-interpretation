"""
Main Entry Point for Thermal IR Colorization Pipeline
Complete end-to-end pipeline with all modules
"""

import os
import sys
import argparse
import yaml
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.preprocessing import LandsatPreprocessor
from data.dataset import create_dataloaders
from models.generator import CoarseToFineGenerator
from models.discriminator import PatchGANDiscriminator
from training.trainer import ColorizationTrainer


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config):
    """Phase 1: Data Preparation"""
    print("\n" + "="*60)
    print("PHASE 1: Data Preparation")
    print("="*60)
    
    preprocessor = LandsatPreprocessor(
        raw_dir=config['data']['raw_dir'],
        output_dir=config['data']['processed_dir'],
        patch_size=config['data']['patch_size'],
        stride=config['data']['stride']
    )
    
    # Find all scenes
    scene_dirs = [d for d in os.listdir(config['data']['raw_dir'])
                  if os.path.isdir(os.path.join(config['data']['raw_dir'], d))]
    
    if not scene_dirs:
        print("❌ No scenes found in raw directory!")
        print(f"   Please place Landsat scenes in: {config['data']['raw_dir']}")
        return
    
    total_patches = 0
    for scene_dir in scene_dirs:
        scene_path = os.path.join(config['data']['raw_dir'], scene_dir)
        patches = preprocessor.process_scene(scene_path, scene_dir)
        total_patches += patches
    
    print(f"\n✅ Data preparation complete!")
    print(f"   Total patches: {total_patches}")
    print(f"   Output directory: {config['data']['processed_dir']}")


def train_model(config):
    """Phase 2: Model Training"""
    print("\n" + "="*60)
    print("PHASE 2: Model Training")
    print("="*60)
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_dir=config['data']['processed_dir'],
        batch_size=config['data']['batch_size'],
        img_size=tuple(config['data']['img_size']),
        num_workers=config['data']['num_workers']
    )
    
    if len(dataloaders['train'].dataset) == 0:
        print("❌ No training data found!")
        print("   Please run data preparation first.")
        return
    
    # Initialize models
    generator = CoarseToFineGenerator(
        in_channels=config['model']['generator']['in_channels'],
        out_channels=config['model']['generator']['out_channels'],
        base_channels=config['model']['generator']['base_channels'],
        num_res_blocks=config['model']['generator']['num_res_blocks']
    )
    
    discriminator = PatchGANDiscriminator(
        input_channels=config['model']['discriminator']['input_channels'],
        base_channels=config['model']['discriminator']['base_channels'],
        n_layers=config['model']['discriminator']['n_layers']
    )
    
    # Training config
    train_config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'g_lr': config['training']['g_lr'],
        'd_lr': config['training']['d_lr'],
        'lambda_adv': config['training']['lambda_adv'],
        'lambda_perceptual': config['training']['lambda_perceptual'],
        'lambda_tv': config['training']['lambda_tv'],
        'lr_step': config['training']['lr_step'],
        'lr_gamma': config['training']['lr_gamma'],
        'log_dir': config['logging']['log_dir'],
        'checkpoint_dir': config['logging']['checkpoint_dir']
    }
    
    print(f"🚀 Using device: {train_config['device']}")
    print(f"📊 Training samples: {len(dataloaders['train'].dataset)}")
    print(f"📊 Validation samples: {len(dataloaders['val'].dataset)}")
    
    # Train
    trainer = ColorizationTrainer(
        generator, discriminator, dataloaders, train_config
    )
    trainer.train(num_epochs=config['training']['num_epochs'])
    
    print("\n✅ Training complete!")
    print(f"   Best model saved to: {config['logging']['checkpoint_dir']}/best_model.pth")


def evaluate_model(config):
    """Phase 3: Evaluation"""
    print("\n" + "="*60)
    print("PHASE 3: Evaluation")
    print("="*60)
    
    from evaluation.evaluate import main as evaluate_main
    evaluate_main()


def run_inference(config):
    """Phase 4: Inference"""
    print("\n" + "="*60)
    print("PHASE 4: Inference")
    print("="*60)
    
    from inference.predict import main as predict_main
    predict_main()


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description='Thermal IR Colorization Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['all', 'preprocess', 'train', 'evaluate', 'inference'],
                       help='Phase to run')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint path for inference')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create directories
    os.makedirs(config['data']['raw_dir'], exist_ok=True)
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    os.makedirs(config['logging']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['output_dir'], exist_ok=True)
    
    # Run phases
    if args.phase in ['all', 'preprocess']:
        prepare_data(config)
    
    if args.phase in ['all', 'train']:
        train_model(config)
    
    if args.phase in ['all', 'evaluate']:
        evaluate_model(config)
    
    if args.phase in ['all', 'inference']:
        run_inference(config)
    
    print("\n✅ Pipeline complete!")


if __name__ == '__main__':
    main()