#!/usr/bin/env python3
"""
Main Training Script for Multi-Language TTS
Simple entry point for the extensible TTS training system
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.training_manager import TTSTrainingManager


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Language TTS Training System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train VITS model with default config
  python train.py

  # Train specific model
  python train.py --model vits

  # Use custom configuration
  python train.py --config my_config.json

  # List available models
  python train.py --list-models

  # Create plugin template for new model
  python train.py --create-plugin fishaudio

  # Load plugins and train
  python train.py --load-plugins plugins --model orpheus
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.json",
        help="Configuration file path (default: config.json)"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        help="Model to train (default: from config)"
    )
    
    parser.add_argument(
        "--list-models", 
        action="store_true", 
        help="List all available models"
    )
    
    parser.add_argument(
        "--create-plugin", 
        type=str, 
        help="Create plugin template for specified model"
    )
    
    parser.add_argument(
        "--load-plugins", 
        type=str, 
        default="plugins",
        help="Load plugins from directory (default: plugins)"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Create training manager
        manager = TTSTrainingManager(args.config)
        
        # Load plugins
        if Path(args.load_plugins).exists():
            manager.load_plugins(args.load_plugins)
        
        # Handle commands
        if args.list_models:
            manager.list_available_models()
            return 0
        
        elif args.create_plugin:
            manager.create_plugin_template(args.create_plugin)
            return 0
        
        else:
            # Start training
            print("🚀 Multi-Language TTS Training System")
            print("=" * 50)
            
            success = manager.train_model(args.model)
            
            if success:
                print("\n✅ Training completed successfully!")
                print("🎵 Check the results/audio_outputs/ directory for test audio")
                return 0
            else:
                print("\n❌ Training failed!")
                return 1
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
