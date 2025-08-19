#!/usr/bin/env python3
"""
Training Manager for Multi-Language TTS Training
Orchestrates the entire training process with plugin support
"""

import logging
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional

from .core.model_registry import model_registry, get_available_models, create_trainer, get_model_info
from .core.base_trainer import DatasetManager

logger = logging.getLogger(__name__)


class TTSTrainingManager:
    """Main training manager for TTS models."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize training manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self._load_config(config_file)
        self.dataset_manager = DatasetManager()
        self._setup_logging()
        self._check_environment()
    
    def _load_config(self, config_file: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "datasets": {
                "khmer": {
                    "path": "data/km_kh_male",
                    "format": "tsv",
                    "language": "km"
                }
            },
            "training": {
                "model": "vits",
                "languages": ["km"],
                "output_path": "models/multilang_trained",
                "batch_size": 32,
                "num_epochs": 1000,
                "learning_rate": 2e-4,
                "save_every": 50,
                "eval_every": 25
            },
            "gpu": {
                "enabled": True,
                "mixed_precision": True,
                "optimize_batch_size": True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                
                # Merge configurations
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                        
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
                logger.info("Using default configuration")
        
        return default_config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _check_environment(self):
        """Check system environment and optimize settings."""
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Optimize batch size based on GPU memory
            if self.config["gpu"]["optimize_batch_size"]:
                if gpu_memory >= 40:
                    self.config["training"]["batch_size"] = 64
                elif gpu_memory >= 20:
                    self.config["training"]["batch_size"] = 32
                else:
                    self.config["training"]["batch_size"] = 16
                
                logger.info(f"Optimized batch size: {self.config['training']['batch_size']}")
        else:
            logger.warning("No GPU detected. Training will be slow on CPU.")
            self.config["gpu"]["enabled"] = False
            self.config["training"]["batch_size"] = 4
    
    def setup_datasets(self):
        """Setup datasets for training."""
        logger.info("Setting up datasets...")
        
        for name, dataset_config in self.config["datasets"].items():
            self.dataset_manager.add_dataset(
                language=dataset_config["language"],
                dataset_path=dataset_config["path"],
                format_type=dataset_config["format"]
            )
        
        # Validate datasets
        validation_results = self.dataset_manager.validate_all()
        
        for lang, is_valid in validation_results.items():
            if is_valid:
                logger.info(f"✅ {lang} dataset validated")
            else:
                logger.warning(f"❌ {lang} dataset validation failed")
        
        return all(validation_results.values())
    
    def list_available_models(self):
        """List all available TTS models."""
        models = get_available_models()
        
        print("\n🎯 Available TTS Models:")
        print("=" * 50)
        
        for model_name in models:
            info = get_model_info(model_name)
            if info:
                print(f"\n📦 {model_name.upper()}")
                print(f"   Name: {info.get('name', 'N/A')}")
                print(f"   Description: {info.get('description', 'N/A')}")
                print(f"   Languages: {', '.join(info.get('supported_languages', []))}")
                print(f"   Quality: {info.get('quality', 'N/A')}")
                print(f"   Speed: {info.get('speed', 'N/A')}")
                print(f"   GPU Required: {info.get('gpu_required', 'N/A')}")
        
        print()
    
    def train_model(self, model_name: Optional[str] = None) -> bool:
        """
        Train a TTS model.
        
        Args:
            model_name: Name of the model to train (uses config if None)
            
        Returns:
            bool: Success status
        """
        if model_name is None:
            model_name = self.config["training"]["model"]
        
        logger.info(f"Starting training for model: {model_name}")
        
        # Check if model is available
        if model_name not in get_available_models():
            logger.error(f"Model '{model_name}' not available")
            self.list_available_models()
            return False
        
        # Setup datasets
        if not self.setup_datasets():
            logger.error("Dataset setup failed")
            return False
        
        # Create trainer
        trainer = create_trainer(
            model_name=model_name,
            dataset_path=self.config["datasets"]["khmer"]["path"],  # Primary dataset
            output_path=self.config["training"]["output_path"],
            config=self.config["training"]
        )
        
        if not trainer:
            logger.error(f"Failed to create trainer for {model_name}")
            return False
        
        # Prepare datasets
        languages = self.config["training"]["languages"]
        if not trainer.prepare_dataset(languages):
            logger.error("Dataset preparation failed")
            return False
        
        # Start training
        logger.info("🚀 Starting training process...")
        success = trainer.train()
        
        if success:
            logger.info("✅ Training completed successfully!")
            
            # Test the model
            test_texts = [
                "សួស្តី! នេះជាការសាកល្បងសម្រាប់ប្រព័ន្ធបំលែងអត្ថបទទៅជាសំឡេង។",  # Khmer
                "Xin chào! Đây là bài kiểm tra cho hệ thống chuyển đổi văn bản thành giọng nói."  # Vietnamese
            ]
            
            logger.info("🎵 Testing trained model...")
            trainer.test_model(test_texts)
            
        else:
            logger.error("❌ Training failed!")
        
        return success
    
    def load_plugins(self, plugins_dir: str = "plugins"):
        """Load model plugins from directory."""
        logger.info(f"Loading plugins from {plugins_dir}...")
        model_registry.load_plugins_from_directory(plugins_dir)
    
    def create_plugin_template(self, model_name: str, output_path: str = "plugins"):
        """Create a plugin template for a new model."""
        from .core.model_registry import PLUGIN_TEMPLATE
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        plugin_file = output_dir / f"{model_name}_trainer.py"
        
        # Customize template
        template = PLUGIN_TEMPLATE.replace("yourmodel", model_name.lower())
        template = template.replace("YourModelTrainer", f"{model_name.title()}Trainer")
        template = template.replace("Your Model Name", f"{model_name.title()} TTS")
        
        with open(plugin_file, 'w', encoding='utf-8') as f:
            f.write(template)
        
        logger.info(f"Plugin template created: {plugin_file}")
        print(f"📝 Plugin template created at: {plugin_file}")
        print(f"Edit this file to implement your {model_name} trainer!")
    
    def save_config(self, output_path: str = "config.json"):
        """Save current configuration to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to {output_path}")


def main():
    """Main entry point for training manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Language TTS Training Manager")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--model", type=str, help="Model to train")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--create-plugin", type=str, help="Create plugin template for model")
    parser.add_argument("--load-plugins", type=str, help="Load plugins from directory")
    
    args = parser.parse_args()
    
    # Create training manager
    manager = TTSTrainingManager(args.config)
    
    # Load plugins if specified
    if args.load_plugins:
        manager.load_plugins(args.load_plugins)
    
    # Handle commands
    if args.list_models:
        manager.list_available_models()
    elif args.create_plugin:
        manager.create_plugin_template(args.create_plugin)
    else:
        # Start training
        success = manager.train_model(args.model)
        exit(0 if success else 1)


if __name__ == "__main__":
    main()
