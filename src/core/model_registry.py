#!/usr/bin/env python3
"""
Model Registry for TTS Training System
Plugin-based architecture for easy model integration
"""

import importlib
import logging
from typing import Dict, List, Type, Optional
from pathlib import Path

from .base_trainer import BaseTTSTrainer

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for TTS model trainers."""
    
    def __init__(self):
        self._models: Dict[str, Type[BaseTTSTrainer]] = {}
        self._model_info: Dict[str, Dict] = {}
        self._load_builtin_models()
    
    def _load_builtin_models(self):
        """Load built-in model trainers."""
        try:
            # Load VITS trainer
            from ..models.vits_trainer import VITSTrainer
            self.register_model("vits", VITSTrainer, {
                "name": "VITS",
                "description": "Variational Inference with adversarial learning for end-to-end Text-to-Speech",
                "supported_languages": ["km", "vi", "en"],
                "quality": "high",
                "speed": "medium",
                "gpu_required": True
            })
            
        except ImportError as e:
            logger.warning(f"Failed to load VITS trainer: {e}")
    
    def register_model(self, 
                      model_name: str, 
                      trainer_class: Type[BaseTTSTrainer], 
                      model_info: Dict):
        """
        Register a new TTS model trainer.
        
        Args:
            model_name: Unique model identifier
            trainer_class: Trainer class that inherits from BaseTTSTrainer
            model_info: Model information dictionary
        """
        if not issubclass(trainer_class, BaseTTSTrainer):
            raise ValueError(f"Trainer class must inherit from BaseTTSTrainer")
        
        self._models[model_name] = trainer_class
        self._model_info[model_name] = model_info
        
        logger.info(f"Registered model: {model_name}")
    
    def get_model(self, model_name: str) -> Optional[Type[BaseTTSTrainer]]:
        """Get a model trainer class by name."""
        return self._models.get(model_name)
    
    def list_models(self) -> List[str]:
        """List all registered models."""
        return list(self._models.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model."""
        return self._model_info.get(model_name)
    
    def get_all_model_info(self) -> Dict[str, Dict]:
        """Get information about all registered models."""
        return self._model_info.copy()
    
    def create_trainer(self, 
                      model_name: str, 
                      dataset_path: str,
                      output_path: str = None,
                      config: Dict = None) -> Optional[BaseTTSTrainer]:
        """
        Create a trainer instance for the specified model.
        
        Args:
            model_name: Name of the model
            dataset_path: Path to dataset
            output_path: Output path for trained model
            config: Model configuration
            
        Returns:
            BaseTTSTrainer instance or None if model not found
        """
        trainer_class = self.get_model(model_name)
        if not trainer_class:
            logger.error(f"Model '{model_name}' not found in registry")
            return None
        
        if output_path is None:
            output_path = f"models/{model_name}_trained"
        
        try:
            return trainer_class(
                dataset_path=dataset_path,
                output_path=output_path,
                config=config
            )
        except Exception as e:
            logger.error(f"Failed to create trainer for {model_name}: {e}")
            return None
    
    def load_plugin(self, plugin_path: str):
        """
        Load a model plugin from a Python file.
        
        Args:
            plugin_path: Path to the plugin Python file
        """
        plugin_path = Path(plugin_path)
        
        if not plugin_path.exists():
            logger.error(f"Plugin file not found: {plugin_path}")
            return
        
        try:
            # Import the plugin module
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            plugin_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plugin_module)
            
            # Look for register_plugin function
            if hasattr(plugin_module, 'register_plugin'):
                plugin_module.register_plugin(self)
                logger.info(f"Loaded plugin: {plugin_path}")
            else:
                logger.error(f"Plugin {plugin_path} missing register_plugin function")
                
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
    
    def load_plugins_from_directory(self, plugins_dir: str):
        """
        Load all plugins from a directory.
        
        Args:
            plugins_dir: Directory containing plugin files
        """
        plugins_dir = Path(plugins_dir)
        
        if not plugins_dir.exists():
            logger.warning(f"Plugins directory not found: {plugins_dir}")
            return
        
        for plugin_file in plugins_dir.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            self.load_plugin(plugin_file)


# Global model registry instance
model_registry = ModelRegistry()


def register_model(model_name: str, trainer_class: Type[BaseTTSTrainer], model_info: Dict):
    """Convenience function to register a model."""
    model_registry.register_model(model_name, trainer_class, model_info)


def get_available_models() -> List[str]:
    """Get list of available models."""
    return model_registry.list_models()


def create_trainer(model_name: str, 
                  dataset_path: str,
                  output_path: str = None,
                  config: Dict = None) -> Optional[BaseTTSTrainer]:
    """Convenience function to create a trainer."""
    return model_registry.create_trainer(model_name, dataset_path, output_path, config)


def get_model_info(model_name: str = None) -> Dict:
    """Get model information."""
    if model_name:
        return model_registry.get_model_info(model_name)
    else:
        return model_registry.get_all_model_info()


# Example plugin template (for documentation)
PLUGIN_TEMPLATE = '''#!/usr/bin/env python3
"""
Example TTS Model Plugin Template
Copy this template to create new model plugins
"""

import logging
from pathlib import Path
from typing import Dict, List

from src.core.base_trainer import BaseTTSTrainer

logger = logging.getLogger(__name__)


class YourModelTrainer(BaseTTSTrainer):
    """Your custom TTS model trainer."""
    
    def __init__(self, dataset_path: str, output_path: str = "models/yourmodel_trained", config: Dict = None):
        # Your model-specific configuration
        your_config = {
            "model_name": "yourmodel",
            "batch_size": 16,
            "learning_rate": 1e-4,
            # Add your model-specific parameters here
        }
        
        if config:
            your_config.update(config)
        
        super().__init__(
            model_name="yourmodel",
            dataset_path=dataset_path,
            output_path=output_path,
            config=your_config
        )
    
    def prepare_dataset(self, languages: List[str]) -> bool:
        """Implement dataset preparation for your model."""
        # Your dataset preparation logic here
        pass
    
    def create_model_config(self) -> Dict:
        """Create your model configuration."""
        # Your model configuration logic here
        pass
    
    def train(self) -> bool:
        """Implement training logic for your model."""
        # Your training logic here
        pass
    
    def test_model(self, test_texts: List[str]) -> bool:
        """Implement model testing."""
        # Your testing logic here
        pass


def register_plugin(registry):
    """Register this plugin with the model registry."""
    registry.register_model(
        "yourmodel",
        YourModelTrainer,
        {
            "name": "Your Model Name",
            "description": "Description of your TTS model",
            "supported_languages": ["km", "vi", "en"],
            "quality": "high",
            "speed": "fast",
            "gpu_required": False
        }
    )
'''
