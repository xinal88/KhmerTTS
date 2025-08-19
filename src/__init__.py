"""
Multi-Language TTS Training System
Extensible plugin-based architecture for TTS model training
"""

__version__ = "1.0.0"
__author__ = "TTS Training System"
__description__ = "Extensible multi-language TTS training system"

from .training_manager import TTSTrainingManager
from .core.model_registry import model_registry, get_available_models, create_trainer

__all__ = [
    "TTSTrainingManager",
    "model_registry", 
    "get_available_models",
    "create_trainer"
]
