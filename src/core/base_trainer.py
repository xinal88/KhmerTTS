#!/usr/bin/env python3
"""
Base Trainer Class for Multi-Language TTS Training
Extensible architecture for adding new TTS models easily
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

logger = logging.getLogger(__name__)


class BaseTTSTrainer(ABC):
    """
    Abstract base class for TTS trainers.
    All TTS model trainers should inherit from this class.
    """
    
    def __init__(self, 
                 model_name: str,
                 dataset_path: str,
                 output_path: str,
                 config: Optional[Dict] = None):
        """
        Initialize base trainer.
        
        Args:
            model_name: Name of the TTS model (e.g., 'vits', 'orpheus', 'styletts')
            dataset_path: Path to the dataset
            output_path: Path to save trained model
            config: Model-specific configuration
        """
        self.model_name = model_name
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.config = config or {}
        
        # Common configuration
        self.base_config = {
            "sample_rate": 22050,
            "batch_size": 16,
            "learning_rate": 2e-4,
            "num_epochs": 1000,
            "save_every": 50,
            "eval_every": 25,
            "use_gpu": True,
            "mixed_precision": True
        }
        
        # Merge with model-specific config
        self.config = {**self.base_config, **self.config}
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for the trainer."""
        log_dir = self.output_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{self.model_name}_training.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(file_handler)
    
    @abstractmethod
    def prepare_dataset(self, languages: List[str]) -> bool:
        """
        Prepare dataset for training.
        
        Args:
            languages: List of language codes (e.g., ['km', 'vi'])
            
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def create_model_config(self) -> Dict:
        """
        Create model-specific configuration.
        
        Returns:
            Dict: Model configuration
        """
        pass
    
    @abstractmethod
    def train(self) -> bool:
        """
        Start training process.
        
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def test_model(self, test_texts: List[str]) -> bool:
        """
        Test the trained model.
        
        Args:
            test_texts: List of texts to test
            
        Returns:
            bool: Success status
        """
        pass
    
    def validate_dataset(self, metadata_file: Path, audio_dir: Path) -> Tuple[bool, Dict]:
        """
        Validate dataset structure and content.
        
        Args:
            metadata_file: Path to metadata file
            audio_dir: Path to audio directory
            
        Returns:
            Tuple[bool, Dict]: (is_valid, statistics)
        """
        if not metadata_file.exists():
            logger.error(f"Metadata file not found: {metadata_file}")
            return False, {}
        
        if not audio_dir.exists():
            logger.error(f"Audio directory not found: {audio_dir}")
            return False, {}
        
        # Read metadata
        try:
            if metadata_file.suffix == '.tsv':
                df = pd.read_csv(metadata_file, sep='\t', header=None)
            elif metadata_file.suffix == '.csv':
                df = pd.read_csv(metadata_file)
            else:
                df = pd.read_csv(metadata_file, sep='|', header=None)
            
            stats = {
                "total_samples": len(df),
                "metadata_file": str(metadata_file),
                "audio_dir": str(audio_dir),
                "valid": True
            }
            
            logger.info(f"Dataset validation successful: {stats}")
            return True, stats
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False, {}
    
    def clean_text(self, text: str, language: str = "km") -> str:
        """
        Clean text for TTS training.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Language-specific cleaning
        if language == "km":  # Khmer
            import re
            khmer_pattern = r'[\u1780-\u17FF\s.,!?;:\-\(\)\[\]"]'
            text = ''.join(re.findall(khmer_pattern, text))
        elif language == "vi":  # Vietnamese
            import re
            vietnamese_pattern = r'[a-zA-ZàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ\s.,!?;:\-\(\)\[\]"]'
            text = ''.join(re.findall(vietnamese_pattern, text))
        
        return text.strip()
    
    def split_dataset(self, samples: List[Dict], val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """
        Split dataset into training and validation sets.
        
        Args:
            samples: List of sample dictionaries
            val_ratio: Validation set ratio
            
        Returns:
            Tuple[List[Dict], List[Dict]]: (train_samples, val_samples)
        """
        import random
        random.seed(42)
        
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        val_size = int(len(shuffled) * val_ratio)
        val_samples = shuffled[:val_size]
        train_samples = shuffled[val_size:]
        
        logger.info(f"Dataset split: {len(train_samples)} train, {len(val_samples)} validation")
        
        return train_samples, val_samples
    
    def save_dataset_info(self, train_samples: List[Dict], val_samples: List[Dict], languages: List[str]):
        """Save dataset information."""
        dataset_info = {
            "model_name": self.model_name,
            "languages": languages,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "total_samples": len(train_samples) + len(val_samples),
            "sample_rate": self.config["sample_rate"],
            "config": self.config
        }
        
        info_file = self.output_path / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset info saved to {info_file}")
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "name": self.model_name,
            "version": "1.0",
            "description": f"{self.model_name.upper()} TTS model trainer",
            "supported_languages": ["km", "vi", "en"],
            "config": self.config
        }


class DatasetManager:
    """Manages multiple datasets for multi-language training."""
    
    def __init__(self):
        self.datasets = {}
    
    def add_dataset(self, language: str, dataset_path: str, format_type: str = "auto"):
        """
        Add a dataset for a specific language.
        
        Args:
            language: Language code (e.g., 'km', 'vi')
            dataset_path: Path to dataset
            format_type: Dataset format ('tsv', 'csv', 'ljspeech', 'auto')
        """
        self.datasets[language] = {
            "path": Path(dataset_path),
            "format": format_type,
            "language": language
        }
        logger.info(f"Added {language} dataset: {dataset_path}")
    
    def get_dataset(self, language: str) -> Optional[Dict]:
        """Get dataset information for a language."""
        return self.datasets.get(language)
    
    def list_datasets(self) -> List[str]:
        """List all available datasets."""
        return list(self.datasets.keys())
    
    def validate_all(self) -> Dict[str, bool]:
        """Validate all datasets."""
        results = {}
        for lang, dataset in self.datasets.items():
            # Basic validation - check if path exists
            results[lang] = dataset["path"].exists()
        return results
