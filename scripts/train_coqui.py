#!/usr/bin/env python3
"""
Coqui TTS Training Script for Khmer Dataset

This script sets up and runs training for Coqui TTS using
the Khmer male voice dataset in data/km_kh_male/
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoquiKhmerTrainer:
    """Trainer class for Coqui TTS with Khmer dataset."""
    
    def __init__(self, dataset_path: str = "data/km_kh_male"):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path("models/coqui_khmer_trained")
        
        # Training configuration (optimized for high-end GPU - RTX 5880 Ada)
        self.config = {
            "model_name": "vits",      # Use VITS model for high quality
            "language": "km",          # Khmer language code
            "speaker_name": "khmer_male_01",
            "sample_rate": 22050,
            "max_audio_length": 10.0,  # Increased for better quality
            "min_audio_length": 1.0,   # seconds
            "batch_size": 32,          # Large batch size for GPU
            "learning_rate": 2e-4,     # Standard VITS learning rate
            "num_epochs": 1000,        # More epochs for better quality
            "save_every": 50,          # Save checkpoints every 50 epochs
            "eval_every": 25,          # Evaluate every 25 epochs
            "text_cleaners": ["basic_cleaners"],
            "use_phonemes": False,     # Can be enabled for better pronunciation
            "cpu_only": False,         # Use GPU training
            "max_samples": None        # Use full dataset
        }
    
    def prepare_dataset(self) -> bool:
        """Prepare the Khmer dataset for Coqui training."""
        logger.info("Preparing Khmer dataset for Coqui TTS training...")
        
        # Check if dataset exists
        if not self.dataset_path.exists():
            logger.error(f"Dataset not found at {self.dataset_path}")
            return False
        
        metadata_file = self.dataset_path / "line_index.tsv"
        audio_dir = self.dataset_path / "wavs"
        
        if not metadata_file.exists() or not audio_dir.exists():
            logger.error("Dataset structure incomplete. Need line_index.tsv and wavs/ directory")
            return False
        
        # Read metadata
        logger.info("Reading metadata...")
        try:
            # Read TSV file
            df = pd.read_csv(metadata_file, sep='\t', header=None, 
                           names=['filename', 'empty', 'transcription'])
            df = df[['filename', 'transcription']].dropna()
            logger.info(f"Found {len(df)} samples in metadata")
            
        except Exception as e:
            logger.error(f"Error reading metadata: {e}")
            return False
        
        # Validate and process audio files
        logger.info("Processing audio files...")
        valid_samples = []
        
        for idx, row in df.iterrows():
            audio_file = audio_dir / f"{row['filename']}.wav"
            
            if audio_file.exists():
                try:
                    import librosa
                    duration = librosa.get_duration(path=str(audio_file))
                    
                    if (self.config['min_audio_length'] <= duration <= 
                        self.config['max_audio_length']):
                        
                        # Clean text
                        text = self._clean_text(row['transcription'])
                        if text:  # Only add if text is not empty after cleaning
                            valid_samples.append({
                                'audio_file': str(audio_file.relative_to(Path.cwd())),
                                'text': text,
                                'speaker_name': self.config['speaker_name'],
                                'language': self.config['language'],
                                'duration': duration
                            })
                    
                except Exception as e:
                    logger.debug(f"Error processing {audio_file.name}: {e}")
        
        logger.info(f"Valid samples after processing: {len(valid_samples)}")
        
        if len(valid_samples) < 100:
            logger.warning("Very few valid samples. Training may not be effective.")
        
        # Create training/validation split
        train_samples, val_samples = self._split_dataset(valid_samples)
        
        # Save in Coqui format
        self._save_coqui_dataset(train_samples, val_samples)
        
        return True
    
    def _clean_text(self, text: str) -> str:
        """Clean Khmer text for TTS training."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Basic Khmer text validation
        import re
        khmer_pattern = r'[\u1780-\u17FF\s.,!?;:\-\(\)\[\]"]'
        cleaned = ''.join(re.findall(khmer_pattern, text))
        
        return cleaned.strip()
    
    def _split_dataset(self, samples: List[Dict], val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """Split dataset into training and validation sets."""
        import random
        random.seed(42)
        
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        val_size = int(len(shuffled) * val_ratio)
        val_samples = shuffled[:val_size]
        train_samples = shuffled[val_size:]
        
        logger.info(f"Dataset split: {len(train_samples)} train, {len(val_samples)} validation")
        
        return train_samples, val_samples
    
    def _save_coqui_dataset(self, train_samples: List[Dict], val_samples: List[Dict]):
        """Save dataset in Coqui TTS format."""
        output_dir = Path("data/processed/coqui_khmer")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata files in Coqui format
        # Format: audio_file|text|speaker_name
        
        # Training metadata
        train_file = output_dir / "metadata_train.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            for sample in train_samples:
                f.write(f"{sample['audio_file']}|{sample['text']}|{sample['speaker_name']}\n")
        
        # Validation metadata
        val_file = output_dir / "metadata_val.txt"
        with open(val_file, 'w', encoding='utf-8') as f:
            for sample in val_samples:
                f.write(f"{sample['audio_file']}|{sample['text']}|{sample['speaker_name']}\n")
        
        # Save dataset info
        dataset_info = {
            "name": "khmer_male_tts",
            "version": "1.0",
            "description": "Khmer male voice TTS dataset",
            "language": "km",
            "speakers": [self.config['speaker_name']],
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "sample_rate": self.config['sample_rate']
        }
        
        info_file = output_dir / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Coqui dataset saved to {output_dir}")
    
    def create_coqui_config(self) -> bool:
        """Create Coqui TTS training configuration."""
        logger.info("Creating Coqui TTS configuration...")
        
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Coqui TTS configuration
        config = {
            "model": "vits",  # Use VITS model for single speaker
            "run_name": "khmer_male_vits",
            "run_description": "Khmer male voice TTS training with VITS",
            
            # Dataset configuration
            "datasets": [{
                "name": "khmer_male",
                "path": "data/processed/coqui_khmer/",
                "meta_file_train": "metadata_train.txt",
                "meta_file_val": "metadata_val.txt",
                "language": "km"
            }],
            
            # Audio configuration
            "audio": {
                "sample_rate": self.config['sample_rate'],
                "resample": True,
                "do_trim_silence": True,
                "trim_db": 23,
                "signal_norm": True,
                "symmetric_norm": True,
                "max_norm": 4.0,
                "clip_norm": True,
                "mel_fmin": 0,
                "mel_fmax": None,
                "spec_gain": 1.0,
                "do_sound_norm": False
            },
            
            # VITS Model configuration
            "model_args": {
                "num_chars": 300,  # Increased for Khmer characters
                "num_speakers": 1,
                "text_cleaner": "basic_cleaners",
                "use_phonemes": self.config['use_phonemes'],
                "phoneme_language": "km" if self.config['use_phonemes'] else None,
                "compute_input_seq_cache": True,
                "precompute_num_workers": 8,  # More workers for GPU
                "use_speaker_embedding": False,
                # VITS specific parameters
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0.1,
                "resblock": "1",
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [8, 8, 2, 2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16, 16, 4, 4],
                "use_spectral_norm": False
            },
            
            # Training configuration (optimized for GPU)
            "batch_size": self.config['batch_size'],
            "eval_batch_size": self.config['batch_size'],
            "num_loader_workers": 8,   # More workers for GPU
            "num_eval_loader_workers": 4,
            "run_eval": True,
            "test_delay_epochs": -1,

            # Optimizer (VITS specific)
            "epochs": self.config['num_epochs'],
            "lr": self.config['learning_rate'],
            "lr_scheduler": "ExponentialLR",
            "lr_scheduler_params": {"gamma": 0.999875},
            "use_grad_clip": True,
            "grad_clip": 5.0,

            # VITS specific training parameters
            "use_weighted_sampler": True,
            "weighted_sampler_attrs": {"language": 1.0, "speaker_name": 1.0},
            "weighted_sampler_multipliers": {},
            "r": 1,  # Reduction factor
            "add_blank": True,
            
            # Logging and saving
            "print_step": 25,
            "plot_step": 100,
            "save_step": self.config['save_every'],
            "save_n_checkpoints": 5,
            "save_checkpoints": True,
            "target_loss": "loss_1",
            "print_eval": False,
            
            # Paths
            "output_path": str(self.output_path),
            
            # Mixed precision training (great for RTX GPUs)
            "mixed_precision": True,
            "use_cuda": True,
            "cudnn_enabled": True,
            "cudnn_benchmark": True,
            
            # Test sentences for evaluation
            "test_sentences": [
                "សួស្តី! នេះជាការសាកល្បងសម្រាប់ប្រព័ន្ធបំលែងអត្ថបទទៅជាសំឡេង។",
                "ប្រទេសកម្ពុជាមានវប្បធម៌ដ៏សម្បូរបែប និងប្រវត្តិសាស្ត្រយ៉ាងយូរលង់។",
                "បច្ចេកវិទ្យាថ្មីៗកំពុងផ្លាស់ប្តូរជីវិតរបស់យើងគ្រប់ៗថ្ងៃ។"
            ]
        }
        
        config_file = self.output_path / "config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Coqui configuration saved to {config_file}")
        return True
    
    def create_training_script(self):
        """Create training script for Coqui TTS."""
        script_content = f'''#!/bin/bash
# Coqui TTS Training Script for Khmer Dataset
# Generated automatically

echo "Starting Coqui TTS training for Khmer..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU, change as needed

# Install Coqui TTS if not already installed
pip install TTS

# Run training
python -m TTS.bin.train_tts \\
    --config_path "{self.output_path}/config.json" \\
    --restore_path "" \\
    --continue_path ""

echo "Training completed! Check {self.output_path} for results."

# Test the trained model
echo "Testing trained model..."
python -m TTS.bin.synthesize \\
    --model_path "{self.output_path}/best_model.pth" \\
    --config_path "{self.output_path}/config.json" \\
    --text "សួស្តី! នេះជាការសាកល្បងសម្រាប់ប្រព័ន្ធបំលែងអត្ថបទទៅជាសំឡេង។" \\
    --out_path "{self.output_path}/test_output.wav"

echo "Test synthesis completed! Check {self.output_path}/test_output.wav"
'''
        
        script_file = self.output_path / "train_coqui_khmer.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_file, 0o755)
        
        logger.info(f"Training script created: {script_file}")
    
    def start_training(self) -> bool:
        """Start Coqui TTS training."""
        logger.info("Starting Coqui TTS training...")
        
        try:
            from TTS.bin.train_tts import main as train_tts
            
            # Training arguments
            args = [
                "--config_path", str(self.output_path / "config.json"),
                "--restore_path", "",
                "--continue_path", ""
            ]
            
            # Start training
            train_tts(args)
            return True
            
        except ImportError:
            logger.error("Coqui TTS not installed. Install with: pip install TTS")
            return False
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False


def main():
    """Main training function."""
    print("🎯 Coqui TTS Training for Khmer Dataset")
    print("=" * 50)
    
    trainer = CoquiKhmerTrainer()
    
    # Step 1: Prepare dataset
    print("\n📊 Step 1: Preparing dataset...")
    if not trainer.prepare_dataset():
        print("❌ Dataset preparation failed!")
        return 1
    
    # Step 2: Create configuration
    print("\n🔧 Step 2: Creating training configuration...")
    if not trainer.create_coqui_config():
        print("❌ Configuration creation failed!")
        return 1
    
    # Step 3: Create training script
    print("\n📝 Step 3: Creating training script...")
    trainer.create_training_script()
    
    # Step 4: Start training
    print("\n🚀 Step 4: Starting training...")
    print("⚠️  Note: Training will take several hours/days depending on your hardware")
    
    if not trainer.start_training():
        print("⚠️  Automatic training failed. Use the manual script instead:")
        print(f"   bash {trainer.output_path}/train_coqui_khmer.sh")
    
    print("\n✅ Training setup completed!")
    print(f"📁 Output directory: {trainer.output_path}")
    print(f"📊 Processed data: data/processed/coqui_khmer/")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
