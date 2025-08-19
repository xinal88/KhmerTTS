#!/usr/bin/env python3
"""
VITS Trainer Implementation
Inherits from BaseTTSTrainer for easy integration
"""

import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from ..core.base_trainer import BaseTTSTrainer, DatasetManager

logger = logging.getLogger(__name__)


class VITSTrainer(BaseTTSTrainer):
    """VITS model trainer implementation."""
    
    def __init__(self, 
                 dataset_path: str,
                 output_path: str = "models/vits_trained",
                 config: Optional[Dict] = None):
        
        # VITS-specific default configuration
        vits_config = {
            "model_name": "vits",
            "batch_size": 32,  # Larger batch for VITS
            "learning_rate": 2e-4,
            "num_epochs": 1000,
            "save_every": 50,
            "eval_every": 25,
            "max_audio_length": 10.0,
            "min_audio_length": 1.0,
            "use_phonemes": False,
            "mixed_precision": True,
            
            # VITS architecture parameters
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
        }
        
        if config:
            vits_config.update(config)
        
        super().__init__(
            model_name="vits",
            dataset_path=dataset_path,
            output_path=output_path,
            config=vits_config
        )
        
        self.dataset_manager = DatasetManager()
    
    def prepare_dataset(self, languages: List[str]) -> bool:
        """Prepare datasets for VITS training."""
        logger.info(f"Preparing datasets for languages: {languages}")
        
        all_samples = []
        
        for language in languages:
            samples = self._process_language_dataset(language)
            if samples:
                all_samples.extend(samples)
                logger.info(f"Added {len(samples)} samples for {language}")
            else:
                logger.warning(f"No valid samples found for {language}")
        
        if not all_samples:
            logger.error("No valid samples found in any dataset")
            return False
        
        # Split dataset
        train_samples, val_samples = self.split_dataset(all_samples)
        
        # Save in VITS format
        self._save_vits_dataset(train_samples, val_samples, languages)
        
        # Save dataset info
        self.save_dataset_info(train_samples, val_samples, languages)
        
        return True
    
    def _process_language_dataset(self, language: str) -> List[Dict]:
        """Process dataset for a specific language."""
        if language == "km":  # Khmer
            return self._process_khmer_dataset()
        elif language == "vi":  # Vietnamese (FOSD)
            return self._process_vietnamese_dataset()
        else:
            logger.warning(f"Unsupported language: {language}")
            return []
    
    def _process_khmer_dataset(self) -> List[Dict]:
        """Process Khmer dataset."""
        dataset_path = Path("data/km_kh_male")
        metadata_file = dataset_path / "line_index.tsv"
        audio_dir = dataset_path / "wavs"
        
        is_valid, stats = self.validate_dataset(metadata_file, audio_dir)
        if not is_valid:
            return []
        
        # Read metadata
        df = pd.read_csv(metadata_file, sep='\t', header=None, 
                        names=['filename', 'empty', 'transcription'])
        df = df[['filename', 'transcription']].dropna()
        
        samples = []
        for _, row in df.iterrows():
            audio_file = audio_dir / f"{row['filename']}.wav"
            
            if audio_file.exists():
                try:
                    import librosa
                    duration = librosa.get_duration(path=str(audio_file))
                    
                    if (self.config['min_audio_length'] <= duration <= 
                        self.config['max_audio_length']):
                        
                        text = self.clean_text(row['transcription'], "km")
                        if text:
                            samples.append({
                                'audio_file': str(audio_file.relative_to(Path.cwd())),
                                'text': text,
                                'speaker_name': 'khmer_male_01',
                                'language': 'km',
                                'duration': duration
                            })
                
                except Exception as e:
                    logger.debug(f"Error processing {audio_file.name}: {e}")
        
        logger.info(f"Processed {len(samples)} Khmer samples")
        return samples
    
    def _process_vietnamese_dataset(self) -> List[Dict]:
        """Process Vietnamese (FOSD) dataset."""
        # This will be implemented when you add the FOSD dataset
        dataset_path = Path("data/fosd_vietnamese")
        
        if not dataset_path.exists():
            logger.warning("Vietnamese FOSD dataset not found at data/fosd_vietnamese")
            return []
        
        # Placeholder for FOSD processing
        # You can implement this based on FOSD dataset structure
        logger.info("Vietnamese dataset processing - to be implemented")
        return []
    
    def _save_vits_dataset(self, train_samples: List[Dict], val_samples: List[Dict], languages: List[str]):
        """Save dataset in VITS format."""
        output_dir = Path("data/processed/vits_multilang")
        output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        logger.info(f"VITS dataset saved to {output_dir}")
    
    def create_model_config(self) -> Dict:
        """Create VITS model configuration."""
        config = {
            "model": "vits",
            "run_name": f"multilang_vits_{'-'.join(['km', 'vi'])}",
            "run_description": "Multi-language VITS training (Khmer + Vietnamese)",
            
            # Dataset configuration
            "datasets": [{
                "name": "multilang_dataset",
                "path": "data/processed/vits_multilang/",
                "meta_file_train": "metadata_train.txt",
                "meta_file_val": "metadata_val.txt",
                "language": "multilang"
            }],
            
            # Audio configuration
            "audio": {
                "sample_rate": self.config["sample_rate"],
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
                "do_sound_norm": False,
                "win_length": 1024,
                "hop_length": 256,
                "n_mel_channels": 80,
                "mel_fmax_loss": None
            },
            
            # VITS Model configuration
            "model_args": {
                "num_chars": 500,  # Increased for multi-language
                "num_speakers": 1,
                "text_cleaner": "basic_cleaners",
                "use_phonemes": self.config['use_phonemes'],
                "compute_input_seq_cache": True,
                "precompute_num_workers": 8,
                "use_speaker_embedding": False,
                
                # VITS architecture
                "hidden_channels": self.config["hidden_channels"],
                "filter_channels": self.config["filter_channels"],
                "n_heads": self.config["n_heads"],
                "n_layers": self.config["n_layers"],
                "kernel_size": self.config["kernel_size"],
                "p_dropout": self.config["p_dropout"],
                "resblock": self.config["resblock"],
                "resblock_kernel_sizes": self.config["resblock_kernel_sizes"],
                "resblock_dilation_sizes": self.config["resblock_dilation_sizes"],
                "upsample_rates": self.config["upsample_rates"],
                "upsample_initial_channel": self.config["upsample_initial_channel"],
                "upsample_kernel_sizes": self.config["upsample_kernel_sizes"],
                "use_spectral_norm": self.config["use_spectral_norm"],
                
                # Additional VITS parameters
                "gin_channels": 256,
                "use_sdp": True,
                "noise_scale": 1.0,
                "inference_noise_scale": 0.667,
                "length_scale": 1.0,
                "noise_scale_dp": 1.0,
                "inference_noise_scale_dp": 1.0,
                "max_inference_len": None,
                "init_discriminator": True
            },
            
            # Training configuration
            "batch_size": self.config['batch_size'],
            "eval_batch_size": self.config['batch_size'],
            "num_loader_workers": 8,
            "num_eval_loader_workers": 4,
            "run_eval": True,
            "test_delay_epochs": -1,
            
            # Optimizer
            "epochs": self.config['num_epochs'],
            "lr": self.config['learning_rate'],
            "lr_scheduler": "ExponentialLR",
            "lr_scheduler_params": {"gamma": 0.999875},
            "use_grad_clip": True,
            "grad_clip": 5.0,
            
            # VITS specific training
            "use_weighted_sampler": True,
            "weighted_sampler_attrs": {"language": 1.0, "speaker_name": 1.0},
            "weighted_sampler_multipliers": {},
            "r": 1,
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
            
            # GPU optimizations
            "mixed_precision": self.config["mixed_precision"],
            "use_cuda": True,
            "cudnn_enabled": True,
            "cudnn_benchmark": True,
            
            # Test sentences
            "test_sentences": [
                "សួស្តី! នេះជាការសាកល្បងសម្រាប់ប្រព័ន្ធបំលែងអត្ថបទទៅជាសំឡេង។",  # Khmer
                "Xin chào! Đây là bài kiểm tra cho hệ thống chuyển đổi văn bản thành giọng nói.",  # Vietnamese
                "ប្រទេសកម្ពុជាមានវប្បធម៌ដ៏សម្បូរបែប និងប្រវត្តិសាស្ត្រយ៉ាងយូរលង់។",  # Khmer
                "Việt Nam có nền văn hóa phong phú và lịch sử lâu đời."  # Vietnamese
            ]
        }
        
        return config
    
    def train(self) -> bool:
        """Start VITS training."""
        logger.info("Starting VITS training...")
        
        try:
            from TTS.bin.train_tts import main as train_tts
            import sys
            
            # Create configuration
            config = self.create_model_config()
            config_file = self.output_path / "config.json"
            
            self.output_path.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {config_file}")
            
            # Set training arguments
            original_argv = sys.argv
            sys.argv = [
                "train_tts",
                "--config_path", str(config_file),
                "--restore_path", "",
                "--continue_path", ""
            ]
            
            # Start training
            train_tts()
            
            # Restore original argv
            sys.argv = original_argv
            
            logger.info("VITS training completed successfully!")
            return True
            
        except ImportError:
            logger.error("Coqui TTS not installed. Install with: pip install TTS")
            return False
        except Exception as e:
            logger.error(f"VITS training failed: {e}")
            return False
    
    def test_model(self, test_texts: List[str]) -> bool:
        """Test the trained VITS model."""
        try:
            from TTS.api import TTS
            
            # Find the best model
            model_path = self.output_path / "best_model.pth"
            config_path = self.output_path / "config.json"
            
            if not model_path.exists():
                # Try to find latest checkpoint
                checkpoints = list(self.output_path.glob("checkpoint_*.pth"))
                if checkpoints:
                    model_path = sorted(checkpoints)[-1]
                else:
                    logger.error("No trained model found")
                    return False
            
            # Initialize TTS
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tts = TTS(model_path=str(model_path), config_path=str(config_path)).to(device)
            
            # Generate test audio
            output_dir = Path("results/audio_outputs/vits_test")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, text in enumerate(test_texts, 1):
                output_path = output_dir / f"vits_test_{i}.wav"
                tts.tts_to_file(text=text, file_path=str(output_path))
                logger.info(f"Generated test audio {i}: {output_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Model testing failed: {e}")
            return False
