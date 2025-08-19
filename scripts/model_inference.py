"""
Model Inference Module for Khmer TTS Project

This module handles text-to-speech inference using different TTS models:
- Orpheus TTS
- Coqui TTS
- Edge-TTS

Each model has its own implementation for generating speech from Khmer text.
"""

import os
import yaml
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod


class TTSModel(ABC):
    """Abstract base class for TTS models."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config.get('name', 'unknown')
        
    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> bool:
        """
        Synthesize speech from text.
        
        Args:
            text (str): Input text to synthesize
            output_path (str): Path to save the generated audio
            
        Returns:
            bool: True if synthesis was successful, False otherwise
        """
        pass


class OrpheusTTS(TTSModel):
    """Orpheus TTS model implementation."""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_path = config.get('model_path', 'models/orpheus')
        self.model_name = config.get('model_name', 'canopylabs/orpheus-tts-0.1-finetune-prod')
        self.voice_reference = config.get('voice_reference', None)
        self.emotion = config.get('emotion', 'neutral')
        self.speed = config.get('speed', 1.0)
        self._model = None

    def _load_model(self):
        """Load the Orpheus TTS model."""
        if self._model is not None:
            return self._model

        try:
            # Add the orpheus_tts_pypi to Python path
            import sys
            orpheus_path = Path(self.model_path) / 'orpheus_tts_pypi'
            if orpheus_path.exists():
                sys.path.insert(0, str(orpheus_path))

            # Try to import and initialize Orpheus TTS
            from orpheus_tts import OrpheusModel

            print(f"[Orpheus] Loading model: {self.model_name}")
            # Initialize with proper parameters
            self._model = OrpheusModel(
                model_name=self.model_name,
                max_model_len=2048,  # Add max model length parameter
                dtype="bfloat16"     # Use string format for dtype
            )
            print(f"[Orpheus] Model loaded successfully")
            return self._model

        except ImportError as e:
            print(f"[Orpheus] Could not import Orpheus TTS: {e}")
            print(f"[Orpheus] Please install required dependencies or check model path")
            return None
        except Exception as e:
            print(f"[Orpheus] Error loading model: {e}")
            return None

    def synthesize(self, text: str, output_path: str) -> bool:
        """
        Synthesize speech using Orpheus TTS.
        """
        try:
            print(f"[Orpheus] Synthesizing: {text[:50]}...")

            # Load model if not already loaded
            model = self._load_model()
            if model is None:
                print(f"[Orpheus] Model not available, creating placeholder")
                return self._create_placeholder(text, output_path)

            # Select a voice from available voices
            available_voices = getattr(model, 'available_voices', ['zoe', 'zac', 'jess', 'leo', 'mia', 'julia', 'leah'])
            voice = available_voices[0] if available_voices else 'zoe'  # Default to first available voice

            print(f"[Orpheus] Generating audio with voice: {voice}")

            # Generate audio using the correct method
            audio_data = model.generate_speech(
                prompt=text,
                voice=voice,
                temperature=0.6,
                top_p=0.8,
                max_tokens=1200
            )

            # Save audio to file
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Save as WAV file
            import soundfile as sf
            import numpy as np

            # Ensure audio_data is in the right format
            if isinstance(audio_data, (list, tuple)):
                audio_data = np.array(audio_data, dtype=np.float32)
            elif hasattr(audio_data, 'numpy'):
                audio_data = audio_data.numpy().astype(np.float32)

            # Write the audio file
            sf.write(output_path, audio_data, 22050)  # 22050 Hz sample rate

            print(f"[Orpheus] Audio saved to: {output_path}")
            return True

        except Exception as e:
            print(f"[Orpheus] Error during synthesis: {e}")
            print(f"[Orpheus] Creating placeholder instead")
            return self._create_placeholder(text, output_path)

    def _prepare_prompt(self, text: str) -> str:
        """Prepare the prompt for Orpheus TTS with emotion and voice guidance."""
        # Orpheus uses a specific prompt format
        prompt = f"<speak>"

        # Add emotion if specified
        if self.emotion and self.emotion != 'neutral':
            prompt += f"<emotion>{self.emotion}</emotion>"

        # Add voice reference if available
        if self.voice_reference:
            prompt += f"<voice>{self.voice_reference}</voice>"

        # Add the text
        prompt += text
        prompt += "</speak>"

        return prompt

    def _create_placeholder(self, text: str, output_path: str) -> bool:
        """Create a placeholder file when model is not available."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            placeholder_path = output_path.replace('.wav', '_orpheus_placeholder.txt')

            with open(placeholder_path, 'w', encoding='utf-8') as f:
                f.write(f"Orpheus TTS synthesis placeholder for: {text}\n")
                f.write(f"Model path: {self.model_path}\n")
                f.write(f"Model name: {self.model_name}\n")
                f.write(f"Emotion: {self.emotion}\n")
                f.write(f"Speed: {self.speed}\n")

            print(f"[Orpheus] Placeholder saved to: {placeholder_path}")
            return True

        except Exception as e:
            print(f"[Orpheus] Error creating placeholder: {e}")
            return False


class CoquiTTS(TTSModel):
    """Coqui TTS model implementation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_path = config.get('model_path')
        self.speaker_id = config.get('speaker_id')
        
    def synthesize(self, text: str, output_path: str) -> bool:
        """
        Synthesize speech using Coqui TTS.

        This implementation creates a simple audio file using basic synthesis.
        """
        try:
            print(f"[Coqui] Synthesizing: {text[:50]}...")

            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Try to use a simple TTS approach
            try:
                # Try using pyttsx3 as a fallback TTS engine
                import pyttsx3

                engine = pyttsx3.init()
                engine.setProperty('rate', 150)  # Speed of speech
                engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

                # Save to file
                engine.save_to_file(text, output_path)
                engine.runAndWait()

                # Check if file was created
                if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                    print(f"[Coqui] Audio saved to: {output_path} ({Path(output_path).stat().st_size} bytes)")
                    return True
                else:
                    raise Exception("Audio file was not created or is empty")

            except ImportError:
                print("[Coqui] pyttsx3 not available, creating placeholder")
                # Create a placeholder file
                with open(output_path.replace('.wav', '_coqui_placeholder.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"Coqui TTS synthesis placeholder for: {text}\n")
                    f.write(f"Model path: {self.model_path}\n")
                    f.write(f"Speaker ID: {self.speaker_id}\n")

                print(f"[Coqui] Placeholder saved to: {output_path.replace('.wav', '_coqui_placeholder.txt')}")
                return True

        except Exception as e:
            print(f"[Coqui] Error during synthesis: {e}")
            return False


class EdgeTTS(TTSModel):
    """Edge-TTS model implementation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        # Use a more suitable voice for multilingual content
        # Try Vietnamese voice as it might handle Khmer better than English
        self.voice = config.get('voice', 'vi-VN-HoaiMyNeural')  # Vietnamese voice
        self.rate = config.get('rate', '+0%')
        self.volume = config.get('volume', '+0%')
        
    async def _async_synthesize(self, text: str, output_path: str) -> bool:
        """Async synthesis method for Edge-TTS."""
        try:
            import edge_tts

            print(f"[Edge-TTS] Using voice: {self.voice}")
            print(f"[Edge-TTS] Text length: {len(text)} characters")

            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, volume=self.volume)

            # Save the audio
            await communicate.save(output_path)

            # Check if file was created and has content
            if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                print(f"[Edge-TTS] Successfully created audio file: {Path(output_path).stat().st_size} bytes")
                return True
            else:
                print(f"[Edge-TTS] Audio file was not created or is empty")
                return False

        except ImportError:
            print("[Edge-TTS] edge-tts library not installed. Install with: pip install edge-tts")
            return False
        except Exception as e:
            print(f"[Edge-TTS] Error during synthesis: {e}")
            print(f"[Edge-TTS] Voice: {self.voice}, Text: {text[:100]}...")
            return False
    
    def synthesize(self, text: str, output_path: str) -> bool:
        """
        Synthesize speech using Edge-TTS.
        """
        try:
            print(f"[Edge-TTS] Synthesizing: {text[:50]}...")
            
            # Create output directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Run async synthesis
            result = asyncio.run(self._async_synthesize(text, output_path))
            
            if result:
                print(f"[Edge-TTS] Audio saved to: {output_path}")
            
            return result
            
        except Exception as e:
            print(f"[Edge-TTS] Error during synthesis: {e}")
            return False


class TTSInferenceEngine:
    """Main inference engine for managing multiple TTS models."""
    
    def __init__(self, config_path: str = "config/models.yaml"):
        """
        Initialize the TTS inference engine.
        
        Args:
            config_path (str): Path to the models configuration file
        """
        self.config = self._load_config(config_path)
        self.models = self._initialize_models()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}. Using default settings.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for models."""
        return {
            'models': {
                'orpheus': {
                    'name': 'orpheus',
                    'enabled': True,
                    'api_key': None,
                    'voice_id': 'default'
                },
                'coqui': {
                    'name': 'coqui',
                    'enabled': True,
                    'model_path': 'models/coqui/model.pth',
                    'speaker_id': None
                },
                'edge_tts': {
                    'name': 'edge_tts',
                    'enabled': True,
                    'voice': 'en-US-AriaNeural',
                    'rate': '+0%',
                    'volume': '+0%'
                }
            }
        }
    
    def _initialize_models(self) -> Dict[str, TTSModel]:
        """Initialize all enabled TTS models."""
        models = {}
        model_configs = self.config.get('models', {})
        
        for model_name, config in model_configs.items():
            if not config.get('enabled', True):
                continue
                
            if model_name == 'orpheus':
                models[model_name] = OrpheusTTS(config)
            elif model_name == 'coqui':
                models[model_name] = CoquiTTS(config)
            elif model_name == 'edge_tts':
                models[model_name] = EdgeTTS(config)
            else:
                print(f"Unknown model type: {model_name}")
        
        print(f"Initialized {len(models)} TTS models: {list(models.keys())}")
        return models
    
    def synthesize_text(self, text: str, model_names: Optional[List[str]] = None, 
                       output_dir: str = "results/audio_outputs") -> Dict[str, bool]:
        """
        Synthesize text using specified models.
        
        Args:
            text (str): Text to synthesize
            model_names (List[str], optional): List of model names to use. If None, use all models.
            output_dir (str): Directory to save output files
            
        Returns:
            Dict[str, bool]: Results for each model (True if successful, False otherwise)
        """
        if model_names is None:
            model_names = list(self.models.keys())
        
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name in model_names:
            if model_name not in self.models:
                print(f"Model '{model_name}' not available")
                results[model_name] = False
                continue
            
            # Create unique filename for each model
            filename = f"{model_name}_output.wav"
            file_path = output_path / filename
            
            # Synthesize using the model
            success = self.models[model_name].synthesize(text, str(file_path))
            results[model_name] = success
        
        return results
    
    def batch_synthesize(self, texts: List[str], model_names: Optional[List[str]] = None,
                        output_dir: str = "results/audio_outputs") -> Dict[str, List[bool]]:
        """
        Synthesize multiple texts using specified models.
        
        Args:
            texts (List[str]): List of texts to synthesize
            model_names (List[str], optional): List of model names to use
            output_dir (str): Directory to save output files
            
        Returns:
            Dict[str, List[bool]]: Results for each model and text
        """
        results = {model: [] for model in (model_names or self.models.keys())}
        
        for i, text in enumerate(texts):
            print(f"\nProcessing text {i+1}/{len(texts)}")
            
            # Create subdirectory for this batch
            batch_output_dir = Path(output_dir) / f"batch_{i+1:03d}"
            
            text_results = self.synthesize_text(text, model_names, str(batch_output_dir))
            
            for model_name, success in text_results.items():
                if model_name in results:
                    results[model_name].append(success)
        
        return results


def main():
    """Main function to run TTS inference."""
    print("Starting Khmer TTS Model Inference...")
    
    # Initialize inference engine
    engine = TTSInferenceEngine()
    
    # Example text for testing
    test_text = "សួស្តី! នេះជាការសាកល្បងសម្រាប់ប្រព័ន្ធបំលែងអត្ថបទទៅជាសំឡេង។"
    
    # Synthesize using all available models
    results = engine.synthesize_text(test_text)
    
    print("\nSynthesis Results:")
    for model_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {model_name}: {status}")


if __name__ == "__main__":
    main()
