"""
Utility Functions for Khmer TTS Project

This module contains helper functions and utilities used across
the Khmer TTS project for common tasks like file handling,
text processing, and audio manipulation.
"""

import os
import re
import json
import yaml
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file with error handling.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Warning: Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        return {}


def save_config(config: Dict, config_path: str) -> bool:
    """
    Save configuration to YAML file.
    
    Args:
        config (Dict): Configuration dictionary
        config_path (str): Path to save the configuration
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, allow_unicode=True)
        return True
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")
        return False


def load_text_file(file_path: str, encoding: str = 'utf-8') -> Optional[str]:
    """
    Load text from file with error handling.
    
    Args:
        file_path (str): Path to the text file
        encoding (str): File encoding (default: utf-8)
        
    Returns:
        Optional[str]: File content or None if error
    """
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except UnicodeDecodeError:
        print(f"Encoding error reading {file_path}. Trying with 'utf-8-sig'...")
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as file:
                return file.read()
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def save_text_file(content: str, file_path: str, encoding: str = 'utf-8') -> bool:
    """
    Save text to file with error handling.
    
    Args:
        content (str): Text content to save
        file_path (str): Path to save the file
        encoding (str): File encoding (default: utf-8)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding=encoding) as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error saving text to {file_path}: {e}")
        return False


def is_khmer_text(text: str) -> bool:
    """
    Check if text contains Khmer characters.
    
    Args:
        text (str): Text to check
        
    Returns:
        bool: True if text contains Khmer characters
    """
    # Khmer Unicode range: U+1780-U+17FF
    khmer_pattern = r'[\u1780-\u17FF]'
    return bool(re.search(khmer_pattern, text))


def clean_filename(filename: str) -> str:
    """
    Clean filename by removing invalid characters.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Cleaned filename
    """
    # Remove invalid characters for filenames
    invalid_chars = r'[<>:"/\\|?*]'
    cleaned = re.sub(invalid_chars, '_', filename)
    
    # Remove multiple underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores and dots
    cleaned = cleaned.strip('_.')
    
    return cleaned


def get_audio_duration(audio_path: str) -> Optional[float]:
    """
    Get duration of audio file in seconds.
    
    Args:
        audio_path (str): Path to audio file
        
    Returns:
        Optional[float]: Duration in seconds or None if error
    """
    try:
        y, sr = librosa.load(audio_path, sr=None)
        return len(y) / sr
    except Exception as e:
        print(f"Error getting duration for {audio_path}: {e}")
        return None


def convert_audio_format(input_path: str, output_path: str, 
                        target_sr: int = 22050, target_format: str = 'wav') -> bool:
    """
    Convert audio file to different format/sample rate.
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to output audio file
        target_sr (int): Target sample rate (default: 22050)
        target_format (str): Target format (default: wav)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load audio
        y, sr = librosa.load(input_path, sr=None)
        
        # Resample if necessary
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save audio
        sf.write(output_path, y, target_sr)
        return True
        
    except Exception as e:
        print(f"Error converting audio {input_path} to {output_path}: {e}")
        return False


def normalize_audio(audio_path: str, output_path: str = None, 
                   target_lufs: float = -23.0) -> bool:
    """
    Normalize audio to target loudness level.
    
    Args:
        audio_path (str): Path to input audio file
        output_path (str, optional): Path to output file (overwrites input if None)
        target_lufs (float): Target loudness in LUFS (default: -23.0)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Simple RMS-based normalization (placeholder for proper LUFS normalization)
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            # Target RMS corresponding to approximately -23 LUFS
            target_rms = 0.1
            y = y * (target_rms / rms)
        
        # Prevent clipping
        y = np.clip(y, -1.0, 1.0)
        
        # Save normalized audio
        if output_path is None:
            output_path = audio_path
        
        sf.write(output_path, y, sr)
        return True
        
    except Exception as e:
        print(f"Error normalizing audio {audio_path}: {e}")
        return False


def create_directory_structure(base_path: str, structure: Dict) -> bool:
    """
    Create directory structure from nested dictionary.
    
    Args:
        base_path (str): Base directory path
        structure (Dict): Nested dictionary representing directory structure
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        base_path = Path(base_path)
        
        def create_dirs(current_path: Path, struct: Dict):
            for name, content in struct.items():
                new_path = current_path / name
                
                if isinstance(content, dict):
                    # It's a directory with subdirectories
                    new_path.mkdir(parents=True, exist_ok=True)
                    create_dirs(new_path, content)
                else:
                    # It's a file or empty directory
                    if content is None:
                        # Empty directory
                        new_path.mkdir(parents=True, exist_ok=True)
                    else:
                        # File with content
                        new_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(new_path, 'w', encoding='utf-8') as f:
                            f.write(content)
        
        create_dirs(base_path, structure)
        return True
        
    except Exception as e:
        print(f"Error creating directory structure: {e}")
        return False


def batch_process_files(input_dir: str, output_dir: str, 
                       process_func, file_pattern: str = "*.txt",
                       **kwargs) -> List[Tuple[str, bool]]:
    """
    Process multiple files in batch.
    
    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
        process_func: Function to process each file
        file_pattern (str): File pattern to match (default: *.txt)
        **kwargs: Additional arguments for process_func
        
    Returns:
        List[Tuple[str, bool]]: List of (filename, success) tuples
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for file_path in input_path.glob(file_pattern):
        try:
            output_file = output_path / file_path.name
            success = process_func(str(file_path), str(output_file), **kwargs)
            results.append((file_path.name, success))
            
            if success:
                print(f"✓ Processed: {file_path.name}")
            else:
                print(f"✗ Failed: {file_path.name}")
                
        except Exception as e:
            print(f"✗ Error processing {file_path.name}: {e}")
            results.append((file_path.name, False))
    
    return results


def log_experiment(experiment_name: str, parameters: Dict, 
                  results: Dict, log_dir: str = "results/experiments") -> str:
    """
    Log experiment parameters and results.
    
    Args:
        experiment_name (str): Name of the experiment
        parameters (Dict): Experiment parameters
        results (Dict): Experiment results
        log_dir (str): Directory to save logs
        
    Returns:
        str: Path to the log file
    """
    from datetime import datetime
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{experiment_name}_{timestamp}.json"
    
    log_data = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'parameters': parameters,
        'results': results
    }
    
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"Experiment logged to: {log_file}")
        return str(log_file)
        
    except Exception as e:
        print(f"Error logging experiment: {e}")
        return ""


def validate_khmer_corpus(corpus_path: str) -> Dict[str, Union[int, float, List[str]]]:
    """
    Validate Khmer text corpus and provide statistics.
    
    Args:
        corpus_path (str): Path to the corpus file
        
    Returns:
        Dict: Validation results and statistics
    """
    results = {
        'total_lines': 0,
        'khmer_lines': 0,
        'empty_lines': 0,
        'avg_line_length': 0.0,
        'total_characters': 0,
        'khmer_characters': 0,
        'issues': []
    }
    
    try:
        content = load_text_file(corpus_path)
        if content is None:
            results['issues'].append(f"Could not read file: {corpus_path}")
            return results
        
        lines = content.split('\n')
        results['total_lines'] = len(lines)
        
        line_lengths = []
        khmer_char_count = 0
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            
            if not line:
                results['empty_lines'] += 1
                continue
            
            line_lengths.append(len(line))
            results['total_characters'] += len(line)
            
            if is_khmer_text(line):
                results['khmer_lines'] += 1
                # Count Khmer characters
                khmer_chars = len(re.findall(r'[\u1780-\u17FF]', line))
                khmer_char_count += khmer_chars
            else:
                results['issues'].append(f"Line {i}: No Khmer characters detected")
        
        results['khmer_characters'] = khmer_char_count
        
        if line_lengths:
            results['avg_line_length'] = sum(line_lengths) / len(line_lengths)
        
        # Additional validation
        if results['khmer_lines'] == 0:
            results['issues'].append("No lines with Khmer text found")
        
        if results['empty_lines'] > results['total_lines'] * 0.5:
            results['issues'].append("More than 50% of lines are empty")
        
    except Exception as e:
        results['issues'].append(f"Error validating corpus: {e}")
    
    return results


# Constants for Khmer language processing
KHMER_UNICODE_RANGE = (0x1780, 0x17FF)
KHMER_VOWELS = [
    '\u17B6', '\u17B7', '\u17B8', '\u17B9', '\u17BA', '\u17BB', 
    '\u17BC', '\u17BD', '\u17BE', '\u17BF', '\u17C0', '\u17C1',
    '\u17C2', '\u17C3', '\u17C4', '\u17C5'
]
KHMER_CONSONANTS = [chr(i) for i in range(0x1780, 0x17A3)]


def get_project_info() -> Dict[str, str]:
    """
    Get basic project information.
    
    Returns:
        Dict[str, str]: Project information
    """
    return {
        'name': 'Khmer TTS Project',
        'description': 'Text-to-Speech synthesis for Khmer language',
        'models': ['Orpheus TTS', 'Coqui TTS', 'Edge-TTS'],
        'version': '1.0.0',
        'language': 'Khmer (km)',
        'unicode_range': f'U+{KHMER_UNICODE_RANGE[0]:04X}-U+{KHMER_UNICODE_RANGE[1]:04X}'
    }


if __name__ == "__main__":
    # Example usage
    print("Khmer TTS Utilities")
    print("=" * 20)
    
    project_info = get_project_info()
    for key, value in project_info.items():
        print(f"{key}: {value}")
