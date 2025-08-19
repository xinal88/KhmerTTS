"""
Evaluation Module for Khmer TTS Project

This module provides comprehensive evaluation metrics and tools for assessing
the quality of Text-to-Speech synthesis across different models.
"""

import os
import json
import yaml
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class TTSEvaluator:
    """
    Comprehensive evaluator for TTS model performance.
    
    Provides both objective and subjective evaluation metrics
    for comparing different TTS models.
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        """
        Initialize the TTS evaluator.
        
        Args:
            config_path (str): Path to the models configuration file
        """
        self.config = self._load_config(config_path)
        self.results_dir = Path("results/evaluation_reports")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}. Using default settings.")
            return {}
    
    def calculate_audio_metrics(self, audio_path: str) -> Dict[str, float]:
        """
        Calculate objective audio quality metrics.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            Dict[str, float]: Dictionary of audio metrics
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Calculate various metrics
            metrics = {}
            
            # Duration
            metrics['duration'] = len(y) / sr
            
            # RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            metrics['rms_mean'] = float(np.mean(rms))
            metrics['rms_std'] = float(np.std(rms))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            metrics['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            metrics['zcr_mean'] = float(np.mean(zcr))
            metrics['zcr_std'] = float(np.std(zcr))
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            metrics['spectral_rolloff_mean'] = float(np.mean(rolloff))
            metrics['spectral_rolloff_std'] = float(np.std(rolloff))
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                metrics[f'mfcc_{i+1}_mean'] = float(np.mean(mfccs[i]))
                metrics[f'mfcc_{i+1}_std'] = float(np.std(mfccs[i]))
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics for {audio_path}: {e}")
            return {}
    
    def evaluate_naturalness(self, audio_path: str) -> Dict[str, float]:
        """
        Evaluate the naturalness of synthesized speech.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            Dict[str, float]: Naturalness metrics
        """
        metrics = {}
        
        try:
            y, sr = librosa.load(audio_path, sr=None)
            
            # Pitch variation (fundamental frequency)
            f0 = librosa.yin(y, fmin=50, fmax=400)
            f0_clean = f0[f0 > 0]  # Remove unvoiced frames
            
            if len(f0_clean) > 0:
                metrics['f0_mean'] = float(np.mean(f0_clean))
                metrics['f0_std'] = float(np.std(f0_clean))
                metrics['f0_range'] = float(np.max(f0_clean) - np.min(f0_clean))
                metrics['f0_variation_coefficient'] = float(np.std(f0_clean) / np.mean(f0_clean))
            
            # Rhythm and timing
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            if len(onset_frames) > 1:
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                inter_onset_intervals = np.diff(onset_times)
                metrics['rhythm_regularity'] = float(1.0 / (1.0 + np.std(inter_onset_intervals)))
                metrics['speech_rate'] = float(len(onset_frames) / (len(y) / sr))
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating naturalness for {audio_path}: {e}")
            return {}
    
    def evaluate_intelligibility(self, audio_path: str, reference_text: str) -> Dict[str, float]:
        """
        Evaluate speech intelligibility (placeholder for future ASR integration).
        
        Args:
            audio_path (str): Path to the audio file
            reference_text (str): Original text that was synthesized
            
        Returns:
            Dict[str, float]: Intelligibility metrics
        """
        # TODO: Implement ASR-based intelligibility evaluation
        # This would require a Khmer ASR model to transcribe the audio
        # and compare with the reference text
        
        metrics = {
            'reference_text_length': len(reference_text),
            'reference_word_count': len(reference_text.split()),
            # Placeholder for future ASR metrics:
            # 'word_error_rate': 0.0,
            # 'character_error_rate': 0.0,
            # 'bleu_score': 0.0
        }
        
        return metrics
    
    def compare_models(self, audio_dir: str, reference_texts: Optional[List[str]] = None) -> Dict:
        """
        Compare multiple TTS models across various metrics.
        
        Args:
            audio_dir (str): Directory containing audio files from different models
            reference_texts (List[str], optional): Reference texts for each audio file
            
        Returns:
            Dict: Comprehensive comparison results
        """
        audio_dir = Path(audio_dir)
        results = {
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'summary': {}
        }
        
        # Find all audio files
        audio_files = list(audio_dir.glob("**/*.wav"))
        
        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return results
        
        print(f"Evaluating {len(audio_files)} audio files...")
        
        for audio_file in audio_files:
            # Extract model name from filename or directory structure
            model_name = self._extract_model_name(audio_file)
            
            if model_name not in results['models']:
                results['models'][model_name] = {
                    'files': [],
                    'metrics': {
                        'audio_quality': [],
                        'naturalness': [],
                        'intelligibility': []
                    }
                }
            
            print(f"Evaluating {model_name}: {audio_file.name}")
            
            # Calculate metrics
            audio_metrics = self.calculate_audio_metrics(str(audio_file))
            naturalness_metrics = self.evaluate_naturalness(str(audio_file))
            
            # Get reference text if available
            ref_text = ""
            if reference_texts and len(results['models'][model_name]['files']) < len(reference_texts):
                ref_text = reference_texts[len(results['models'][model_name]['files'])]
            
            intelligibility_metrics = self.evaluate_intelligibility(str(audio_file), ref_text)
            
            # Store results
            file_result = {
                'filename': audio_file.name,
                'path': str(audio_file),
                'reference_text': ref_text,
                'audio_metrics': audio_metrics,
                'naturalness_metrics': naturalness_metrics,
                'intelligibility_metrics': intelligibility_metrics
            }
            
            results['models'][model_name]['files'].append(file_result)
            results['models'][model_name]['metrics']['audio_quality'].append(audio_metrics)
            results['models'][model_name]['metrics']['naturalness'].append(naturalness_metrics)
            results['models'][model_name]['metrics']['intelligibility'].append(intelligibility_metrics)
        
        # Calculate summary statistics
        results['summary'] = self._calculate_summary_stats(results['models'])
        
        return results
    
    def _extract_model_name(self, audio_path: Path) -> str:
        """Extract model name from audio file path."""
        # Try to extract from filename first
        filename = audio_path.stem.lower()
        
        if 'orpheus' in filename:
            return 'orpheus'
        elif 'coqui' in filename:
            return 'coqui'
        elif 'edge' in filename:
            return 'edge_tts'
        
        # Try to extract from parent directory
        parent_name = audio_path.parent.name.lower()
        if 'orpheus' in parent_name:
            return 'orpheus'
        elif 'coqui' in parent_name:
            return 'coqui'
        elif 'edge' in parent_name:
            return 'edge_tts'
        
        # Default fallback
        return 'unknown'
    
    def _calculate_summary_stats(self, models_data: Dict) -> Dict:
        """Calculate summary statistics across all models."""
        summary = {}
        
        for model_name, model_data in models_data.items():
            model_summary = {}
            
            # Calculate averages for key metrics
            audio_metrics = model_data['metrics']['audio_quality']
            naturalness_metrics = model_data['metrics']['naturalness']
            
            if audio_metrics:
                # Average duration
                durations = [m.get('duration', 0) for m in audio_metrics if 'duration' in m]
                if durations:
                    model_summary['avg_duration'] = np.mean(durations)
                
                # Average RMS energy
                rms_means = [m.get('rms_mean', 0) for m in audio_metrics if 'rms_mean' in m]
                if rms_means:
                    model_summary['avg_rms_energy'] = np.mean(rms_means)
            
            if naturalness_metrics:
                # Average F0 variation
                f0_vars = [m.get('f0_variation_coefficient', 0) for m in naturalness_metrics 
                          if 'f0_variation_coefficient' in m]
                if f0_vars:
                    model_summary['avg_f0_variation'] = np.mean(f0_vars)
            
            model_summary['total_files'] = len(model_data['files'])
            summary[model_name] = model_summary
        
        return summary
    
    def generate_report(self, evaluation_results: Dict, output_filename: str = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results (Dict): Results from compare_models()
            output_filename (str, optional): Custom output filename
            
        Returns:
            str: Path to the generated report
        """
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"tts_evaluation_report_{timestamp}.json"
        
        report_path = self.results_dir / output_filename
        
        # Save detailed JSON report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
        # Generate human-readable summary
        summary_path = report_path.with_suffix('.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("Khmer TTS Model Evaluation Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {evaluation_results['timestamp']}\n\n")
            
            f.write("Model Summary:\n")
            f.write("-" * 20 + "\n")
            
            for model_name, summary in evaluation_results['summary'].items():
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  Total files: {summary.get('total_files', 0)}\n")
                f.write(f"  Avg duration: {summary.get('avg_duration', 0):.2f}s\n")
                f.write(f"  Avg RMS energy: {summary.get('avg_rms_energy', 0):.4f}\n")
                f.write(f"  Avg F0 variation: {summary.get('avg_f0_variation', 0):.4f}\n")
        
        print(f"Evaluation report saved to: {report_path}")
        print(f"Summary report saved to: {summary_path}")
        
        return str(report_path)


def main():
    """Main function to run TTS evaluation."""
    print("Starting Khmer TTS Model Evaluation...")
    
    evaluator = TTSEvaluator()
    
    # Example evaluation
    audio_dir = "results/audio_outputs"
    
    if not Path(audio_dir).exists():
        print(f"Audio directory not found: {audio_dir}")
        print("Please run model inference first to generate audio files.")
        return
    
    # Run evaluation
    results = evaluator.compare_models(audio_dir)
    
    # Generate report
    report_path = evaluator.generate_report(results)
    
    print(f"\nEvaluation completed! Report saved to: {report_path}")


if __name__ == "__main__":
    main()
