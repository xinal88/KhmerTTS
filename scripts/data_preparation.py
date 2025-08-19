"""
Data Preparation Module for Khmer TTS Project

This module handles the preprocessing and preparation of Khmer text data
for Text-to-Speech synthesis. It includes functions for text cleaning,
normalization, and formatting specific to the Khmer language.
"""

import os
import re
import yaml
from pathlib import Path
from typing import List, Dict, Optional


class KhmerTextProcessor:
    """
    A class for processing Khmer text data for TTS applications.
    
    Handles text cleaning, normalization, and preparation specific
    to the Khmer language characteristics.
    """
    
    def __init__(self, config_path: str = "config/data.yaml"):
        """
        Initialize the Khmer text processor.
        
        Args:
            config_path (str): Path to the data configuration file
        """
        self.config = self._load_config(config_path)
        self.raw_data_path = Path(self.config.get('raw_data_path', 'data/raw'))
        self.processed_data_path = Path(self.config.get('processed_data_path', 'data/processed'))
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}. Using default settings.")
            return {}
    
    def clean_khmer_text(self, text: str) -> str:
        """
        Clean and normalize Khmer text.
        
        Args:
            text (str): Raw Khmer text
            
        Returns:
            str: Cleaned and normalized text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove non-Khmer characters (keeping basic punctuation)
        # Khmer Unicode range: U+1780-U+17FF
        khmer_pattern = r'[^\u1780-\u17FF\s.,!?;:\-\(\)\[\]"]'
        text = re.sub(khmer_pattern, '', text)
        
        # Normalize punctuation spacing
        text = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', text)
        
        return text.strip()
    
    def segment_sentences(self, text: str) -> List[str]:
        """
        Segment Khmer text into sentences.
        
        Args:
            text (str): Input Khmer text
            
        Returns:
            List[str]: List of sentences
        """
        # Split on common sentence endings
        sentences = re.split(r'[.!?។]', text)
        
        # Clean and filter empty sentences
        sentences = [self.clean_khmer_text(s) for s in sentences if s.strip()]
        
        return sentences
    
    def prepare_corpus(self, input_file: str = "khmer_text_corpus.txt") -> None:
        """
        Prepare the Khmer text corpus for TTS training.
        
        Args:
            input_file (str): Name of the input corpus file
        """
        input_path = self.raw_data_path / input_file
        output_path = self.processed_data_path / "processed_corpus.txt"
        
        # Ensure output directory exists
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        if not input_path.exists():
            print(f"Input file not found: {input_path}")
            print("Please add your Khmer text corpus to the raw data directory.")
            return
        
        print(f"Processing corpus from: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as infile:
            raw_text = infile.read()
        
        # Clean and segment the text
        cleaned_text = self.clean_khmer_text(raw_text)
        sentences = self.segment_sentences(cleaned_text)
        
        # Write processed sentences
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for sentence in sentences:
                if len(sentence.strip()) > 10:  # Filter very short sentences
                    outfile.write(sentence + '\n')
        
        print(f"Processed {len(sentences)} sentences")
        print(f"Output saved to: {output_path}")
    
    def create_training_splits(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> None:
        """
        Split processed corpus into training, validation, and test sets.
        
        Args:
            train_ratio (float): Ratio for training set
            val_ratio (float): Ratio for validation set
        """
        corpus_path = self.processed_data_path / "processed_corpus.txt"
        
        if not corpus_path.exists():
            print("Processed corpus not found. Run prepare_corpus() first.")
            return
        
        with open(corpus_path, 'r', encoding='utf-8') as file:
            sentences = [line.strip() for line in file if line.strip()]
        
        total_sentences = len(sentences)
        train_size = int(total_sentences * train_ratio)
        val_size = int(total_sentences * val_ratio)
        
        train_data = sentences[:train_size]
        val_data = sentences[train_size:train_size + val_size]
        test_data = sentences[train_size + val_size:]
        
        # Save splits
        splits = {
            'train.txt': train_data,
            'validation.txt': val_data,
            'test.txt': test_data
        }
        
        for filename, data in splits.items():
            output_path = self.processed_data_path / filename
            with open(output_path, 'w', encoding='utf-8') as file:
                for sentence in data:
                    file.write(sentence + '\n')
            print(f"Created {filename} with {len(data)} sentences")


def main():
    """Main function to run data preparation pipeline."""
    print("Starting Khmer TTS Data Preparation...")
    
    processor = KhmerTextProcessor()
    
    # Prepare the corpus
    processor.prepare_corpus()
    
    # Create training splits
    processor.create_training_splits()
    
    print("Data preparation completed!")


if __name__ == "__main__":
    main()
