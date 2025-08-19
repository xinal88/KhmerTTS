# Khmer TTS Usage Guide

## Quick Start

### 1. Installation and Setup

```bash
# Clone the repository
git clone <repository-url>
cd khmer-tts

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Run the complete pipeline
python main.py

# Or run individual components
python scripts/data_preparation.py
python scripts/model_inference.py
python scripts/evaluation.py
```

## Detailed Usage Instructions

### Data Preparation

#### Preparing Khmer Text Corpus

1. **Add your Khmer text data**:
   ```bash
   # Place your Khmer text file in the raw data directory
   cp your_khmer_text.txt data/raw/khmer_text_corpus.txt
   ```

2. **Run data preparation**:
   ```python
   from scripts.data_preparation import KhmerTextProcessor
   
   processor = KhmerTextProcessor()
   processor.prepare_corpus()
   processor.create_training_splits()
   ```

3. **Validate your data**:
   ```python
   from scripts.utils import validate_khmer_corpus
   
   results = validate_khmer_corpus('data/raw/khmer_text_corpus.txt')
   print(results)
   ```

#### Data Format Requirements

- **Encoding**: UTF-8
- **Content**: Primarily Khmer text (Unicode range U+1780-U+17FF)
- **Format**: Plain text, one sentence per line (optional)
- **Size**: Minimum 1000 sentences recommended

#### Example Khmer Text

```
សួស្តី! នេះជាការសាកល្បងសម្រាប់ប្រព័ន្ធបំលែងអត្ថបទទៅជាសំឡេង។
ប្រទេសកម្ពុជាមានវប្បធម៌ដ៏សម្បូរបែប និងប្រវត្តិសាស្ត្រយ៉ាងយូរលង់។
បច្ចេកវិទ្យាថ្មីៗកំពុងផ្លាស់ប្តូរជីវិតរបស់យើងគ្រប់ៗថ្ងៃ។
```

### Model Configuration

#### Setting Up Model Configurations

1. **Edit model configuration**:
   ```yaml
   # config/models.yaml
   models:
     orpheus:
       enabled: true
       api_key: "your-api-key-here"
       voice_id: "default"
     
     coqui:
       enabled: true
       model_path: "models/coqui/model.pth"
     
     edge_tts:
       enabled: true
       voice: "en-US-AriaNeural"  # Update when Khmer voices available
   ```

2. **Configure data paths**:
   ```yaml
   # config/data.yaml
   paths:
     raw_data_path: "data/raw"
     processed_data_path: "data/processed"
     output_path: "results/audio_outputs"
   ```

### Model Inference

#### Running TTS Synthesis

1. **Basic synthesis**:
   ```python
   from scripts.model_inference import TTSInferenceEngine
   
   engine = TTSInferenceEngine()
   
   # Synthesize single text
   text = "សួស្តី! នេះជាការសាកល្បង។"
   results = engine.synthesize_text(text)
   print(results)
   ```

2. **Batch synthesis**:
   ```python
   texts = [
       "សួស្តី! នេះជាការសាកល្បង។",
       "ប្រទេសកម្ពុជាមានវប្បធម៌ដ៏សម្បូរបែប។",
       "បច្ចេកវិទ្យាថ្មីៗកំពុងផ្លាស់ប្តូរជីវិត។"
   ]
   
   results = engine.batch_synthesize(texts)
   ```

3. **Model-specific synthesis**:
   ```python
   # Use only specific models
   results = engine.synthesize_text(text, model_names=['coqui', 'edge_tts'])
   ```

#### Command Line Usage

```bash
# Run inference with default settings
python scripts/model_inference.py

# Run with custom configuration
python scripts/model_inference.py --config config/custom_models.yaml

# Run specific models only
python scripts/model_inference.py --models orpheus coqui
```

### Evaluation and Analysis

#### Running Evaluation

1. **Automatic evaluation**:
   ```python
   from scripts.evaluation import TTSEvaluator
   
   evaluator = TTSEvaluator()
   results = evaluator.compare_models('results/audio_outputs')
   report_path = evaluator.generate_report(results)
   ```

2. **Custom evaluation**:
   ```python
   # Evaluate specific audio file
   metrics = evaluator.calculate_audio_metrics('path/to/audio.wav')
   naturalness = evaluator.evaluate_naturalness('path/to/audio.wav')
   ```

#### Using Jupyter Notebooks

1. **Data exploration**:
   ```bash
   jupyter notebook notebooks/exploration.ipynb
   ```

2. **Model analysis**:
   ```bash
   jupyter notebook notebooks/analysis.ipynb
   ```

### Configuration Options

#### Model-Specific Settings

**Orpheus TTS**:
```yaml
orpheus:
  api_key: "your-api-key"
  voice_settings:
    speed: 1.0
    pitch: 1.0
    emotion: "neutral"
  parameters:
    temperature: 0.7
    top_p: 0.9
```

**Coqui TTS**:
```yaml
coqui:
  model_path: "models/coqui/model.pth"
  config_path: "models/coqui/config.json"
  parameters:
    length_scale: 1.0
    noise_scale: 0.667
```

**Edge-TTS**:
```yaml
edge_tts:
  voice: "en-US-AriaNeural"
  rate: "+0%"
  volume: "+0%"
  pitch: "+0Hz"
```

#### Data Processing Settings

```yaml
processing:
  text_processing:
    remove_extra_whitespace: true
    normalize_punctuation: true
    min_sentence_length: 10
    max_sentence_length: 500
  
  audio_processing:
    target_sample_rate: 22050
    normalize_audio: true
    target_lufs: -23.0
```

## Advanced Usage

### Custom Model Integration

1. **Create custom model class**:
   ```python
   from scripts.model_inference import TTSModel
   
   class CustomTTS(TTSModel):
       def synthesize(self, text, output_path):
           # Implement your custom TTS logic
           pass
   ```

2. **Register with inference engine**:
   ```python
   engine = TTSInferenceEngine()
   engine.models['custom'] = CustomTTS(config)
   ```

### Custom Evaluation Metrics

```python
from scripts.evaluation import TTSEvaluator

class CustomEvaluator(TTSEvaluator):
    def custom_metric(self, audio_path):
        # Implement custom evaluation metric
        pass
```

### Batch Processing

#### Processing Multiple Files

```python
from scripts.utils import batch_process_files

def process_text_file(input_path, output_path):
    # Custom processing function
    pass

results = batch_process_files(
    input_dir='data/raw',
    output_dir='data/processed',
    process_func=process_text_file,
    file_pattern='*.txt'
)
```

#### Parallel Processing

```python
# Enable parallel processing in configuration
global_settings:
  parallel_processing: true
  max_workers: 4
```

### API Integration

#### Creating REST API Wrapper

```python
from flask import Flask, request, jsonify
from scripts.model_inference import TTSInferenceEngine

app = Flask(__name__)
engine = TTSInferenceEngine()

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text')
    model = data.get('model', 'edge_tts')
    
    results = engine.synthesize_text(text, [model])
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors

**Problem**: Model fails to load
```
Error: Model file not found
```

**Solution**:
- Check model paths in configuration
- Ensure models are downloaded and placed correctly
- Verify file permissions

#### 2. Audio Output Issues

**Problem**: No audio files generated
```
Error: Audio synthesis failed
```

**Solutions**:
- Check API keys for cloud-based models
- Verify internet connection for online services
- Check disk space and write permissions
- Review model-specific error logs

#### 3. Khmer Text Processing

**Problem**: Text not recognized as Khmer
```
Warning: No Khmer characters detected
```

**Solutions**:
- Verify text encoding (should be UTF-8)
- Check Unicode range (U+1780-U+17FF)
- Validate text content manually

#### 4. Evaluation Errors

**Problem**: Evaluation fails with audio files
```
Error: Could not analyze audio file
```

**Solutions**:
- Install required audio libraries (librosa, soundfile)
- Check audio file format compatibility
- Verify file paths and accessibility

### Performance Optimization

#### Memory Management

```python
# Configure memory limits
performance:
  memory:
    chunk_size: 1000
    max_memory_usage: "2GB"
```

#### GPU Acceleration

```python
# Enable GPU if available
global_settings:
  gpu_acceleration: true
```

### Debugging

#### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Configuration Validation

```python
from scripts.utils import load_config

config = load_config('config/models.yaml')
print("Configuration loaded successfully:", bool(config))
```

## Best Practices

### Data Quality

1. **Text Preparation**:
   - Use high-quality, diverse Khmer text
   - Remove or fix encoding errors
   - Ensure proper sentence segmentation

2. **Audio Quality**:
   - Use consistent audio formats
   - Normalize audio levels
   - Remove background noise

### Model Selection

1. **For Development**: Use Edge-TTS for quick testing
2. **For Quality**: Use Orpheus TTS for best results
3. **For Customization**: Use Coqui TTS for fine-tuning

### Evaluation Strategy

1. **Objective Metrics**: Use automated evaluation for initial screening
2. **Subjective Evaluation**: Include native speakers for final assessment
3. **Comparative Analysis**: Always compare multiple models

### Configuration Management

1. **Version Control**: Track configuration changes
2. **Environment-specific**: Use different configs for dev/prod
3. **Documentation**: Document all configuration options

## Examples and Tutorials

### Example 1: Basic TTS Pipeline

```python
# Complete pipeline example
from scripts.data_preparation import KhmerTextProcessor
from scripts.model_inference import TTSInferenceEngine
from scripts.evaluation import TTSEvaluator

# 1. Prepare data
processor = KhmerTextProcessor()
processor.prepare_corpus()

# 2. Run inference
engine = TTSInferenceEngine()
text = "សួស្តី! នេះជាការសាកល្បង។"
results = engine.synthesize_text(text)

# 3. Evaluate results
evaluator = TTSEvaluator()
evaluation = evaluator.compare_models('results/audio_outputs')
report = evaluator.generate_report(evaluation)

print(f"Pipeline completed. Report saved to: {report}")
```

### Example 2: Custom Evaluation

```python
# Custom evaluation workflow
import librosa
import numpy as np

def custom_quality_score(audio_path):
    y, sr = librosa.load(audio_path)
    
    # Calculate custom metrics
    rms = np.mean(librosa.feature.rms(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # Combine into score
    score = (rms * 0.5) + (spectral_centroid / 5000 * 0.5)
    return min(1.0, score)

# Apply to all audio files
audio_files = ['file1.wav', 'file2.wav', 'file3.wav']
scores = [custom_quality_score(f) for f in audio_files]
print(f"Average quality score: {np.mean(scores):.3f}")
```

## Support and Resources

### Documentation
- [Architecture Guide](architecture.md)
- [API Reference](api_reference.md)
- [Configuration Reference](config_reference.md)

### Community
- GitHub Issues: Report bugs and feature requests
- Discussions: Ask questions and share experiences
- Wiki: Community-contributed guides and examples

### Contributing
- Follow the contribution guidelines
- Submit pull requests for improvements
- Report issues with detailed information
