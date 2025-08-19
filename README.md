# Multi-Language TTS Training System

An extensible, plugin-based Text-to-Speech (TTS) training system supporting multiple languages and models. Currently optimized for Khmer and Vietnamese with easy integration for additional languages and TTS models.

## 🎯 Project Overview

The Khmer TTS project addresses the challenge of generating high-quality speech synthesis for the Khmer language, which is considered a low-resource language in the context of modern AI and speech technologies. This project provides a comprehensive framework for:

- **Data Processing**: Cleaning and preparing Khmer text corpora
- **Model Integration**: Supporting multiple TTS engines (Orpheus, Coqui, Edge-TTS)
- **Evaluation**: Comprehensive metrics for audio quality and naturalness
- **Analysis**: Tools for comparing model performance

## 📁 Project Structure

```
.
├── README.md
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── data/
│   ├── raw/               # Original data
│   │   ├── khmer_text_corpus.txt
│   │   └── audio_samples/
│   ├── processed/         # Cleaned data
│   └── interim/           # Temporary files
├── models/
│   ├── orpheus/           # Orpheus TTS models
│   ├── coqui/             # Coqui TTS models
│   └── edge-tts/          # Edge-TTS cache
├── scripts/
│   ├── data_preparation.py
│   ├── model_inference.py
│   ├── evaluation.py
│   └── utils.py
├── notebooks/
│   ├── exploration.ipynb
│   └── analysis.ipynb
├── config/
│   ├── models.yaml
│   └── data.yaml
├── results/
│   ├── audio_outputs/
│   └── evaluation_reports/
└── docs/
    ├── architecture.md
    └── usage.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for some TTS models)
- **Khmer dataset** (see setup guide below)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/xinal88/KhmerTTS
   cd KhmerTTS
   ```

2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install TTS  # For actual training
   ```

4. **Setup Dataset (REQUIRED)**:

   📋 **IMPORTANT**: This repository doesn't include the dataset to keep it lightweight.

   **Follow the dataset setup guide**: [DATASET_SETUP.md](DATASET_SETUP.md)

   **Quick setup:**
   - Create `data/km_kh_male/wavs/` directory
   - Add your Khmer audio files (.wav)
   - Create `data/km_kh_male/line_index.tsv` metadata file
   - Run `python verify_khmer_setup.py` to verify

5. **Verify setup**:
   ```bash
   python verify_khmer_setup.py
   ```

### Basic Usage

1. **Run the complete pipeline**:
   ```bash
   python main.py
   ```

2. **Run individual components**:
   ```bash
   # Data preparation only
   python main.py --data-only
   
   # Model inference only
   python main.py --inference-only
   
   # Evaluation only
   python main.py --evaluation-only
   ```

3. **Use specific models**:
   ```bash
   python main.py --models orpheus coqui
   ```

## 🔧 Configuration

### Model Configuration

Edit `config/models.yaml` to configure TTS models:

```yaml
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
    voice: "en-US-AriaNeural"
```

### Data Configuration

Edit `config/data.yaml` to configure data processing:

```yaml
paths:
  raw_data_path: "data/raw"
  processed_data_path: "data/processed"
  output_path: "results/audio_outputs"

processing:
  text_processing:
    min_sentence_length: 10
    max_sentence_length: 500
```

## 📊 Models Used

This project utilizes and evaluates the following Text-to-Speech models:

### 🎭 Orpheus TTS
- **Type**: Llama-based Speech-LLM
- **Features**: High-quality, empathetic text-to-speech generation
- **Access**: API-based (requires API key)
- **Website**: [Fal.ai Orpheus TTS](https://fal.ai/models/fal-ai/orpheus-tts)

### 🔊 Coqui TTS
- **Type**: Open-source deep learning toolkit
- **Features**: Voice cloning, multilingual support, customizable
- **Access**: Local installation
- **Repository**: [Coqui-ai/TTS](https://github.com/coqui-ai/TTS)

### 🌐 Edge-TTS
- **Type**: Microsoft Edge's online text-to-speech service
- **Features**: Neural voices, natural prosody, multiple languages
- **Access**: Online service (free)
- **Repository**: [rany2/edge-tts](https://github.com/rany2/edge-tts)

## 🇰🇭 Khmer Language Considerations

Khmer is a low-resource language with unique characteristics that pose challenges for TTS systems:

- **Script**: Abugida writing system (consonant-vowel combinations)
- **Unicode Range**: U+1780-U+17FF
- **Challenges**: 
  - Complex script handling
  - Lack of explicit word boundaries
  - Limited training data availability
  - Pronunciation variations

This project addresses these challenges through:
- Specialized text preprocessing for Khmer script
- Unicode validation and normalization
- Custom evaluation metrics
- Comprehensive data quality assessment

## 📈 Usage Examples

### Data Exploration

```python
from scripts.data_preparation import KhmerTextProcessor
from scripts.utils import validate_khmer_corpus

# Process Khmer text
processor = KhmerTextProcessor()
processor.prepare_corpus()

# Validate corpus quality
results = validate_khmer_corpus('data/raw/khmer_text_corpus.txt')
print(f"Khmer content: {results['khmer_percentage']:.1f}%")
```

### Model Inference

```python
from scripts.model_inference import TTSInferenceEngine

# Initialize TTS engine
engine = TTSInferenceEngine()

# Synthesize Khmer text
text = "សួស្តី! នេះជាការសាកល្បង។"
results = engine.synthesize_text(text)

# Check results
for model, success in results.items():
    print(f"{model}: {'✓' if success else '✗'}")
```

### Evaluation

```python
from scripts.evaluation import TTSEvaluator

# Evaluate model outputs
evaluator = TTSEvaluator()
results = evaluator.compare_models('results/audio_outputs')

# Generate report
report_path = evaluator.generate_report(results)
print(f"Report saved: {report_path}")
```

## 📓 Jupyter Notebooks

The project includes interactive Jupyter notebooks for exploration and analysis:

### Data Exploration (`notebooks/exploration.ipynb`)
- Text corpus analysis
- Character frequency analysis
- Data quality assessment
- Visualization of text statistics

### Model Analysis (`notebooks/analysis.ipynb`)
- Model performance comparison
- Audio quality metrics
- Statistical analysis
- Interactive dashboards

To use the notebooks:
```bash
jupyter notebook notebooks/exploration.ipynb
```

## 🔍 Evaluation Metrics

The project provides comprehensive evaluation metrics:

### Audio Quality Metrics
- **RMS Energy**: Audio signal strength
- **Spectral Centroid**: Frequency distribution
- **Zero Crossing Rate**: Signal characteristics
- **MFCC**: Mel-frequency cepstral coefficients

### Naturalness Metrics
- **F0 Variation**: Pitch variation patterns
- **Rhythm Regularity**: Speech timing consistency
- **Speech Rate**: Speaking speed analysis

### Intelligibility Metrics
- **Duration Analysis**: Audio length consistency
- **Quality Scoring**: Composite quality metrics
- **Comparative Analysis**: Model ranking

## 🛠️ Development

### Project Structure

The project follows a modular architecture:

- **Data Layer**: Text processing and validation
- **Model Layer**: TTS engine abstractions
- **Evaluation Layer**: Metrics and analysis
- **Configuration Layer**: YAML-based settings

### Adding New Models

To add a new TTS model:

1. Create a new class inheriting from `TTSModel`
2. Implement the `synthesize` method
3. Add configuration to `models.yaml`
4. Register in `TTSInferenceEngine`

Example:
```python
class NewTTS(TTSModel):
    def synthesize(self, text, output_path):
        # Implement synthesis logic
        pass
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 Documentation

- **[Architecture Guide](docs/architecture.md)**: Detailed system architecture
- **[Usage Guide](docs/usage.md)**: Comprehensive usage instructions
- **[API Reference](docs/api_reference.md)**: Code documentation (coming soon)

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Check API keys for cloud models
   - Verify model file paths
   - Ensure internet connectivity

2. **Audio Output Issues**:
   - Check disk space and permissions
   - Verify audio codec support
   - Review model-specific logs

3. **Khmer Text Processing**:
   - Ensure UTF-8 encoding
   - Validate Unicode range
   - Check text content quality

### Getting Help

- Check the [Usage Guide](docs/usage.md) for detailed instructions
- Review log files in `results/logs/`
- Open an issue on GitHub for bugs or feature requests

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Orpheus TTS**: Fal.ai for providing advanced TTS capabilities
- **Coqui TTS**: The Coqui community for open-source TTS tools
- **Edge-TTS**: Microsoft for accessible neural TTS
- **Khmer Language Community**: For supporting low-resource language research

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please:

- Open an issue on GitHub
- Contact the development team
- Join our community discussions

## 🔮 Future Enhancements

- **Real-time TTS**: Streaming synthesis capabilities
- **Voice Cloning**: Custom voice training for Khmer
- **Mobile Support**: Lightweight model deployment
- **Web Interface**: Browser-based TTS interface
- **API Service**: RESTful API for integration

---

**Note**: This project is designed to advance Khmer language technology and support the preservation and accessibility of the Khmer language in the digital age.
