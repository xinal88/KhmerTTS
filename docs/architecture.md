# Khmer TTS Project Architecture

## Overview

The Khmer Text-to-Speech (TTS) project is designed to evaluate and compare different TTS models for synthesizing natural-sounding Khmer speech. The architecture follows a modular design that supports multiple TTS engines, comprehensive evaluation metrics, and extensible data processing pipelines.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Khmer TTS System                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    Data     │  │   Models    │  │ Evaluation  │         │
│  │ Processing  │  │  Inference  │  │   Engine    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Orpheus TTS │  │ Coqui TTS   │  │ Edge-TTS    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    Audio    │  │    Text     │  │   Config    │         │
│  │   Output    │  │    Input    │  │ Management  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Processing Layer (`scripts/data_preparation.py`)

**Purpose**: Handles preprocessing and preparation of Khmer text data for TTS synthesis.

**Key Features**:
- Khmer text cleaning and normalization
- Unicode range validation (U+1780-U+17FF)
- Sentence segmentation
- Training/validation/test splits
- Text quality validation

**Classes**:
- `KhmerTextProcessor`: Main class for text processing operations

### 2. Model Inference Engine (`scripts/model_inference.py`)

**Purpose**: Provides unified interface for multiple TTS models.

**Architecture Pattern**: Abstract Factory + Strategy Pattern

**Key Components**:
- `TTSModel` (Abstract Base Class): Defines common interface
- `OrpheusTTS`: Llama-based Speech-LLM implementation
- `CoquiTTS`: Open-source deep learning TTS
- `EdgeTTS`: Microsoft Edge online TTS service
- `TTSInferenceEngine`: Orchestrates multiple models

**Design Principles**:
- **Modularity**: Each model is implemented as a separate class
- **Extensibility**: Easy to add new TTS models
- **Configuration-driven**: Model settings managed via YAML
- **Error Handling**: Graceful degradation when models fail

### 3. Evaluation Framework (`scripts/evaluation.py`)

**Purpose**: Comprehensive evaluation and comparison of TTS model outputs.

**Evaluation Metrics**:
- **Audio Quality**: RMS energy, spectral features, MFCC
- **Naturalness**: F0 variation, rhythm regularity, speech rate
- **Intelligibility**: ASR-based metrics (future implementation)

**Classes**:
- `TTSEvaluator`: Main evaluation engine
- Supports both objective and subjective evaluation metrics

### 4. Utility Functions (`scripts/utils.py`)

**Purpose**: Common utilities and helper functions.

**Key Features**:
- Configuration management
- File I/O operations
- Audio processing utilities
- Khmer language validation
- Logging and experiment tracking

## Data Flow Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Raw Text  │───▶│  Processed  │───▶│   Model     │
│   Corpus    │    │    Text     │    │  Inference  │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Evaluation  │◀───│   Audio     │◀───│   Audio     │
│   Reports   │    │  Analysis   │    │   Output    │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Directory Structure

```
khmer-tts/
├── data/                    # Data storage
│   ├── raw/                # Original data
│   ├── processed/          # Cleaned data
│   └── interim/            # Temporary files
├── models/                 # Model storage
│   ├── orpheus/           # Orpheus TTS models
│   ├── coqui/             # Coqui TTS models
│   └── edge-tts/          # Edge-TTS cache
├── scripts/               # Core processing scripts
│   ├── data_preparation.py
│   ├── model_inference.py
│   ├── evaluation.py
│   └── utils.py
├── notebooks/             # Jupyter notebooks
│   ├── exploration.ipynb
│   └── analysis.ipynb
├── config/                # Configuration files
│   ├── models.yaml
│   └── data.yaml
├── results/               # Output and results
│   ├── audio_outputs/
│   └── evaluation_reports/
└── docs/                  # Documentation
    ├── architecture.md
    └── usage.md
```

## Configuration Management

The system uses YAML-based configuration for flexibility and maintainability:

### Model Configuration (`config/models.yaml`)
- Model-specific settings
- API keys and endpoints
- Audio output parameters
- Language-specific configurations

### Data Configuration (`config/data.yaml`)
- Data paths and file locations
- Processing parameters
- Validation rules
- Quality thresholds

## Khmer Language Considerations

### Unicode Support
- **Range**: U+1780-U+17FF (Khmer script)
- **Character Categories**: Consonants, vowels, diacritics, symbols
- **Text Processing**: Handles complex script characteristics

### Linguistic Features
- **Script Type**: Abugida (consonant-vowel writing system)
- **Word Boundaries**: No explicit spaces between words
- **Pronunciation**: Complex consonant clusters and vowel combinations

### Technical Challenges
- Limited training data availability
- Complex script rendering and processing
- Lack of standardized phonetic representations
- Pronunciation variations across dialects

## Model Integration Patterns

### 1. API-based Models (Orpheus TTS)
```python
class OrpheusTTS(TTSModel):
    def synthesize(self, text, output_path):
        # API call to external service
        response = api_client.synthesize(text, voice_settings)
        save_audio(response, output_path)
```

### 2. Local Models (Coqui TTS)
```python
class CoquiTTS(TTSModel):
    def synthesize(self, text, output_path):
        # Local model inference
        tts = TTS(model_path=self.model_path)
        tts.tts_to_file(text=text, file_path=output_path)
```

### 3. Online Services (Edge-TTS)
```python
class EdgeTTS(TTSModel):
    async def synthesize(self, text, output_path):
        # Async online service call
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
```

## Evaluation Architecture

### Objective Metrics
- **Audio Quality**: Technical audio characteristics
- **Naturalness**: Prosodic and rhythmic features
- **Consistency**: Variation across multiple samples

### Comparative Analysis
- Statistical significance testing
- Model ranking algorithms
- Performance visualization

### Extensibility
- Plugin architecture for new metrics
- Configurable evaluation pipelines
- Support for custom evaluation criteria

## Scalability Considerations

### Horizontal Scaling
- Batch processing support
- Parallel model inference
- Distributed evaluation

### Performance Optimization
- Caching mechanisms
- Memory management
- GPU acceleration support

### Monitoring and Logging
- Comprehensive logging framework
- Performance metrics tracking
- Error reporting and debugging

## Security and Privacy

### Data Protection
- Local data processing
- Secure API key management
- Privacy-preserving evaluation

### Model Security
- Input validation and sanitization
- Rate limiting for API calls
- Secure model storage

## Future Enhancements

### Planned Features
1. **Real-time TTS**: Streaming synthesis capabilities
2. **Voice Cloning**: Speaker adaptation for Khmer voices
3. **Emotion Control**: Emotional expression in synthesis
4. **Mobile Deployment**: Lightweight model variants

### Research Directions
1. **Khmer-specific Models**: Custom architectures for Khmer
2. **Pronunciation Modeling**: Advanced phonetic representations
3. **Dialect Support**: Multi-dialect synthesis
4. **Quality Metrics**: Khmer-specific evaluation metrics

## Dependencies and Requirements

### Core Dependencies
- Python 3.8+
- PyTorch/TensorFlow (for Coqui TTS)
- librosa (audio processing)
- edge-tts (Microsoft TTS)

### Optional Dependencies
- Jupyter (for notebooks)
- plotly (visualization)
- scipy (statistical analysis)

### System Requirements
- **Memory**: 4GB+ RAM recommended
- **Storage**: 10GB+ for models and data
- **Network**: Internet connection for API-based models

## Deployment Strategies

### Development Environment
- Local development with Jupyter notebooks
- Configuration-based model switching
- Interactive evaluation and analysis

### Production Environment
- Containerized deployment (Docker)
- API service wrapper
- Monitoring and logging integration

### Cloud Deployment
- Support for major cloud platforms
- Scalable inference endpoints
- Managed model storage
