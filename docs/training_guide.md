# Khmer TTS Training Guide

This guide explains how to train and fine-tune TTS models using your Khmer dataset in `data/km_kh_male/`.

## 📊 **Your Dataset Overview**

Your dataset contains:
- **2,907 audio samples** (WAV format)
- **Khmer transcriptions** in `line_index.tsv`
- **Male Khmer voice** recordings
- **High-quality audio** suitable for TTS training

## 🎯 **Training Options**

### ❌ **Edge-TTS: Cannot be trained**
Edge-TTS is Microsoft's online service and cannot be trained or fine-tuned with custom data.

### ✅ **Option 1: Orpheus TTS (Recommended)**
- **Best quality** for custom voices
- **LLM-based** architecture
- **Emotion control** capabilities
- **Voice cloning** support

### ✅ **Option 2: Coqui TTS (Alternative)**
- **Open-source** and well-documented
- **Proven training pipeline**
- **Good for research** and experimentation
- **Multiple model architectures**

## 🚀 **Quick Start Training**

### Orpheus TTS Training

```bash
# 1. Prepare and start Orpheus training
python scripts/train_orpheus.py

# 2. If automatic training fails, use manual script
bash models/orpheus_khmer_trained/train_khmer.sh
```

### Coqui TTS Training

```bash
# 1. Install Coqui TTS
pip install TTS

# 2. Prepare and start Coqui training
python scripts/train_coqui.py

# 3. If automatic training fails, use manual script
bash models/coqui_khmer_trained/train_coqui_khmer.sh
```

## 📋 **Detailed Training Steps**

### Prerequisites

1. **Hardware Requirements**:
   - **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
   - **RAM**: 16GB+ system RAM
   - **Storage**: 20GB+ free space
   - **Time**: 6-24 hours depending on hardware

2. **Software Requirements**:
   ```bash
   # Core dependencies
   pip install torch>=2.0.0 torchaudio>=2.0.0
   pip install transformers accelerate datasets
   
   # For Coqui TTS
   pip install TTS
   
   # For audio processing
   pip install librosa soundfile
   ```

### Step-by-Step Process

#### 1. **Dataset Preparation**

Your dataset is automatically processed by the training scripts:

- ✅ **Audio validation**: Checks duration (0.5-10 seconds)
- ✅ **Text cleaning**: Removes invalid characters
- ✅ **Train/validation split**: 90%/10% split
- ✅ **Format conversion**: Converts to model-specific format

#### 2. **Training Configuration**

Both scripts create optimized configurations:

**Orpheus Configuration**:
```json
{
  "model_name": "canopylabs/orpheus-tts-0.1-pretrained",
  "language": "km",
  "speaker_name": "khmer_male_01",
  "batch_size": 4,
  "learning_rate": 1e-5,
  "num_epochs": 100
}
```

**Coqui Configuration**:
```json
{
  "model": "vits",
  "language": "km",
  "batch_size": 8,
  "learning_rate": 1e-4,
  "num_epochs": 1000
}
```

#### 3. **Training Process**

**Orpheus Training**:
1. Downloads base model from HuggingFace
2. Fine-tunes on your Khmer dataset
3. Saves checkpoints every 10 epochs
4. Evaluates every 5 epochs

**Coqui Training**:
1. Initializes VITS model architecture
2. Trains from scratch on your dataset
3. Saves checkpoints every 100 epochs
4. Generates test samples during training

#### 4. **Monitoring Training**

**Check Training Progress**:
```bash
# View training logs
tail -f models/orpheus_khmer_trained/train.log
# or
tail -f models/coqui_khmer_trained/train.log

# Monitor GPU usage
nvidia-smi -l 1
```

**Training Metrics to Watch**:
- **Loss**: Should decrease over time
- **Validation loss**: Should not increase (overfitting)
- **Audio quality**: Listen to generated samples
- **Training time**: ~1-2 minutes per epoch

## 🔧 **Advanced Configuration**

### Orpheus TTS Customization

Edit `models/orpheus_khmer_trained/training_config.json`:

```json
{
  "training": {
    "batch_size": 2,        // Reduce if GPU memory issues
    "learning_rate": 5e-6,  // Lower for more stable training
    "num_epochs": 200,      // Increase for better quality
    "gradient_accumulation": 4,  // Simulate larger batch size
    "warmup_steps": 1000,   // Learning rate warmup
    "weight_decay": 0.01    // Regularization
  }
}
```

### Coqui TTS Customization

Edit `models/coqui_khmer_trained/config.json`:

```json
{
  "batch_size": 16,           // Increase if you have more GPU memory
  "lr": 2e-4,                // Higher learning rate for faster training
  "epochs": 2000,            // More epochs for better quality
  "save_step": 50,           // Save more frequently
  "mixed_precision": true,    // Use FP16 for faster training
  "grad_clip": 5.0           // Gradient clipping for stability
}
```

## 📈 **Training Tips**

### Performance Optimization

1. **GPU Memory**:
   ```bash
   # Reduce batch size if out of memory
   # Orpheus: batch_size: 2 or 1
   # Coqui: batch_size: 4 or 2
   ```

2. **Training Speed**:
   ```bash
   # Use mixed precision training
   # Enable gradient accumulation
   # Use multiple GPUs if available
   ```

3. **Quality Improvement**:
   ```bash
   # Train for more epochs
   # Use lower learning rate
   # Add data augmentation
   # Clean dataset further
   ```

### Common Issues and Solutions

#### 1. **Out of Memory Error**
```bash
# Solution: Reduce batch size
# Orpheus: batch_size: 1
# Coqui: batch_size: 2
```

#### 2. **Training Too Slow**
```bash
# Solution: Enable optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
```

#### 3. **Poor Audio Quality**
```bash
# Solution: 
# - Train for more epochs
# - Check dataset quality
# - Adjust learning rate
# - Use data augmentation
```

#### 4. **Model Not Learning**
```bash
# Solution:
# - Increase learning rate
# - Check data preprocessing
# - Verify dataset format
# - Add learning rate warmup
```

## 🧪 **Testing Trained Models**

### Test Orpheus Model

```python
from scripts.model_inference import TTSInferenceEngine

# Update config to use trained model
config = {
    'model_path': 'models/orpheus_khmer_trained',
    'model_name': 'models/orpheus_khmer_trained/best_model'
}

engine = TTSInferenceEngine()
text = "សួស្តី! នេះជាការសាកល្បងសម្រាប់ប្រព័ន្ធបំលែងអត្ថបទទៅជាសំឡេង។"
results = engine.synthesize_text(text, ['orpheus'])
```

### Test Coqui Model

```bash
# Direct synthesis with trained model
python -m TTS.bin.synthesize \
    --model_path "models/coqui_khmer_trained/best_model.pth" \
    --config_path "models/coqui_khmer_trained/config.json" \
    --text "សួស្តី! នេះជាការសាកល្បងសម្រាប់ប្រព័ន្ធបំលែងអត្ថបទទៅជាសំឡេង។" \
    --out_path "test_khmer_output.wav"
```

## 📊 **Evaluation and Comparison**

After training, evaluate your models:

```bash
# Run evaluation on trained models
python main.py --evaluation-only

# Compare with pre-trained models
python scripts/evaluation.py --compare-all
```

## 🎯 **Expected Results**

### Training Timeline

- **Orpheus TTS**: 6-12 hours for 100 epochs
- **Coqui TTS**: 12-24 hours for 1000 epochs

### Quality Expectations

- **Initial quality**: Audible but robotic (first 10-20 epochs)
- **Good quality**: Natural-sounding (50-100 epochs)
- **High quality**: Near-human quality (100+ epochs)

### Success Metrics

- **Intelligibility**: Text should be clearly understandable
- **Naturalness**: Speech should sound human-like
- **Consistency**: Similar quality across different texts
- **Khmer pronunciation**: Correct pronunciation of Khmer words

## 🔄 **Next Steps After Training**

1. **Model Integration**: Update `config/models.yaml` to use trained models
2. **Quality Assessment**: Run comprehensive evaluation
3. **Fine-tuning**: Adjust parameters based on results
4. **Deployment**: Set up inference pipeline for production use
5. **Data Collection**: Gather more data for improved training

## 📚 **Additional Resources**

- **Orpheus Documentation**: https://github.com/canopyai/Orpheus-TTS
- **Coqui TTS Documentation**: https://docs.coqui.ai/
- **TTS Training Best Practices**: https://tts.readthedocs.io/
- **Khmer Language Resources**: Unicode standards and linguistic guides

## 🆘 **Getting Help**

If you encounter issues:

1. **Check logs**: Look in training output directories
2. **Reduce complexity**: Start with smaller batch sizes
3. **Verify data**: Ensure dataset is properly formatted
4. **Community support**: Ask in TTS community forums
5. **Hardware check**: Verify GPU and memory availability
