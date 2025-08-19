# 🇰🇭 Khmer TTS Training Guide

## 📋 Quick Overview

This guide focuses specifically on training TTS models with the **Khmer dataset** (2,889 high-quality samples). The system is already configured and tested for Khmer-only training.

### ✅ Current Status
- **Dataset**: 2,889 valid Khmer samples ready
- **Configuration**: Optimized for Khmer language
- **Training Pipeline**: Fully tested and working
- **Models Available**: VITS, Orpheus, StyleTTS

---

## 🚀 Quick Start (10 Minutes)

### Step 1: Install Dependencies
```bash
# Install basic dependencies
pip install -r requirements.txt

# Install Coqui TTS for actual training
pip install TTS

# Optional: Install additional audio processing libraries
pip install librosa soundfile
```

### Step 2: Setup Dataset (IMPORTANT!)
```bash
# The repository doesn't include the dataset (to keep it lightweight)
# Follow the dataset setup guide:
```
📋 **READ FIRST**: [DATASET_SETUP.md](DATASET_SETUP.md) - **Required before training!**

**Quick dataset setup:**
1. Create directory: `data/km_kh_male/wavs/`
2. Add your audio files (.wav) to the `wavs/` folder
3. Create `data/km_kh_male/line_index.tsv` with metadata
4. Run verification: `python verify_khmer_setup.py`

### Step 3: Verify Setup
```bash
# Check if everything is working (including dataset)
python verify_khmer_setup.py
```
**Expected Output:**
```
🇰🇭 Khmer dataset...
   ✅ Metadata file: XXXX samples
   ✅ Audio files: XXXX WAV files
```

### Step 4: Start Training
```bash
# Train VITS model (recommended for beginners)
python train.py --model vits
```

---

## 📊 Dataset Information

### Khmer Dataset Stats
- **Total samples**: 2,906 audio files
- **Valid samples**: 2,889 (99.4% success rate)
- **Training samples**: 2,601
- **Validation samples**: 288
- **Audio format**: 22kHz, 16-bit, mono WAV
- **Duration range**: 1-10 seconds per sample
- **Language**: Khmer (km)
- **Speaker**: Male voice

### Dataset Structure
```
data/km_kh_male/
├── line_index.tsv          # Metadata file
└── wavs/                   # Audio files
    ├── 001.wav
    ├── 002.wav
    └── ... (2,906 files)
```

---

## 🎯 Training Options

### Option 1: VITS Training (Recommended)
```bash
# Basic VITS training
python train.py --model vits

# With custom configuration
python train.py --model vits --config custom_config.json
```

**Advantages:**
- ✅ Fast training (few hours on GPU)
- ✅ Good quality output
- ✅ Well-tested and stable
- ✅ Lower GPU memory requirements

### Option 2: Coqui TTS Training
```bash
# Coqui TTS training (alternative approach)
python scripts/train_coqui.py
```

**Advantages:**
- ✅ Industry-standard TTS library
- ✅ Extensive documentation
- ✅ Multiple model architectures
- ✅ Active community support

### Option 3: Advanced Models
```bash
# Orpheus TTS (highest quality)
python train.py --model orpheus

# StyleTTS (style control)
python train.py --model styletts
```

**Note:** These require more GPU memory and training time.

---

## ⚙️ Configuration

### Current Khmer Configuration (`config.json`)
```json
{
  "datasets": {
    "khmer": {
      "path": "data/km_kh_male",
      "format": "tsv",
      "language": "km"
    }
  },
  "training": {
    "model": "vits",
    "languages": ["km"],
    "batch_size": 32,
    "num_epochs": 1000,
    "learning_rate": 0.0002
  }
}
```

### Customization Options
```bash
# Adjust batch size for your GPU
# GTX 1060/1070: batch_size = 8
# RTX 3070/4070: batch_size = 16  
# RTX 3080/4080: batch_size = 32
# RTX 4090: batch_size = 64
```

---

## 💻 Hardware Requirements

### Minimum Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 10GB free space
- **Training Time**: 2-3 days on CPU

### Recommended Requirements  
- **GPU**: NVIDIA GTX 1060+ (6GB VRAM)
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Storage**: 20GB free space
- **Training Time**: 4-8 hours on GPU

### Optimal Requirements
- **GPU**: NVIDIA RTX 3070+ (8GB+ VRAM)
- **CPU**: 12+ cores
- **RAM**: 32GB+
- **Storage**: 50GB free space
- **Training Time**: 2-4 hours on GPU

---

## 📁 Output Structure

After training, you'll find:
```
models/multilang_vits_trained/
├── config.json              # Model configuration
├── model.pth                # Trained model weights
├── dataset_info.json        # Dataset statistics
└── logs/                    # Training logs
    └── vits_training.log

results/
├── audio_outputs/           # Generated test audio
└── evaluation_reports/      # Quality metrics
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. "Coqui TTS not installed"
```bash
pip install TTS
```

#### 2. "CUDA out of memory"
```bash
# Reduce batch size in config.json
"batch_size": 8  # or 4 for very limited GPU
```

#### 3. "No valid samples found"
```bash
# Check dataset path
ls data/km_kh_male/wavs/  # Should show .wav files
cat data/km_kh_male/line_index.tsv  # Should show metadata
```

#### 4. Training is very slow
```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

---

## 🎵 Testing Your Trained Model

### After Training Completes
```bash
# Test the trained model
python test_trained_model.py

# Generate specific text
python -c "
from scripts.model_inference import TTSInferenceEngine
engine = TTSInferenceEngine()
engine.synthesize_text('សួស្តី! នេះជាការសាកល្បង។')
"
```

---

## ❓ Do You Need to Clone Coqui TTS Again?

### **Answer: NO! 🎉**

You **DO NOT** need to clone the entire Coqui TTS repository again. Here's why:

1. **pip install TTS** gives you everything you need
2. **The huge Coqui source code was removed** during cleanup (saved ~2GB)
3. **Our system uses Coqui as a library**, not as source code
4. **All training scripts are already included** in this repository

### What You Need:
```bash
# This is ALL you need:
pip install TTS

# NOT needed:
# git clone https://github.com/coqui-ai/TTS.git  ❌
```

---

## 🚀 Ready to Train!

Your Khmer TTS training system is **100% ready**. Just run:

```bash
# Install TTS library
pip install TTS

# Start training immediately
python train.py --model vits
```

**Expected Training Time:**
- **GPU (RTX 3070+)**: 3-6 hours
- **GPU (GTX 1060)**: 8-12 hours  
- **CPU only**: 2-3 days

The system will automatically:
1. ✅ Process all 2,889 Khmer samples
2. ✅ Create train/validation splits
3. ✅ Generate model configuration
4. ✅ Start training process
5. ✅ Save checkpoints every 50 epochs
6. ✅ Generate test audio samples

**Happy training! 🎉**
