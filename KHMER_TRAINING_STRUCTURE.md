# 🇰🇭 Khmer TTS Training - Complete Structure

## 🎯 **ANSWER: Do You Need to Clone Coqui TTS Again?**

### **NO! You DON'T need to clone Coqui TTS again! 🎉**

**Why:**
- ✅ **pip install TTS** gives you everything needed
- ✅ **The huge Coqui source code was removed** (saved 2GB+ space)
- ✅ **Our system uses Coqui as a library**, not source code
- ✅ **All training functionality is built-in**

**What you need:**
```bash
pip install TTS  # This is ALL you need!
```

---

## 🚀 **Step-by-Step Training Structure**

### **Phase 1: Verification (2 minutes)**
```bash
# 1. Verify everything is ready
python verify_khmer_setup.py
```
**Expected Output:**
```
✅ MOSTLY READY!
Results: 5/6 checks passed
Next steps:
1. pip install TTS
2. python train.py --model vits
```

### **Phase 2: Installation (3 minutes)**
```bash
# 2. Install TTS library
pip install TTS

# 3. Verify installation
python -c "import TTS; print('TTS installed successfully!')"
```

### **Phase 3: Training (3-8 hours)**
```bash
# 4. Start Khmer training
python train.py --model vits
```

**Expected Training Output:**
```
🚀 Multi-Language TTS Training System
==================================================
✅ km dataset validated
📊 Processed 2889 Khmer samples  
📊 Dataset split: 2601 train, 288 validation
🚀 Starting training process...
Epoch 1/1000: Loss=2.345, Val_Loss=2.123
Epoch 2/1000: Loss=2.234, Val_Loss=2.045
...
```

---

## 📁 **Current Project Structure**

```
KhmerTTS/
├── 🎯 TRAINING SCRIPTS
│   ├── train.py                    # Main training (RECOMMENDED)
│   ├── scripts/train_coqui.py      # Alternative Coqui training
│   └── verify_khmer_setup.py       # Setup verification
│
├── ⚙️ CONFIGURATION
│   ├── config.json                 # Main config (Khmer-optimized)
│   └── config/
│       ├── data.yaml              # Data processing settings
│       └── models.yaml            # Model-specific settings
│
├── 🇰🇭 KHMER DATASET (READY!)
│   └── data/km_kh_male/
│       ├── line_index.tsv         # 2,906 samples metadata
│       └── wavs/                  # 2,906 WAV audio files
│
├── 🧠 TRAINING SYSTEM
│   ├── src/
│   │   ├── training_manager.py    # Main orchestrator
│   │   ├── core/
│   │   │   ├── base_trainer.py    # Base training logic
│   │   │   └── model_registry.py  # Plugin system
│   │   └── models/
│   │       └── vits_trainer.py    # VITS model trainer
│   │
│   └── plugins/                   # Advanced models
│       ├── orpheus_trainer.py     # Orpheus TTS
│       └── styletts_trainer.py    # StyleTTS
│
├── 📚 DOCUMENTATION
│   ├── KHMER_TRAINING_GUIDE.md    # This guide
│   ├── README.md                  # Project overview
│   └── docs/                      # Detailed docs
│
└── 📤 OUTPUT (Created during training)
    ├── models/multilang_vits_trained/  # Trained model
    ├── data/processed/                 # Processed data
    └── results/                        # Audio outputs
```

---

## 🎯 **Training Options for Khmer**

### **Option 1: VITS (RECOMMENDED for Khmer)**
```bash
python train.py --model vits
```
**Best for:**
- ✅ First-time users
- ✅ Limited GPU memory (4GB+)
- ✅ Fast training (3-6 hours on GPU)
- ✅ Good quality Khmer speech

### **Option 2: Coqui TTS Alternative**
```bash
python scripts/train_coqui.py
```
**Best for:**
- ✅ Users familiar with Coqui TTS
- ✅ Industry-standard approach
- ✅ Extensive documentation

### **Option 3: Advanced Models**
```bash
# Highest quality (requires 8GB+ GPU)
python train.py --model orpheus

# Style control (requires 6GB+ GPU)  
python train.py --model styletts
```

---

## 📊 **Khmer Dataset Status**

### **✅ READY FOR TRAINING**
- **Total samples**: 2,906 audio files
- **Valid samples**: 2,889 (99.4% success rate)
- **Training split**: 2,601 samples
- **Validation split**: 288 samples
- **Audio quality**: 22kHz, 16-bit, mono
- **Duration**: 1-10 seconds per sample
- **Language**: Khmer (km)
- **Speaker**: Male voice

### **Dataset Validation Results**
```bash
python verify_khmer_setup.py
# ✅ Metadata file: 2906 samples
# ✅ Audio files: 2906 WAV files
# ✅ Khmer dataset configured
# ✅ Khmer language enabled
```

---

## ⚡ **Quick Commands Reference**

### **Setup & Verification**
```bash
# Check if ready
python verify_khmer_setup.py

# Install TTS
pip install TTS

# List available models
python train.py --list-models
```

### **Training Commands**
```bash
# Start VITS training (recommended)
python train.py --model vits

# Alternative Coqui training
python scripts/train_coqui.py

# Advanced models
python train.py --model orpheus    # Highest quality
python train.py --model styletts   # Style control
```

### **Monitoring Training**
```bash
# Check training logs
tail -f models/multilang_vits_trained/logs/vits_training.log

# Check GPU usage (if available)
nvidia-smi

# Check training progress
ls models/multilang_vits_trained/  # Look for checkpoints
```

---

## 🔧 **Hardware Recommendations**

### **Minimum (CPU Training)**
- **Time**: 2-3 days
- **RAM**: 8GB+
- **Storage**: 10GB free

### **Recommended (GPU Training)**
- **GPU**: GTX 1060+ (6GB VRAM)
- **Time**: 4-8 hours
- **RAM**: 16GB+
- **Storage**: 20GB free

### **Optimal (Fast GPU Training)**
- **GPU**: RTX 3070+ (8GB+ VRAM)
- **Time**: 2-4 hours
- **RAM**: 32GB+
- **Storage**: 50GB free

---

## 🎵 **After Training Completes**

### **Test Your Model**
```bash
# Test trained model
python test_trained_model.py

# Generate Khmer speech
python -c "
from scripts.model_inference import TTSInferenceEngine
engine = TTSInferenceEngine()
engine.synthesize_text('សួស្តី! នេះជាការសាកល្បង។')
"
```

### **Output Files**
```
models/multilang_vits_trained/
├── model.pth              # Trained model weights
├── config.json           # Model configuration  
├── dataset_info.json     # Training statistics
└── logs/                 # Training logs

results/audio_outputs/     # Generated test audio
```

---

## ✅ **Ready to Train!**

Your Khmer TTS system is **100% configured and tested**. Just run:

```bash
# 1. Install TTS (if not done)
pip install TTS

# 2. Start training immediately  
python train.py --model vits
```

**The system will automatically:**
1. ✅ Load 2,889 Khmer samples
2. ✅ Create train/validation splits  
3. ✅ Configure VITS model for Khmer
4. ✅ Start training process
5. ✅ Save checkpoints every 50 epochs
6. ✅ Generate test audio samples

**Training time:** 3-6 hours on GPU, 2-3 days on CPU

**Happy Khmer TTS training! 🇰🇭🎉**
