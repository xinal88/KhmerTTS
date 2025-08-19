# Getting Started with Multi-Language TTS Training

This guide will help you get started with the extensible TTS training system in just a few minutes.

## 🚀 Quick Setup (3 Steps)

### Step 1: Setup Environment
```bash
# Run the automated setup
python setup.py
```

This will:
- ✅ Check Python version (3.8+ required)
- ✅ Detect your GPU (RTX 5880 Ada)
- ✅ Install PyTorch with CUDA support
- ✅ Install Coqui TTS
- ✅ Create necessary directories
- ✅ Verify everything works

### Step 2: Start Training
```bash
# Train VITS model with your Khmer dataset
python train.py
```

### Step 3: Test Your Model
```bash
# After training completes, test the model
python -c "
import sys
sys.path.append('src')
from training_manager import TTSTrainingManager
manager = TTSTrainingManager()
# Your trained model will be tested automatically
"
```

## 📊 What You Get

### Current Setup
- ✅ **VITS Model**: Ready to train with your 2,907 Khmer samples
- ✅ **GPU Optimized**: Automatic optimization for RTX 5880 Ada
- ✅ **Multi-Language**: Support for Khmer + Vietnamese (FOSD)
- ✅ **Plugin System**: Easy to add new models

### Future Models (Templates Ready)
- 🔧 **Orpheus**: High-quality TTS with emotions
- 🔧 **StyleTTS**: Style transfer and prosody control
- ➕ **Your Models**: Easy plugin creation

## 🔌 Adding New Models (Super Easy!)

Want to add FishAudio, Bark, or any other TTS model?

### 1. Create Plugin Template
```bash
python train.py --create-plugin fishaudio
```

### 2. Edit the Generated File
The system creates `plugins/fishaudio_trainer.py` with everything you need:

```python
class FishAudioTrainer(BaseTTSTrainer):
    def prepare_dataset(self, languages):
        # Your dataset preparation logic
        pass
    
    def train(self):
        # Your training logic
        pass
    
    def test_model(self, test_texts):
        # Your testing logic
        pass
```

### 3. Use Immediately
```bash
python train.py --model fishaudio
```

That's it! The system handles everything else automatically.

## 📁 Your Clean Project Structure

```
KhmerTTS/
├── 🎯 train.py              # Main entry point
├── ⚙️ config.json           # Configuration
├── 🔧 setup.py              # Automated setup
├── 📚 README.md             # Documentation
├── 
├── src/                     # Core system (don't modify)
│   ├── core/               # Base classes
│   ├── models/             # Model implementations
│   └── training_manager.py # Main orchestrator
├── 
├── plugins/                 # Your model plugins
│   ├── orpheus_trainer.py  # Orpheus template
│   ├── styletts_trainer.py # StyleTTS template
│   └── [your_model].py     # Your custom models
├── 
├── data/                   # Your datasets
│   ├── km_kh_male/        # Khmer (ready)
│   └── fosd_vietnamese/   # Vietnamese (add yours)
├── 
├── models/                # Trained models output
├── results/               # Training results & audio
└── logs/                  # Training logs
```

## 🎯 Usage Examples

### Basic Training
```bash
# Train with default settings
python train.py

# Train specific model
python train.py --model vits

# Use custom config
python train.py --config my_config.json
```

### Model Management
```bash
# List available models
python train.py --list-models

# Create new model plugin
python train.py --create-plugin yourmodel

# Load plugins from directory
python train.py --load-plugins plugins
```

### Configuration
Edit `config.json` to customize:
- Dataset paths
- Training parameters
- GPU settings
- Model-specific options

## 🔧 Troubleshooting

### Common Issues

**GPU Not Detected**
```bash
# Check GPU
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Import Errors**
```bash
# Reinstall dependencies
python setup.py
```

**Training Fails**
```bash
# Check logs
tail -f logs/training.log

# Reduce batch size in config.json
"batch_size": 16  # or 8
```

## 🎵 Expected Results

### Training Timeline (RTX 5880 Ada)
- **Setup**: 5-10 minutes
- **Training**: 2-3 days (1000 epochs)
- **Testing**: 1-2 minutes

### Quality Progression
- **100 epochs**: Basic speech
- **500 epochs**: Clear pronunciation
- **1000 epochs**: Natural-sounding speech

## 🚀 Next Steps

1. **Start Training**: Run `python train.py` now!
2. **Monitor Progress**: Use TensorBoard or check logs
3. **Add Vietnamese**: Put FOSD dataset in `data/fosd_vietnamese/`
4. **Create Plugins**: Add your favorite TTS models
5. **Share Results**: Contribute back to the community

## 💡 Pro Tips

- **Let it train**: VITS needs many epochs for quality
- **Monitor GPU**: Use `nvidia-smi` to check utilization
- **Save checkpoints**: Training saves every 50 epochs
- **Test early**: Generate samples to check progress
- **Experiment**: Try different models and configurations

## 🆘 Need Help?

1. Check the logs in `logs/` directory
2. Verify GPU memory usage with `nvidia-smi`
3. Test with smaller batch sizes
4. Ensure dataset integrity

---

**Ready to start?** Run `python setup.py` and then `python train.py`! 🚀
