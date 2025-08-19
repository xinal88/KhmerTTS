#!/usr/bin/env python3
"""
Setup script for Multi-Language TTS Training System
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_gpu():
    """Check if CUDA GPU is available."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected!")
            # Extract GPU info from first line
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'CUDA Version' in line:
                    print(f"   {line.strip()}")
                    break
            return True
        else:
            print("⚠️  No NVIDIA GPU detected")
            return False
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found")
        return False

def install_basic_requirements():
    """Install basic requirements."""
    print("📦 Installing basic requirements...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Basic requirements installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install basic requirements: {e}")
        return False

def install_pytorch_gpu():
    """Install PyTorch with CUDA support."""
    print("🔧 Installing PyTorch with CUDA support...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)
        print("✅ PyTorch with CUDA installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False

def install_pytorch_cpu():
    """Install PyTorch CPU version."""
    print("🔧 Installing PyTorch (CPU version)...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchaudio", "--index-url", 
            "https://download.pytorch.org/whl/cpu"
        ], check=True)
        print("✅ PyTorch (CPU) installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch: {e}")
        return False

def install_coqui_tts():
    """Install Coqui TTS."""
    print("🎤 Installing Coqui TTS...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "TTS"
        ], check=True)
        print("✅ Coqui TTS installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install Coqui TTS: {e}")
        return False

def verify_installation():
    """Verify installation."""
    print("🔍 Verifying installation...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        
        import TTS
        print(f"✅ Coqui TTS {TTS.__version__}")
        
        import librosa
        print(f"✅ Librosa {librosa.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "models",
        "results/audio_outputs", 
        "results/logs",
        "data/processed",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def main():
    """Main setup function."""
    print("🚀 Multi-Language TTS Training System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Install basic requirements
    if not install_basic_requirements():
        return 1
    
    # Install PyTorch
    if has_gpu:
        pytorch_success = install_pytorch_gpu()
    else:
        print("Installing CPU version of PyTorch...")
        pytorch_success = install_pytorch_cpu()
    
    if not pytorch_success:
        return 1
    
    # Install Coqui TTS
    if not install_coqui_tts():
        return 1
    
    # Create directories
    create_directories()
    
    # Verify installation
    if not verify_installation():
        return 1
    
    print("\n✅ Setup completed successfully!")
    print("\n🎯 Next steps:")
    print("1. Add your datasets to the data/ directory")
    print("2. Configure training in config.json")
    print("3. Start training: python train.py")
    print("4. List available models: python train.py --list-models")
    print("5. Create new model plugins: python train.py --create-plugin yourmodel")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
