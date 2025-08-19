#!/usr/bin/env python3
"""
Khmer TTS Setup Verification Script
Verifies that everything is ready for Khmer dataset training
"""

import sys
import json
from pathlib import Path
import pandas as pd

def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (Good)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('pandas', 'Pandas'),
        ('librosa', 'Librosa'),
        ('pathlib', 'Pathlib (built-in)'),
    ]
    
    optional_packages = [
        ('TTS', 'Coqui TTS (for training)'),
    ]
    
    all_good = True
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {description}")
        except ImportError:
            print(f"   ❌ {description} - Install with: pip install {package}")
            all_good = False
    
    print("\n   Optional packages:")
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"   ✅ {description}")
        except ImportError:
            print(f"   ⚠️  {description} - Install with: pip install {package}")
    
    return all_good

def check_dataset():
    """Check Khmer dataset."""
    print("\n🇰🇭 Checking Khmer dataset...")

    dataset_path = Path("data/km_kh_male")
    metadata_file = dataset_path / "line_index.tsv"
    audio_dir = dataset_path / "wavs"

    if not dataset_path.exists():
        print(f"   ⚠️  Dataset directory not found: {dataset_path}")
        print("   📋 Please follow DATASET_SETUP.md to add your dataset")
        print("   💡 The repository doesn't include the dataset to keep it lightweight")
        return False

    if not metadata_file.exists():
        print(f"   ❌ Metadata file not found: {metadata_file}")
        print("   📋 Please create line_index.tsv following DATASET_SETUP.md")
        return False

    if not audio_dir.exists():
        print(f"   ❌ Audio directory not found: {audio_dir}")
        print("   📋 Please create wavs/ directory and add your audio files")
        return False
    
    # Check metadata
    try:
        df = pd.read_csv(metadata_file, sep='\t', header=None)
        total_samples = len(df)
        print(f"   ✅ Metadata file: {total_samples} samples")
    except Exception as e:
        print(f"   ❌ Error reading metadata: {e}")
        return False
    
    # Check audio files
    audio_files = list(audio_dir.glob("*.wav"))
    print(f"   ✅ Audio files: {len(audio_files)} WAV files")
    
    if len(audio_files) < 1000:
        print(f"   ⚠️  Warning: Only {len(audio_files)} audio files found")
    
    return True

def check_configuration():
    """Check configuration file."""
    print("\n⚙️ Checking configuration...")
    
    config_file = Path("config.json")
    if not config_file.exists():
        print(f"   ❌ Configuration file not found: {config_file}")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Check Khmer dataset configuration
        if 'datasets' in config and 'khmer' in config['datasets']:
            khmer_config = config['datasets']['khmer']
            print(f"   ✅ Khmer dataset configured: {khmer_config['path']}")
        else:
            print("   ❌ Khmer dataset not configured in config.json")
            return False
        
        # Check training configuration
        if 'training' in config:
            training_config = config['training']
            languages = training_config.get('languages', [])
            if 'km' in languages:
                print(f"   ✅ Khmer language enabled: {languages}")
            else:
                print(f"   ❌ Khmer language not in training languages: {languages}")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading configuration: {e}")
        return False

def check_training_system():
    """Check if training system is working."""
    print("\n🚀 Checking training system...")
    
    # Check main training script
    train_script = Path("train.py")
    if not train_script.exists():
        print(f"   ❌ Training script not found: {train_script}")
        return False
    
    print(f"   ✅ Main training script: {train_script}")
    
    # Check source code structure
    src_dir = Path("src")
    if not src_dir.exists():
        print(f"   ❌ Source directory not found: {src_dir}")
        return False
    
    required_modules = [
        "src/training_manager.py",
        "src/core/base_trainer.py",
        "src/core/model_registry.py",
        "src/models/vits_trainer.py"
    ]
    
    for module in required_modules:
        if Path(module).exists():
            print(f"   ✅ {module}")
        else:
            print(f"   ❌ {module}")
            return False
    
    return True

def check_gpu():
    """Check GPU availability."""
    print("\n🖥️ Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ GPU available: {gpu_name} ({gpu_count} device(s))")
            
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   ✅ GPU memory: {gpu_memory:.1f} GB")
            
            if gpu_memory >= 6:
                print("   ✅ GPU memory sufficient for training")
            else:
                print("   ⚠️  GPU memory may be limited (recommend 6GB+)")
            
            return True
        else:
            print("   ⚠️  No GPU detected - training will be slow on CPU")
            return False
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

def main():
    """Main verification function."""
    print("🇰🇭 Khmer TTS Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Khmer Dataset", check_dataset),
        ("Configuration", check_configuration),
        ("Training System", check_training_system),
        ("GPU Support", check_gpu),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ Error during {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} {name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} checks passed")
    
    if passed == len(results):
        print("\n🎉 ALL CHECKS PASSED!")
        print("Your Khmer TTS system is ready for training!")
        print("\nNext steps:")
        print("1. pip install TTS  (if not already installed)")
        print("2. python train.py --model vits")
    elif passed >= len(results) - 1:
        print("\n✅ MOSTLY READY!")
        print("Your system is ready for training. Address any warnings if needed.")
        print("\nNext steps:")
        print("1. pip install TTS  (if not already installed)")
        print("2. python train.py --model vits")
    else:
        print("\n⚠️ SETUP INCOMPLETE")
        print("Please address the failed checks before training.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
