# 📊 Dataset Setup Guide

## 🎯 Overview

This repository does **NOT** include the actual dataset to keep the repository lightweight and fast to clone. You need to add your own Khmer dataset following the structure below.

## 📁 Required Dataset Structure

### Place your dataset in this exact location:
```
data/km_kh_male/
├── line_index.tsv          # Metadata file (required)
└── wavs/                   # Audio files directory (required)
    ├── audio_001.wav
    ├── audio_002.wav
    └── ... (your audio files)
```

### Metadata File Format (`line_index.tsv`)

The `line_index.tsv` file should be a tab-separated file with this format:
```
filename	empty_column	transcription
audio_001		សួស្តី! នេះជាការសាកល្បង។
audio_002		ប្រទេសកម្ពុជាមានវប្បធម៌ដ៏សម្បូរបែប។
audio_003		បច្ចេកវិទ្យាថ្មីៗកំពុងផ្លាស់ប្តូរជីវិត។
```

**Important Notes:**
- **Column 1**: Audio filename (without .wav extension)
- **Column 2**: Empty column (leave blank)
- **Column 3**: Khmer transcription text
- **Separator**: Use TAB character (not spaces)
- **Encoding**: Save as UTF-8

## 🔧 Configuration Files

### 1. Main Configuration (`config.json`)
```json
{
  "datasets": {
    "khmer": {
      "path": "data/km_kh_male",          ← Your dataset path
      "format": "tsv",
      "language": "km",
      "description": "Khmer male voice dataset"
    }
  }
}
```

### 2. Alternative Dataset Paths

If you want to use a different location, update the path in `config.json`:

```json
{
  "datasets": {
    "khmer": {
      "path": "/path/to/your/khmer/dataset",    ← Change this
      "format": "tsv",
      "language": "km"
    }
  }
}
```

**Examples:**
- **Local path**: `"path": "data/my_khmer_dataset"`
- **Absolute path**: `"path": "/home/user/datasets/khmer"`
- **Windows path**: `"path": "C:\\Datasets\\Khmer"`
- **Network path**: `"path": "\\\\server\\datasets\\khmer"`

## 📋 Dataset Requirements

### Audio Files
- **Format**: WAV files (recommended)
- **Sample Rate**: 22050 Hz (preferred) or 16000 Hz
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit
- **Duration**: 1-10 seconds per file (optimal)
- **Quality**: Clear speech, minimal background noise

### Text Requirements
- **Language**: Khmer text (Unicode range U+1780-U+17FF)
- **Encoding**: UTF-8
- **Content**: Clean, grammatically correct Khmer sentences
- **Length**: 10-500 characters per sentence

### Minimum Dataset Size
- **For testing**: 100+ samples
- **For basic training**: 1,000+ samples  
- **For good quality**: 5,000+ samples
- **For high quality**: 10,000+ samples

## 🚀 Quick Setup Steps

### Step 1: Prepare Your Dataset
1. Create the directory structure:
   ```bash
   mkdir -p data/km_kh_male/wavs
   ```

2. Copy your audio files to `data/km_kh_male/wavs/`

3. Create `data/km_kh_male/line_index.tsv` with your metadata

### Step 2: Verify Dataset
```bash
# Run the verification script
python verify_khmer_setup.py
```

**Expected output:**
```
🇰🇭 Checking Khmer dataset...
   ✅ Metadata file: XXXX samples
   ✅ Audio files: XXXX WAV files
```

### Step 3: Test Training
```bash
# Test with a small batch first
python train.py --model vits
```

## 📝 Example Dataset Creation

### Create Sample Metadata File
```python
import pandas as pd

# Sample data
data = [
    ["sample_001", "", "សួស្តី! នេះជាការសាកល្បង។"],
    ["sample_002", "", "ប្រទេសកម្ពុជាមានវប្បធម៌ដ៏សម្បូរបែប។"],
    ["sample_003", "", "បច្ចេកវិទ្យាថ្មីៗកំពុងផ្លាស់ប្តូរជីវិត។"]
]

# Create DataFrame
df = pd.DataFrame(data, columns=["filename", "empty", "transcription"])

# Save as TSV
df.to_csv("data/km_kh_male/line_index.tsv", sep="\t", index=False, header=False)
```

### Validate Audio Files
```python
import librosa
from pathlib import Path

audio_dir = Path("data/km_kh_male/wavs")
for audio_file in audio_dir.glob("*.wav"):
    try:
        duration = librosa.get_duration(path=str(audio_file))
        print(f"{audio_file.name}: {duration:.2f}s")
    except Exception as e:
        print(f"Error with {audio_file.name}: {e}")
```

## 🔍 Troubleshooting

### Common Issues

#### 1. "Dataset not found"
- Check the path in `config.json`
- Ensure the directory exists
- Verify folder permissions

#### 2. "No valid samples found"
- Check metadata file format (TSV with tabs)
- Verify audio files exist in `wavs/` directory
- Ensure filenames match between metadata and audio files

#### 3. "Encoding errors"
- Save metadata file as UTF-8
- Check for special characters in filenames
- Ensure Khmer text is properly encoded

#### 4. "Audio processing errors"
- Check audio file formats (WAV recommended)
- Verify sample rates (22050 Hz preferred)
- Test with a few files first

### Debug Commands
```bash
# Check dataset structure
ls -la data/km_kh_male/
ls -la data/km_kh_male/wavs/ | head -10

# Check metadata file
head -5 data/km_kh_male/line_index.tsv
wc -l data/km_kh_male/line_index.tsv

# Test audio files
python -c "
import librosa
print('Testing audio file...')
duration = librosa.get_duration(path='data/km_kh_male/wavs/your_first_file.wav')
print(f'Duration: {duration:.2f}s')
"
```

## 📊 Dataset Statistics

After setup, you can check your dataset statistics:

```bash
python verify_khmer_setup.py
```

This will show:
- Total number of samples
- Audio file count
- Duration statistics
- Text statistics
- Quality validation results

## 🎯 Ready for Training

Once your dataset is properly set up:

1. ✅ Dataset in correct location
2. ✅ Metadata file formatted correctly  
3. ✅ Audio files accessible
4. ✅ Verification script passes

You can start training immediately:
```bash
python train.py --model vits
```

## 📞 Need Help?

If you encounter issues:
1. Run `python verify_khmer_setup.py` for diagnostics
2. Check the troubleshooting section above
3. Review the example dataset creation code
4. Open an issue on GitHub with error details

**Happy training! 🇰🇭🎉**
