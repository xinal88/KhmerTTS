# 📊 Khmer Dataset Directory

## 🚨 Dataset Not Included

This directory is **empty** because the actual dataset is not included in the repository to keep it lightweight and fast to clone.

## 📋 What You Need to Add

### 1. Audio Files
Place your Khmer audio files in the `wavs/` directory:
```
data/km_kh_male/wavs/
├── audio_001.wav
├── audio_002.wav
├── audio_003.wav
└── ... (your audio files)
```

### 2. Metadata File
Create a `line_index.tsv` file in this directory:
```
data/km_kh_male/line_index.tsv
```

**Format** (tab-separated):
```
audio_001		សួស្តី! នេះជាការសាកល្បង។
audio_002		ប្រទេសកម្ពុជាមានវប្បធម៌ដ៏សម្បូរបែប។
audio_003		បច្ចេកវិទ្យាថ្មីៗកំពុងផ្លាស់ប្តូរជីវិត។
```

## 📖 Complete Setup Guide

For detailed instructions, see: **[DATASET_SETUP.md](../../DATASET_SETUP.md)**

## ✅ Verification

After adding your dataset, verify it works:
```bash
python verify_khmer_setup.py
```

## 🎯 Expected Structure

After setup, this directory should look like:
```
data/km_kh_male/
├── README.md              # This file
├── line_index.tsv         # Your metadata file
└── wavs/                  # Your audio files
    ├── audio_001.wav
    ├── audio_002.wav
    └── ... (more files)
```

## 📞 Need Help?

- Read the complete guide: [DATASET_SETUP.md](../../DATASET_SETUP.md)
- Run verification: `python verify_khmer_setup.py`
- Check troubleshooting section in the main guide
