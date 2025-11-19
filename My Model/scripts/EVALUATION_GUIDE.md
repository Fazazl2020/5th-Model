# CMGAN-Style Evaluation Guide for Your Model

## Overview

This guide explains the modifications made to align your model's testing mechanism with CMGAN's approach while maintaining compatibility with your trained checkpoint.

---

## 🔑 Key Points

### **CRITICAL: STFT Parameters Cannot Be Changed**

Your model was trained with:
- **n_fft: 320** (→ 161 frequency bins)
- **hop_length: 160**
- **power: 0.3**

❌ **DO NOT** change these to CMGAN's values (400/100) as it will break compatibility with your checkpoint!

✅ The new evaluation script uses YOUR parameters while following CMGAN's testing procedure.

---

## 📁 New Files Created

### 1. **evaluation.py** (Main evaluation script)
- Location: `My Model/scripts/evaluation.py`
- Purpose: CMGAN-style file-by-file evaluation
- Features:
  - File-by-file processing (like CMGAN)
  - Audio repetition padding (like CMGAN)
  - RMS normalization (like CMGAN)
  - Inline metrics computation
  - Compatible with your 320/160 STFT parameters

### 2. **tools/compute_metrics.py** (Metrics computation)
- Location: `My Model/scripts/tools/compute_metrics.py`
- Purpose: Compute speech enhancement metrics
- Metrics: PESQ, CSIG, CBAK, COVL, SSNR, STOI

### 3. **tools/__init__.py**
- Location: `My Model/scripts/tools/__init__.py`
- Purpose: Python package initialization

---

## 🚀 Installation

### Install Required Dependencies

```bash
# Install metrics libraries
pip install pesq pystoi pysepm

# Optional: For better CSIG/CBAK/COVL computation
pip install pysepm
```

### Verify Installation

```bash
cd "/home/user/5th-Model/My Model/scripts"
python tools/compute_metrics.py
```

Expected output: Test metrics computation with dummy signals

---

## 📋 Usage

### Basic Usage

```bash
cd "/home/user/5th-Model/My Model/scripts"

python evaluation.py \
    --model_path ./ckpt/models/best_pesq.pt \
    --test_dir /gdata/fewahab/data/Voicebank+demand/My_train_valid_test/test \
    --save_dir ./saved_tracks_evaluation \
    --save_tracks True
```

### Advanced Usage

```bash
python evaluation.py \
    --model_path ./ckpt/models/best_pesq.pt \
    --noisy_dir /path/to/test/noisy \
    --clean_dir /path/to/test/clean \
    --save_dir ./results \
    --save_tracks True \
    --n_fft 320 \
    --hop 160 \
    --power 0.3 \
    --device cuda
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `./ckpt/models/best_pesq.pt` | Path to model checkpoint |
| `--test_dir` | (your test dir) | Base test directory with noisy/ and clean/ subdirs |
| `--noisy_dir` | `test_dir/noisy` | Noisy test files |
| `--clean_dir` | `test_dir/clean` | Clean reference files |
| `--save_tracks` | `True` | Save enhanced audio |
| `--save_dir` | `./saved_tracks_evaluation` | Output directory |
| `--n_fft` | `320` | FFT size ⚠️ **DO NOT CHANGE** |
| `--hop` | `160` | Hop length ⚠️ **DO NOT CHANGE** |
| `--power` | `0.3` | Power compression exponent |
| `--device` | `cuda` | Device (cuda or cpu) |

---

## 📊 Expected Output

```
======================================================================
CMGAN-STYLE EVALUATION (Adapted for Your Model)
======================================================================
Model: ./ckpt/models/best_pesq.pt
Noisy dir: /path/to/noisy
Clean dir: /path/to/clean
Output dir: ./saved_tracks_evaluation
STFT params: n_fft=320, hop=160, power=0.3
Device: cuda
======================================================================

Loading model...
✓ Model loaded from: ./ckpt/models/best_pesq.pt
  Epoch: 50
  Best loss: 0.1234
✓ Output directory: ./saved_tracks_evaluation

Found 824 audio files to process

======================================================================
Processing files...
======================================================================
  Progress: 100/824 files (12.1%)
  Progress: 200/824 files (24.3%)
  ...
  Progress: 824/824 files (100.0%)

======================================================================
EVALUATION COMPLETED
======================================================================
Processed: 824 files

Average Metrics (over 824 files):
  PESQ: 2.8534
  CSIG: 4.2156
  CBAK: 3.8923
  COVL: 3.7821
  SSNR: 8.2341 dB
  STOI: 0.9456

Enhanced tracks saved to: ./saved_tracks_evaluation
======================================================================
```

---

## 🔄 Comparison: Old vs New Testing

### Old Method (`test.py` → `models.py::test()`)

```bash
python test.py
```

**Features:**
- DataLoader-based batch processing
- SNR-organized output structure
- Saves to `estimates_MA/snr_-06/`, `snr_000/`, etc.
- No inline metrics (expects external evaluation)
- Uses configs.py for all parameters

### New Method (`evaluation.py`)

```bash
python evaluation.py --model_path ... --test_dir ...
```

**Features:**
- File-by-file processing (CMGAN-style)
- Single output directory
- Inline metrics computation
- Audio repetition padding (CMGAN-style)
- Command-line arguments
- More similar to CMGAN's evaluation.py

---

## 📦 Files Summary

### Files You Should Keep (Unchanged)

✅ **Training files** (still needed):
- `train.py` - Training script
- `configs.py` - Configuration
- `utils/models.py` - Model class with train() and old test()
- `utils/networks.py` - Network architecture
- `utils/pipeline_modules.py` - Audio processing pipeline
- `utils/criteria.py` - Loss functions
- `utils/data_utils.py` - Data loading
- `utils/stft.py` - STFT wrapper
- `utils/utils.py` - Utility functions

✅ **Checkpoint files** (DON'T REMOVE):
- `ckpt/models/best.pt` - Best loss model
- `ckpt/models/best_pesq.pt` - Best PESQ model
- `ckpt/models/latest.pt` - Latest checkpoint
- All other checkpoint files

### Files You Can Optionally Rename

🔄 **Old test.py** (optional):
```bash
# Rename if you want to keep both testing methods
mv test.py test_snr_organized.py
```

Then you can use:
- `python test_snr_organized.py` - For SNR-organized output
- `python evaluation.py` - For CMGAN-style evaluation

### New Files (Created)

✨ **New evaluation files**:
- `evaluation.py` - CMGAN-style evaluation script
- `tools/compute_metrics.py` - Metrics computation
- `tools/__init__.py` - Package initialization
- `EVALUATION_GUIDE.md` - This guide

---

## ⚠️ Important Notes

### 1. **Checkpoint Compatibility**

Your model checkpoint expects:
- Input: `[batch, 2, time, freq]` with 161 freq bins
- STFT: n_fft=320, hop=160
- Power compression: 0.3

The new evaluation.py **maintains** these parameters. ✅

### 2. **CMGAN Differences**

CMGAN uses different parameters that are **NOT compatible** with your checkpoint:
- CMGAN: n_fft=400 (201 bins) ❌
- Your model: n_fft=320 (161 bins) ✅

### 3. **What Was Aligned**

✅ **Successfully aligned with CMGAN:**
- File-by-file processing loop
- Audio repetition padding strategy
- RMS normalization approach
- Power compression/decompression
- Metrics computation structure
- Progress reporting style

❌ **Cannot be aligned** (would break checkpoint):
- STFT parameters (n_fft, hop_length)
- Network architecture (different freq bins)

---

## 🧪 Testing the New Evaluation

### Quick Test (Single File)

```bash
cd "/home/user/5th-Model/My Model/scripts"

# Create a small test directory
mkdir -p test_eval/noisy test_eval/clean

# Copy a few test files
cp /gdata/fewahab/data/Voicebank+demand/My_train_valid_test/test/noisy/*.wav test_eval/noisy/ | head -5
cp /gdata/fewahab/data/Voicebank+demand/My_train_valid_test/test/clean/*.wav test_eval/clean/ | head -5

# Run evaluation
python evaluation.py \
    --model_path ./ckpt/models/best_pesq.pt \
    --noisy_dir ./test_eval/noisy \
    --clean_dir ./test_eval/clean \
    --save_dir ./test_eval/output
```

### Full Test

```bash
python evaluation.py \
    --model_path ./ckpt/models/best_pesq.pt \
    --test_dir /gdata/fewahab/data/Voicebank+demand/My_train_valid_test/test \
    --save_dir ./full_evaluation_results
```

---

## 📝 Summary

### What to Do

1. ✅ **Install dependencies**: `pip install pesq pystoi pysepm`

2. ✅ **Use the new evaluation.py** for CMGAN-style testing:
   ```bash
   python evaluation.py --model_path ./ckpt/models/best_pesq.pt --test_dir /path/to/test
   ```

3. ✅ **Optionally keep old test.py** for SNR-organized output:
   ```bash
   mv test.py test_snr_organized.py
   ```

### What NOT to Do

1. ❌ **Don't change n_fft or hop_length** in evaluation.py (breaks checkpoint compatibility)

2. ❌ **Don't delete checkpoint files**

3. ❌ **Don't delete training files** (you might need to retrain or resume)

---

## 🤝 Support

If you encounter issues:

1. **Metrics not working**: Install dependencies
   ```bash
   pip install pesq pystoi pysepm
   ```

2. **Model not loading**: Check model_path points to correct checkpoint

3. **Wrong output shape**: Verify n_fft=320 and hop=160 (don't change!)

4. **Out of memory**: Reduce `--cut_len` parameter

---

## 📚 References

- CMGAN Paper: [Link if available]
- PESQ: ITU-T P.862
- STOI: IEEE/ACM TASLP 2011
- Your Network: Custom architecture with 161 frequency bins

---

**Created**: 2025-11-19
**Author**: Claude (Anthropic)
**Purpose**: Align testing mechanism with CMGAN while preserving checkpoint compatibility
