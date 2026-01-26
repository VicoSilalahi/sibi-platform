# SIBI Platform: Dynamic Sign Language Recognition

A modular, optimized Python application for real-time SIBI (Sistem Isyarat Bahasa Indonesia) recognition using GRU-based deep learning. Designed for low-end CPUs and industrial reliability.

## Key Features
- **Optimized for CPU**: Uses GRU instead of LSTM and supports TF-Lite for low-latency inference.
- **Modular Data Ingestion**: Unified landmark extraction for live webcam and offline videos.
- **Dual Modality**: Built-in support for Camera (MediaPipe) and Glove (Serial/Raw) data.
- **Stable Inference**: Sliding temporal windows (30 frames) with gating and confidence thresholds.
- **Instant Loading**: Implementation of lazy imports reduces startup time from 30s to <1s.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install opencv-python mediapipe tensorflow scikit-learn numpy
   ```

## Quick Start

**Action Management:**
View current actions and sample statistics:
```bash
python -m app.training.manage_dataset --list
```

Add a new sign to the system:
```bash
python -m app.training.manage_dataset --add "nama_sign"
```

### 1. Data Collection
Build your dataset by recording new samples:
```bash
python -m app.training.collect_data --modality camera --action yang --samples 20
```

### 2. Data Augmentation (Optional)
Multiply your dataset size (4x) with synthetic variations:
```bash
python -m app.training.augment_data
```

### 3. Training
Train the modality-specific models:
```bash
python -m app.training.train_camera
python -m app.training.train_glove
```

### 3. Inference
Run real-time recognition with the UI:
```bash
python -m app.main --mode camera
```

## Project Structure
- `app/data/`: Modality-specific extraction and readers.
- `app/models/`: GRU definitions and management utilities.
- `app/inference/`: Sliding window and gating logic.
- `app/training/`: Scripts for data collection and model training.
- `app/ui.py`: Lightweight OpenCV rendering.
- `app/config.py`: Shared constants and performance tweaks.
- `datasets/`: Storage for captured landmarks and sensor readings.

## Technical Details
For an in-depth breakdown of the architecture, data flow, and optimizations, see [TECHNICAL_EXPLANATION.md](./TECHNICAL_EXPLANATION.md).
