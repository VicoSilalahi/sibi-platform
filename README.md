# SIBI Platform: Dynamic Sign Language Recognition

A modular, high-performance Python application for real-time SIBI (Sistem Isyarat Bahasa Indonesia) recognition. This platform is optimized for low-end CPUs and provides a complete pipeline from data collection to deployment.

## High-Performance Features
- **GPU-Grade Performance on CPU**: Transitioned from LSTM to optimized GRU layers with TF-Lite support (~33% faster).
- **Parallel Pipeline**: Multi-threaded landmark extraction ensures a consistent 60 FPS UI responsiveness.
- **Robust Inference**: Signal filtering (EMA) reduces landmark jitter for higher accuracy.
- **Dual Modality**: Built-in support for **Camera** (MediaPipe) and **Glove** (Serial Sensor) inputs.
- **Instant Initialization**: Lazy imports reduce startup time from 30s to <1s.

## Unified CLI Control
All platform features are accessible through the central **`sibi.py`** script.

### 1. Installation
```bash
pip install opencv-python mediapipe tensorflow scikit-learn numpy
```

### 2. Action Management
```bash
python sibi.py dataset --list              # View statistics (Original vs. Total)
python sibi.py dataset --add "terima_kasih" # Add new sign
```

### 3. Data Collection
Build your dataset using direct recording or raw source logs (AVI/CSV):

**Option A: Direct Recording**
```bash
python sibi.py collect --modality camera --action yang  # Camera only
python sibi.py collect --modality glove --action yang   # Glove only
python sibi.py collect --modality both --action yang    # Synchronized Both
```

**Option B: Raw Ingestion Pipeline**
```bash
# 1. Record raw logs (Unique filename: [hostname]_[timestamp])
python sibi.py record --modality camera --action yang --samples 10

# 2. Extract landmarks and mark files in-place as '_done'
python sibi.py ingest --modality camera

# 3. Consolidate and Clean (Rename old files & remove 'processed' subfolders)
python sibi.py migrate
python sibi.py migrate --undo # Reset '_done' files to allow re-ingestion
```

### 4. Optimize & Train
```bash
python sibi.py augment                     # 4x dataset via augmentation
python sibi.py train --modality camera     # Train GRU model
python sibi.py export                      # Export to TF-Lite (CPU Speedup)
```

### 5. Run Inference
```bash
python sibi.py run --modality camera
```

### 6. Cloud Training
If your local machine is too slow for training, use Google Colab:
- See the provided [train_on_colab.ipynb](./train_on_colab.ipynb) for instructions on how to offload training to the cloud.

## Project Structure
- `sibi.py`: The unified control script.
- `scripts/migrate_data.py`: Dataset consolidation and unique naming tool.
- `app/data/`: Extraction modules (Webcam, Video, Glove).
- `app/models/`: GRU architectures and TFLite utilities.
- `app/inference/`: Gating logic, signal filters, and stability pass.
- `app/training/`: Training loops, augmentation, and ingestion scripts.
- `datasets/`: The landmark-based "Source of Truth" (.npy).
- `raw_videos/` & `raw_glove/`: Flat storage for raw source data.

## Technical Reference
- [TECHNICAL_EXPLANATION.md](./TECHNICAL_EXPLANATION.md): Deep dive into architecture and optimizations.
- [OPTIMIZATION_STRATEGIES.md](./OPTIMIZATION_STRATEGIES.md): Roadmap for advanced hardware tuning.
