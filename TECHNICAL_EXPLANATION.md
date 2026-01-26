# Technical Explanation: SIBI Platform Architecture

This document provides a deep dive into the engineering decisions that make the SIBI platform efficient enough to run on low-end hardware without sacrificing accuracy.

## 0. Synchronized Multi-Modality
The platform supports synchronized **Hybrid Collection** (`--modality both`).
- **Mechanism**: The system locks the sampling frequency to the camera's processing loop. For every landmark frame extracted, it simultaneously reads a corresponding serial packet from the ESP32.
- **Benefit**: This creates perfectly aligned `.npy` sequences for both vision and sensor modalities, allowing for future "ensemble" or "fusion" models.

## 1. The "Raw-First" Data Pipeline

Traditional video-based recognition is slow. Our platform converts visual data into a 1D landmark stream immediately.

### Parallel Landmark Extraction (Multi-threaded)
The system uses a dual-threaded pipeline (Capture & Process) to ensure a consistent 60 FPS UI, even on low-end hardware.

### Deterministic Video Ingestion
To maintain the relative order of recorded samples, the ingestion pipeline implements:
1. **Natural Numeric Sorting**: Videos named `1.avi`, `2.avi`, `10.avi` are processed in strict numerical order (1 -> 2 -> 10) instead of alphabetical (1 -> 10 -> 2).
2. **Post-Ingestion Migration**: Successfully processed videos are automatically moved to a `processed/` folder. This acts as a "checksum" to prevent duplicate data if the script is run multiple times.
This ensures the display (UI) consistently runs at 60 FPS, even if landmark extraction drops to 15-20 FPS.

### Signal Filtering (Jitter Reduction)
Landmarks extracted from webcams are naturally "jumpy" (jitter). We apply an **Exponential Moving Average (EMA) Filter** ($\alpha=0.6$) to the landmark stream. This smooths out micro-tremors, allowing the GRU model to focus on the overall intent of the gesture rather than sensor noise.

## 2. Model Evolution: LSTM to Optimized GRU

### Why GRU?
Long Short-Term Memory (LSTM) cells have four internal gates. The **Gated Recurrent Unit (GRU)** simplifies this to two gates (update/reset), reducing parameter count by $\approx 33\%$. For sign language, where temporal patterns are relatively short, GRU provides identical accuracy with significantly faster CPU execution.

### TF-Lite & Unrolling
Standard Recurrent Neural Networks (RNNs) use dynamic loops, which are difficult for mobile/edge CPUs to optimize.
- **Unrolled GRU**: We set `unroll=True` in our Keras layers. This replaces the temporal loop with a static graph of repeated operations, which is significantly faster in TF-Lite.
- **Select TF Ops**: We enabled `SELECT_TF_OPS` during conversion to ensure the complex internal operations of the GRU are supported in the TF-Lite runtime.

## 3. Inference Stability & Gating

Running a neural network 30 times a second on a low-end CPU is wasteful when no one is signing.

### Motion Energy Gating
Before running the model, we calculate the variance of hand landmarks over the last 30 frames. If the variance is below a threshold, the hand is considered "idle," and the system skips the inference pass, saving 100% of the model's compute power.

### Inference Striding (Frame Skipping)
To further reduce CPU load, the system allows skipping inference for a fixed number of frames.
- **`INFERENCE_STRIDE`**: Configured in `app/config.py`. If set to 5, the model only makes a prediction every 5 frames. Landmarks are still collected every frame to maintain a high-quality sliding window, but the computationally expensive neural network pass is throttled.

---

## 4. Software Engineering Best Practices

### Lazy Loading
To ensure the CLI feels responsive, we implemented lazy loading for TensorFlow and MediaPipe. These heavy libraries are only imported inside the specific functions that require them.
- **Result**: `python -m app.main --help` returns in **<0.5s**, compared to **30s+** if libraries were imported at the top level.

### Dynamic Configuration
All sign labels are managed via `actions.json`. The app dynamically builds its UI and final classification layers based on this file, allowing for easy extension without codebase modification.
