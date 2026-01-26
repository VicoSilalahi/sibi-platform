# Technical Explanation: SIBI Platform Architecture

This document provides a deep dive into the technical design and architectural decisions behind the refactored SIBI recognition system.

## 1. Data Processing Pipeline

The system follows a "Raw Landmarks" architecture to minimize storage and maximize CPU efficiency.

### Landmark Extraction
- **Camera Modality**: Uses MediaPipe Holistic. Instead of saving raw video frames, the system extracts 75 keypoints (33 pose, 21 left hand, 21 right hand) per frame.
- **Dimensionality**: Each landmark has $(x, y, z)$ coordinates, resulting in a $(75, 3)$ input vector per frame.
- **Normalization**: Coordinates are normalized by MediaPipe relative to the frame dimensions $(0.0 - 1.0)$.

### Sliding Window Mechanism
For real-time inference, the system maintains a queue (buffer) of the last 30 frames.
- **Temporal Depth**: 30 frames ($\approx 1$ second at 30 FPS) provides enough temporal context to capture dynamic signs.
- **Overlap**: Inference is performed on every new frame, providing a smooth "rolling" prediction.

## 2. Model Architecture

### Transition: LSTM → GRU
The original notebooks used LSTM (Long Short-Term Memory). While powerful, LSTMs are computationally expensive for low-end CPUs due to their 4-gate structure.
- **GRU (Gated Recurrent Unit)**: We switched to GRU, which has only 2 gates (update and reset). This reduces the parameter count and computation by $\approx 25-33\%$ while maintaining similar performance for this complexity of sign language.

### Camera Model (CNN-GRU)
1. **Spatial Features**: We use a `TimeDistributed(Conv1D)` layer. This treats the 75 landmarks as a 1D signal per frame, extracting spatial relationships before passing them to the temporal layer.
2. **Temporal Modeling**: A single GRU layer (64 units) processes the sequence of extracted spatial features.
3. **Dropout**: Systematic dropout (0.2) is used to prevent overfitting on the relatively small dataset.

## 3. Inference Stability & Gating

Static hands or background noise can trigger false positives. We implemented two specific filters:

### Gating (Motion Energy)
The `app/inference/gating.py` module calculates the variance of landmarks over the 30-frame window.
- **Logic**: If the average variance is below `MOTION_ENERGY_THRESHOLD`, the system assumes the hand is idle and returns "No sign" without running the expensive neural network inference.

### Temporal Stability Filter
Isolated high-confidence frames can cause flickering labels.
- **Logic**: The system keeps a small history of recent predictions. A label is only displayed if it remains consistent across `TEMPORAL_STABILITY_FRAMES` consecutive windows.

## 4. Performance Optimizations

### Lazy Loading
TensorFlow and MediaPipe intake significant RAM and add 10-30s to script initialization.
- **Implementation**: Imports are moved inside function scopes or guarded by global checks. Scripts like `collect_data.py` or `--help` commands now start instantly.

### Inference Striding (Frame Skipping)
To further reduce CPU load, the system allows skipping inference for a fixed number of frames.
- **`INFERENCE_STRIDE`**: Configured in `app/config.py`. If set to 5, the model only makes a prediction every 5 frames. Landmarks are still collected every frame to maintain a high-quality sliding window, but the computationally expensive neural network pass is throttled.
