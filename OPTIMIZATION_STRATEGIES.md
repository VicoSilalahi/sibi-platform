# SIBI Platform: Advanced Optimization Strategies

To further push the performance and reliability of the SIBI recognition system on low-end hardware, we can implement the following strategies:

## 1. Multi-Threaded Data Flow (Pipeline Optimization)
**Objective**: Decouple camera capture, landmark extraction, and model inference.

- **Background Extraction**: Move the `mediapipe.Holistic` processing to a dedicated Python thread (or process). This ensures the UI remains responsive (60+ FPS) even if the landmark extraction is slow ($\approx 15-20$ FPS).
- **Inference Throttling**: Run the GRU model only when a new landmark set is available, rather than waiting for it in the main loop.

## 2. Temporal Smoothing & Jitter Reduction (Accuracy Optimization)
**Objective**: Clean up the "noisy" landmark data before it reaches the GRU.

- **Savitzky-Golay Filter**: Apply a sliding window smoothing filter to the landmark time-series. This removes high-frequency jitter (hand tremors) which often confused sequence-based models like GRU/LSTM.
- **One-Euro Filter**: A frequency-based filter specifically designed for low-latency human-computer interaction to balance jitter reduction and lag.

## 3. Data Augmentation for Landmarks (Robustness Optimization)
**Objective**: Make the model resistant to variations in camera angle and distance.

- **Spatial Augmentation**: Randomly rotate, scale, and shift the landmarks in the training set.
- **Temporal Augmentation**: Randomly drop/repeat frames in the sequences to simulate different signing speeds.
- **Noise Injection**: Add small Gaussian noise to the landmark coordinates to prevent the model from overfitting to perfect coordinates.

## 4. Hardware-Specific Optimizations
**Objective**: Squeeze every bit of power from the CPU.

- **MediaPipe ROI (Region of Interest)**: Once hands are detected, we can crop the MediaPipe input to just the areas around the hands/face to speed up the underlying CNN.
- **OpenVINO / ONNX**: For Intel-based low-end CPUs, converting the Keras model to OpenVINO can yield a $2x-5x$ speedup over raw TensorFlow.
