import os
import numpy as np
from app.config import MODEL_PATH

def save_model(model, name):
    """Save the model to the configured model path."""
    path = os.path.join(MODEL_PATH, f"{name}.keras")
    model.save(path)
    print(f"Model saved to {path}")
    return path

class TFLiteModel:
    """Wrapper for TF-Lite model to mimic Keras model.predict()."""
    def __init__(self, model_path):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, X):
        X = X.astype(np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], X)
        self.interpreter.invoke()
        res = self.interpreter.get_tensor(self.output_details[0]['index'])
        return res

def load_model(name):
    """Load the model (preferring .tflite for performance)."""
    tflite_path = os.path.join(MODEL_PATH, f"{name}.tflite")
    keras_path = os.path.join(MODEL_PATH, f"{name}.keras")
    
    # 1. Try TF-Lite first (Much faster for inference)
    if os.path.exists(tflite_path):
        print(f"Loading TF-Lite model: {tflite_path}")
        return TFLiteModel(tflite_path)
    
    # 2. Fallback to Keras
    if os.path.exists(keras_path):
        print(f"Loading Keras model: {keras_path} (Pro-tip: run 'sibi.py export' for faster speed)")
        import tensorflow as tf
        return tf.keras.models.load_model(keras_path)
    
    print(f"Model not found at {MODEL_PATH}")
    return None

def convert_to_tflite(name):
    """Convert a saved Keras model to TF-Lite format."""
    import tensorflow as tf
    keras_path = os.path.join(MODEL_PATH, f"{name}.keras")
    tflite_path = os.path.join(MODEL_PATH, f"{name}.tflite")
    
    if os.path.exists(keras_path):
        model = tf.keras.models.load_model(keras_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimize for size/latency
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # RNN Fixes: Enable Select TF ops and disable tensor list lowering
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS    # enable TensorFlow ops.
        ]
        converter._experimental_lower_tensor_list_ops = False
        
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TF-Lite model saved to {tflite_path}")
        return tflite_path
    else:
        print(f"Keras model not found at {keras_path}")
        return None
