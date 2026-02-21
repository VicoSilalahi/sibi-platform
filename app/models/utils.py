import os
from app.config import MODEL_PATH

# Compatibility Fix: Keras 3 sometimes fails to load models saved with older versions
# due to unrecognized arguments like 'time_major' in GRU layers.
import tensorflow as tf
try:
    import keras
    from keras import layers
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers

# Store original GRU
OriginalGRU = layers.GRU

class FixedGRU(OriginalGRU):
    def __init__(self, *args, **kwargs):
        # Strip arguments that cause ValueError in Keras 3
        kwargs.pop('time_major', None)
        kwargs.pop('reset_after', None)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, config):
        # Strip from config as well
        config.pop('time_major', None)
        config.pop('reset_after', None)
        return super().from_config(config)

# Monkey-patch at the source
layers.GRU = FixedGRU
tf.keras.layers.GRU = FixedGRU
if hasattr(tf.keras.utils, 'get_custom_objects'):
    tf.keras.utils.get_custom_objects().update({'GRU': FixedGRU})

def save_model(model, name):
    """Save the model to the configured model path."""
    path = os.path.join(MODEL_PATH, f"{name}.keras")
    model.save(path)
    print(f"Model saved to {path}")
    return path

def load_model(name):
    """Load the model from the configured model path with Keras 3 compatibility fixes."""
    # Try exact path first, then with .keras
    path = os.path.join(MODEL_PATH, name)
    if not os.path.exists(path):
        path = os.path.join(MODEL_PATH, f"{name}.keras")
        
    if os.path.exists(path):
        print(f"Loading model from {path}")
        # Register custom objects globally for this load session
        custom_objects = {'GRU': FixedGRU}
        
        try:
            # Try loading with custom objects
            return tf.keras.models.load_model(path, custom_objects=custom_objects)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Attempting to load without custom objects...")
            try:
                return tf.keras.models.load_model(path)
            except Exception as e2:
                print(f"Failed to load model: {e2}")
                return None
    else:
        print(f"Model not found at {path}")
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
