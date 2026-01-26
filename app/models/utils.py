import os
from app.config import MODEL_PATH

def save_model(model, name):
    """Save the model to the configured model path."""
    path = os.path.join(MODEL_PATH, f"{name}.keras")
    model.save(path)
    print(f"Model saved to {path}")
    return path

def load_model(name):
    """Load the model from the configured model path."""
    import tensorflow as tf
    path = os.path.join(MODEL_PATH, f"{name}.keras")
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
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
        
        tflite_model = converter.convert()
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TF-Lite model saved to {tflite_path}")
        return tflite_path
    else:
        print(f"Keras model not found at {keras_path}")
        return None
