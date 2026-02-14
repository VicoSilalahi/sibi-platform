import os
from sklearn.model_selection import train_test_split
from app.config import ACTIONS, MODEL_PATH
from app.models.camera_gru import get_camera_gru_model
from app.training.data_loader import load_data
from app.models.utils import save_model

# --- Tambahkan Bagian Ini di Paling Atas File train_camera.py ---
import sys
import os

# Pastikan folder tempat config_gpu.py berada bisa terbaca
# Sesuaikan path ini dengan lokasi Anda menyimpan file config_gpu.py tadi

try:
    from app.training.config_gpu import configure_gpu
    configure_gpu() # <--- INI PENTING!
except ImportError:
    print("⚠️ Warning: config_gpu.py tidak ditemukan, mencoba jalan tanpa config.")

def train_camera():
    # Load data
    print("Loading camera data...")
    X, y = load_data(modality='camera')
    
    if len(X) == 0:
        print("No data found for camera modality.")
        return

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    # Initialize model
    model = get_camera_gru_model()

    # Callbacks
    import tensorflow as tf
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001
    )

    # Train
    print("Starting training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=[early_stop, reduce_lr]
    )

    # Save
    save_model(model, "camera_gru")
    print("Training complete.")

if __name__ == "__main__":
    train_camera()
