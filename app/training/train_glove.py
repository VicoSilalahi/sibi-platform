from sklearn.model_selection import train_test_split
from app.models.glove_gru import get_glove_gru_model
from app.training.data_loader import load_data
from app.models.utils import save_model

def train_glove():
    # Load data
    print("Loading glove data...")
    X, y = load_data(modality='glove')
    
    if len(X) == 0:
        print("No data found for glove modality.")
        return

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

    # Initialize model
    model = get_glove_gru_model()

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
    save_model(model, "glove_gru")
    print("Training complete.")

if __name__ == "__main__":
    train_glove()
