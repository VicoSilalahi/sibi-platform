from app.config import ACTIONS, CAMERA_INPUT_SHAPE

def get_camera_gru_model():
    """Build and return the optimized Camera GRU model."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, Flatten, TimeDistributed, Conv1D, MaxPooling1D, Input
    model = Sequential([
        Input(shape=CAMERA_INPUT_SHAPE),  # (30, 75, 3)
        
        # Spatial feature extraction per frame
        TimeDistributed(Conv1D(32, kernel_size=3, activation='relu')),
        TimeDistributed(MaxPooling1D(pool_size=2)),
        TimeDistributed(Flatten()),
        
        Dropout(0.2),
        
        # Temporal modeling with GRU (Optimized: single layer, fewer units, unrolled for TFLite)
        GRU(64, return_sequences=False, unroll=True),
        
        Dropout(0.2),
        
        # Classification head
        Dense(64, activation='relu'),
        Dense(len(ACTIONS), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model

if __name__ == "__main__":
    model = get_camera_gru_model()
    model.summary()
