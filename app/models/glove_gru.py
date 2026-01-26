from app.config import ACTIONS, GLOVE_INPUT_SHAPE

def get_glove_gru_model():
    """Build and return the optimized Glove GRU model."""
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout, Input
    model = Sequential([
        Input(shape=GLOVE_INPUT_SHAPE),  # (30, 22)
        
        # Temporal modeling with GRU (Optimized: single layer, small hidden state, unrolled for TFLite)
        GRU(32, return_sequences=False, unroll=True),
        
        Dropout(0.2),
        
        # Classification head
        Dense(32, activation='relu'),
        Dense(len(ACTIONS), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model

if __name__ == "__main__":
    model = get_glove_gru_model()
    model.summary()
