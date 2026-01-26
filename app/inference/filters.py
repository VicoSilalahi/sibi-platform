import numpy as np

class EMAFilter:
    """Exponential Moving Average filter for multi-dimensional data."""
    
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.state = None

    def apply(self, data):
        """Apply EMA filter to data.
        
        Args:
            data (np.array): New data point (e.g., shape (75, 3))
        Returns:
            np.array: Smoothed data point.
        """
        if self.state is None:
            self.state = data.copy()
        else:
            self.state = self.alpha * data + (1 - self.alpha) * self.state
        return self.state

    def reset(self):
        self.state = None
