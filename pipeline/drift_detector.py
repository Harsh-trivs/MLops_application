import numpy as np

class DriftDetector:
    def __init__(self, window_size=7, threshold=25.0):
        self.window_size = window_size
        self.threshold = threshold
        self.errors = []

    def update(self, error):
        """Add a new error to the rolling window."""
        self.errors.append(error)
        if len(self.errors) > self.window_size:
            self.errors.pop(0)

    def should_retrain(self):
        """Return True if average error exceeds threshold."""
        if len(self.errors) < self.window_size:
            return False
        mean_error = np.mean(self.errors)
        return mean_error > self.threshold

    def get_recent_mae(self):
        if len(self.errors) == 0:
            return 0.0
        return np.mean(self.errors)


if __name__ == "__main__":
    detector = DriftDetector(window_size=3, threshold=10.0)
    for e in [5, 12, 15]:
        detector.update(e)
        print(f"Updated with error: {e} | MAE: {detector.get_recent_mae():.2f} | Retrain: {detector.should_retrain()}")