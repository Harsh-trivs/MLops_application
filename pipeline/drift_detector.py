import numpy as np
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self, window_size=7, threshold=25):
        self.window_size = window_size
        self.threshold = threshold
        self.errors = []

    def update(self, error, timestamp=None):
        self.errors.append((timestamp, error) if timestamp else ("unknown", error))

    def should_retrain(self):
        if len(self.errors) < self.window_size:
            return False
        recent_errors = [e for _, e in self.errors[-self.window_size:]]
        recent_mae = np.mean(recent_errors)
        if recent_mae > self.threshold:
            logger.warning(f"🚨 Drift detected! Recent MAE = {recent_mae:.2f} > threshold = {self.threshold}")
            return True
        return False

    def reset(self):
        self.errors = []
