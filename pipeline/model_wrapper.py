import pickle
from pathlib import Path
import pandas as pd
import json

# Config
MODEL_PATH = Path("models/model.pkl")
METADATA_PATH = Path("models/metadata.json")

class ForecastingModel:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.model = None

    def load(self):
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

    def predict_n_days(self, n):
        if self.model is None:
            raise ValueError("Model not loaded. Call .load() first.")

        last_date = self.get_last_trained_date()
        if last_date is None:
            raise ValueError("Last trained date not found in metadata.")

        last_date = pd.to_datetime(last_date)

        # Create future dataframe
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n)

        future_df = pd.DataFrame({"ds": future_dates})

        # Prophet prediction
        forecast = self.model.predict(future_df)

        # We return the full forecast dataframe
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def get_last_trained_date(self):
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                meta = json.load(f)
                return meta.get("last_trained_date")
        return None
