import pandas as pd
from pathlib import Path
import json
import logging
import time

from drift_detector import DriftDetector
from model_trainer import save_metadata, save_model, train_model
from model_wrapper import ForecastingModel

# Config
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / ".." / "data_simulation" / "data.csv"
MODEL_PATH = BASE_DIR  / "models" / "model.pkl"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
SEASONAL_PERIODS = 30
DRIFT_WINDOW = 7
DRIFT_THRESHOLD = 45

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"], index_col="date").asfreq("D")
    return df

def train_initial_model(df):
    """Train the initial model using the first 6 months of data."""
    initial_data = df.head(6 * 30)  # First 6 months of data
    logger.info("Training initial model using first 6 months of data...")
    model = train_model(initial_data.reset_index().rename(columns={"date": "ds", "demand": "y"}))
    save_model(model)
    save_metadata(initial_data.reset_index().rename(columns={"date": "ds", "demand": "y"}))
    logger.info("Initial model trained and saved.")

def retrain_model(start_date, df):
    """Retrain model from start_date onwards."""
    logger.info(f"🔁 Retraining model from {start_date.date()} onwards...")
    time.sleep(5)  # Simulate time taken to retrain
    df_subset = df.loc[:start_date]
    model = train_model(df_subset.reset_index().rename(columns={"date": "ds", "demand": "y"}))
    save_model(model)
    save_metadata(df_subset.reset_index().rename(columns={"date": "ds", "demand": "y"}))
    logger.info("✅ Model retrained.")

def get_last_trained_date():
    """Retrieve the date the model was last trained from metadata."""
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)
            return pd.to_datetime(meta.get("last_trained_date"))
    return None

def main():
    df = load_data()
    drift_detector = DriftDetector(window_size=DRIFT_WINDOW, threshold=DRIFT_THRESHOLD)
    
    # Step 1: Train initial model with first 6 months of data
    train_initial_model(df)

    model = ForecastingModel()
    model.load()
    last_trained_date = get_last_trained_date()

    current_date = last_trained_date + pd.Timedelta(days=1)

    # Step 2: Process remaining data and check for drift
    while current_date <= df.index.max():
        forecast_df = model.predict_n_days((current_date - last_trained_date).days)
        forecast_row = forecast_df[forecast_df['ds'] == current_date]

        if forecast_row.empty:
            logger.warning(f"No forecast available for {current_date.date()}. Skipping...")
            current_date += pd.Timedelta(days=1)
            continue

        predicted = forecast_row['yhat'].values[0]

        # Get actual
        actual = df.loc[current_date, "demand"]
        error = abs(actual - predicted)

        logger.info(f"📅 {current_date.date()} | Prediction: {predicted:.2f} | Actual: {actual:.2f} | Error: {error:.2f}")

        # Update drift detector
        drift_detector.update(error, current_date)

        if drift_detector.should_retrain():
            drift_start_date = current_date - pd.Timedelta(days=DRIFT_WINDOW)
            last_trained_date = get_last_trained_date()

            if drift_start_date == last_trained_date:
                logger.warning("Drift detected even after training. Need more data or Potentially an outlier exists.")
                drift_detector.reset()
                current_date += pd.Timedelta(days=1)
                continue

            # Retrain the model from the drift start date
            retrain_model(drift_start_date, df)
            model.load()  # Reload the newly retrained model
            drift_detector.reset()

            # Resume from the next day after retraining
            current_date = drift_start_date + pd.Timedelta(days=1)
            continue

        current_date += pd.Timedelta(days=1)

if __name__ == "__main__":
    main()
