import pandas as pd
import pickle
import logging
from pathlib import Path
import json
from prophet import Prophet  # Prophet for forecasting

# Config
DATA_PATH = Path("data_simulation/data.csv")
MODEL_PATH = Path("models/model.pkl")
METADATA_PATH = Path("models/metadata.json")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df.rename(columns={"date": "ds", "demand": "y"})  # Prophet expects 'ds' and 'y'
    return df

def train_model(df: pd.DataFrame):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive'  # because your simulation uses additive effects
    )

    # ➡️ Add custom monthly seasonality (30 days)
    model.add_seasonality(
        name='monthly',
        period=30,
        fourier_order=5  # controls how flexible the seasonal curve is
    )

    model.fit(df)
    return model

def save_model(model, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def save_metadata(df, path=METADATA_PATH):
    last_date = df["ds"].max().strftime("%Y-%m-%d")
    metadata = {"last_trained_date": last_date}
    with open(path, 'w') as f:
        json.dump(metadata, f)

def main():
    logger.info("Loading data...")
    df = load_data()
    logger.info("Training Prophet model with dual seasonality...")
    model = train_model(df)
    logger.info("Saving model...")
    save_model(model)
    save_metadata(df)
    logger.info(f"✅ Model trained and saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
