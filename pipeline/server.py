from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import pandas as pd
import io
import json
import logging
from prometheus_client import start_http_server, Gauge, Counter
from drift_detector import DriftDetector
from model_trainer import save_metadata, save_model, train_model
from model_wrapper import ForecastingModel
from pathlib import Path
import time

# FastAPI app
app = FastAPI()

# Config
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
SEASONAL_PERIODS = 30
PROMETHEUS_PORT = 8001  # Port for Prometheus metrics

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Prometheus Metrics
drift_detected_metric = Counter('drift_detected_count', 'Total number of drift detections')
model_retrain_metric = Counter('model_retrain_count', 'Total number of model retrainings')
drift_not_enough_data_metric = Counter('drift_not_enough_data_count', 'Drift detected but not enough data for retraining')
predicted_metric = Gauge('predicted_demand', 'Predicted demand for a given date', ['date'])
actual_metric = Gauge('actual_demand', 'Actual demand for a given date', ['date'])

# Global state placeholders
model: Optional[ForecastingModel] = None
df_global: Optional[pd.DataFrame] = None
drift_detector: Optional[DriftDetector] = None


# Utility functions
def train_initial_model(df):
    initial_data = df.head(6 * 30)  # First 6 months
    logger.info("Training initial model...")
    model = train_model(initial_data.reset_index().rename(columns={"date": "ds", "demand": "y"}))
    save_model(model)
    save_metadata(initial_data.reset_index().rename(columns={"date": "ds", "demand": "y"}))
    logger.info("Initial model trained and saved.")

def retrain_model(start_date, df):
    logger.info(f"🔁 Retraining model from {start_date.date()} onwards...")
    df_subset = df.loc[:start_date]
    model = train_model(df_subset.reset_index().rename(columns={"date": "ds", "demand": "y"}))
    save_model(model)
    save_metadata(df_subset.reset_index().rename(columns={"date": "ds", "demand": "y"}))
    logger.info("✅ Model retrained.")

def get_last_trained_date():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)
            return pd.to_datetime(meta.get("last_trained_date"))
    return None


@app.post("/model_init")
async def model_init(file: UploadFile = File(...), window_size: int = Form(...), threshold: float = Form(...)):
    global model, df_global, drift_detector
    try : 
        # Load incoming CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), parse_dates=["date"], index_col="date").asfreq("D")
    
        df_global = df
        # Initialize drift detector
        drift_detector = DriftDetector(window_size=window_size, threshold=threshold)

        # Train initial model
        train_initial_model(df)
        model = ForecastingModel()
        model.load()
        return JSONResponse(content={"message": "Model initialized successfully."})
    except Exception as e:
        logger.exception("Error during processing")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/last_trained_date")
async def last_trained_date():
    try:
        last_trained_date = get_last_trained_date()
        if last_trained_date is not None:
            return JSONResponse(content={"last_trained_date": last_trained_date.strftime("%Y-%m-%d")})
        else:
            return JSONResponse(content={"error": "Model has not been trained yet."}, status_code=404)
    except Exception as e:
        logger.exception("Error during processing")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/predict_for_date")
async def predict_for_current_date(current_date: str):
    global model, df_global, drift_detector
    try:
        # Remove quotes if present
        current_date = current_date.replace('"', '')
        current_date = pd.to_datetime(current_date)
        drift_detected = 0
        last_trained_date = get_last_trained_date()
        if model is None or df_global is None:
            return JSONResponse(content={"error": "Model not initialized."}, status_code=500)
        forecast_df = model.predict_n_days((current_date - last_trained_date).days)
        forecast_row = forecast_df[forecast_df['ds'] == current_date]
        if forecast_row.empty:
            return JSONResponse(content={"error": "No forecast available for the given date."}, status_code=404)
        
        predicted = forecast_row['yhat'].values[0]
        if df_global is None or model is None:
            return JSONResponse(content={"error": "Model not initialized."}, status_code=500)
        actual = df_global.loc[current_date, "demand"]
        error = abs(actual - predicted)

        drift_detector.update(error, current_date)
        if drift_detector.should_retrain():
            drift_start_date = current_date
            last_trained_date = get_last_trained_date()
            # Retrain
            retrain_model(drift_start_date, df_global)
            model.load()
            drift_detected = 1
            drift_detector.reset()

        predicted = float(predicted)
        actual = float(actual)
        error = float(error)
        json_response = {
            "date": current_date.strftime("%Y-%m-%d"),
            "predicted": predicted,
            "actual": actual,
            "error": error,
            'drift_detected': drift_detected
        }
        print(json_response)
        return JSONResponse(content=json_response)

    except Exception as e:
        logger.exception("Error during prediction")
        return JSONResponse(content={"error": str(e)}, status_code=500)


