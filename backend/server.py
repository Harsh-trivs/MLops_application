from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from functools import wraps
import pandas as pd
import io
import json
import logging
import time
import psutil
import pickle
import mlflow
import tempfile
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from prometheus_client import start_http_server, Gauge, Counter, Histogram, ProcessCollector
from utils.drift_detector import DriftDetector
from utils.model_trainer import save_metadata, save_model, train_model
from utils.model_wrapper import ForecastingModel
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

# FastAPI app
app = FastAPI()

# Config
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
SEASONAL_PERIODS = 30
PROMETHEUS_PORT = 8001  # Port for Prometheus metrics

# MLflow Config
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MLFLOW_EXPERIMENT_NAME = "DemandForecasting"

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ======================
# Prometheus Metrics
# ======================

# Endpoint-specific metrics
MODEL_INIT_REQUESTS = Counter(
    'model_init_requests_total',
    'Total requests to /model_init',
    ['method', 'status_code']
)
MODEL_INIT_LATENCY = Histogram(
    'model_init_request_latency_seconds',
    'Latency of /model_init requests',
    ['method']
)

LAST_TRAINED_DATE_REQUESTS = Counter(
    'last_trained_date_requests_total',
    'Total requests to /last_trained_date',
    ['method', 'status_code']
)
LAST_TRAINED_DATE_LATENCY = Histogram(
    'last_trained_date_request_latency_seconds',
    'Latency of /last_trained_date requests',
    ['method']
)

PREDICT_FOR_DATE_REQUESTS = Counter(
    'predict_for_date_requests_total',
    'Total requests to /predict_for_date',
    ['method', 'status_code']
)
PREDICT_FOR_DATE_LATENCY = Histogram(
    'predict_for_date_request_latency_seconds',
    'Latency of /predict_for_date requests',
    ['method']
)

REQUESTS_PER_IP = Counter(
    'api_requests_per_ip_total',
    'Total requests per client IP',
    ['client_ip', 'endpoint', 'method']
)

# System metrics
CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percent')
MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percent')
DISK_USAGE = Gauge('system_disk_usage_percent', 'System disk usage percent')
NETWORK_BYTES_SENT = Gauge('system_network_bytes_sent', 'Network bytes sent')
NETWORK_BYTES_RECV = Gauge('system_network_bytes_recv', 'Network bytes received')
FILE_HANDLES = Gauge('system_file_handles', 'Number of open file handles')

# Global state placeholders
model: Optional[ForecastingModel] = None
df_global: Optional[pd.DataFrame] = None
drift_detector: Optional[DriftDetector] = None

# ======================
# Middleware
# ======================

class EndpointMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        method = request.method
        path = request.url.path
        client_ip = request.client.host or "unknown"
        
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise e
        finally:
            duration = time.time() - start_time
            
            # Update metrics based on endpoint
            if path == "/model_init":
                MODEL_INIT_REQUESTS.labels(method=method, status_code=status_code).inc()
                MODEL_INIT_LATENCY.labels(method=method).observe(duration)
            elif path == "/last_trained_date":
                LAST_TRAINED_DATE_REQUESTS.labels(method=method, status_code=status_code).inc()
                LAST_TRAINED_DATE_LATENCY.labels(method=method).observe(duration)
            elif path == "/predict_for_date":
                PREDICT_FOR_DATE_REQUESTS.labels(method=method, status_code=status_code).inc()
                PREDICT_FOR_DATE_LATENCY.labels(method=method).observe(duration)
            
            REQUESTS_PER_IP.labels(
                client_ip=client_ip,
                endpoint=request.url.path,
                method=request.method
            ).inc()
        
        return response

app.add_middleware(EndpointMetricsMiddleware)

# ======================
# System Metrics Collector
# ======================

def update_system_metrics():
    while True:
        try:
            CPU_USAGE.set(psutil.cpu_percent())
            MEMORY_USAGE.set(psutil.virtual_memory().percent)
            DISK_USAGE.set(psutil.disk_usage('/').percent)
            net_io = psutil.net_io_counters()
            NETWORK_BYTES_SENT.set(net_io.bytes_sent)
            NETWORK_BYTES_RECV.set(net_io.bytes_recv)
            FILE_HANDLES.set(len(psutil.Process().open_files()))
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        time.sleep(5)

# ======================
# Utility Functions
# ======================

def train_initial_model(df):
    initial_data = df.head(6 * 30)  # First 6 months
    logger.info("Training initial model...")
    model = train_model(initial_data.reset_index().rename(columns={"date": "ds", "demand": "y"}))
    save_metadata(initial_data.reset_index().rename(columns={"date": "ds", "demand": "y"}))
    logger.info("Initial model trained and saved.")

def retrain_model(start_date, df,drift_metadata=None):
    logger.info(f"🔁 Retraining model from {start_date.date()} onwards...")
    df_subset = df.loc[:start_date]
    
    tags = {
        "retraining": "true",
        "retraining_trigger": "drift_detection" if drift_metadata else "periodic"
    }
    
    if drift_metadata:
        tags.update({
            "drift_score": str(drift_metadata.get('score')),
            "drift_threshold": str(drift_metadata.get('threshold')),
            "drift_window_size": str(drift_metadata.get('window_size'))
        })
    
    with mlflow.start_run(tags=tags):
        model = train_model(
            df_subset.reset_index().rename(columns={"date": "ds", "demand": "y"})
        )
        save_metadata(df_subset.reset_index().rename(columns={"date": "ds", "demand": "y"}))
        
        if drift_metadata:
            mlflow.log_params({
                f"drift_{k}": v for k, v in drift_metadata.items()
            })
        
        logger.info("✅ Model retrained.")  

def get_last_trained_date():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            meta = json.load(f)
            return pd.to_datetime(meta.get("last_trained_date"))
    return None

# ======================
# API Endpoints
# ======================

@app.post("/model_init")
async def model_init(file: UploadFile = File(...), window_size: int = Form(...), threshold: float = Form(...)):
    global model, df_global, drift_detector
    try: 
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

        # Drift detection
        drift_detector.update(error, current_date)
        if drift_detector.should_retrain():
            drift_start_date = current_date
            drift_metadata = {
                'threshold': drift_detector.threshold,
                'window_size': drift_detector.window_size,
                'trigger_date': str(current_date)
            }
            
            # Log drift detection
            with mlflow.start_run(nested=True):
                mlflow.log_params(drift_metadata)
            
            retrain_model(drift_start_date, df_global, drift_metadata)
            model.load()
            drift_detected = 1
            drift_detector.reset()

        json_response = {
            "date": current_date.strftime("%Y-%m-%d"),
            "predicted": float(predicted),
            "actual": float(actual),
            "error": float(error),
            "drift_detected": drift_detected
        }
        
        return JSONResponse(content=json_response)

    except Exception as e:
        logger.exception("Error during prediction")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ======================
# Startup Event
# ======================

@app.on_event("startup")
async def startup_event():
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Start Prometheus metrics server
    start_http_server(PROMETHEUS_PORT)
    logger.info(f"Prometheus metrics server started on port {PROMETHEUS_PORT}")
    
    # Start system metrics collection
    import threading
    threading.Thread(target=update_system_metrics, daemon=True).start()
    
    # Add process metrics collector
    ProcessCollector(namespace='forecast_app')
    
    # Load model if exists
    global model
    if MODEL_PATH.exists():
        model = ForecastingModel()
        model.load()
        logger.info("Loaded existing model")