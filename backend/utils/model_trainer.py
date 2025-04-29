import pandas as pd
import pickle
import logging
from pathlib import Path
import json
from prophet import Prophet  # Prophet for forecasting
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import tempfile
from prophet.plot import plot_plotly, plot_components_plotly

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

def train_model(df: pd.DataFrame,custom_params=None):
    DEFAULT_PROPHET_PARAMS = {
    'growth': 'linear',
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'holidays_prior_scale': 10.0,
    'seasonality_mode': 'additive',
    'changepoint_range': 0.8,
    'yearly_seasonality': True,
    'weekly_seasonality': True,
    'daily_seasonality': False,
    'holidays': None,
    'mcmc_samples': 0,
    'interval_width': 0.8,
    'uncertainty_samples': 1000,
    'stan_backend': None
    }

    params = DEFAULT_PROPHET_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
    
    with mlflow.start_run(nested=True):
        # Initialize and fit model
        model = Prophet(**params)
        model.add_seasonality(
            name='monthly',
            period=30,
            fourier_order=5  # controls how flexible the seasonal curve is
        )
        model.fit(df)
        
        # Save model to disk (original behavior)
        MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        
        # MLflow tracking
        mlflow.log_params(params)
        
        # Calculate and log metrics
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        y_true = df['y'].values
        y_pred = forecast.loc[:len(y_true)-1, 'yhat'].values
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        
        mlflow.log_metrics({
            'mae': mae,
            'rmse': rmse
        })
        
        # Log the model.pkl as an artifact
        mlflow.log_artifact(MODEL_PATH, "model")
        
        # Log plots
        with tempfile.TemporaryDirectory() as tmpdir:
            # Forecast plot
            fig1 = model.plot(forecast)
            plot_path = f"{tmpdir}/forecast_plot.png"
            fig1.savefig(plot_path)
            mlflow.log_artifact(plot_path)
            
            # Component plot
            fig2 = model.plot_components(forecast)
            components_path = f"{tmpdir}/components_plot.png"
            fig2.savefig(components_path)
            mlflow.log_artifact(components_path)
            
            # Interactive plots
            interactive_plot = plot_plotly(model, forecast)
            interactive_plot.write_html(f"{tmpdir}/interactive_plot.html")
            mlflow.log_artifact(f"{tmpdir}/interactive_plot.html")
            
            components_plot = plot_components_plotly(model, forecast)
            components_plot.write_html(f"{tmpdir}/interactive_components.html")
            mlflow.log_artifact(f"{tmpdir}/interactive_components.html")
        
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
