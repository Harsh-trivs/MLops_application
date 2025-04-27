import numpy as np
import pandas as pd

def simulate_single_sku_demand(start_date="2025-01-01", days=730, seed=42):
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, periods=days, freq="D")

    # Base level and trend
    base_level = 120
    trend = 0.05 * np.arange(days)

    # Dual Seasonality
    weekly_seasonality = 8 * np.sin(2 * np.pi * np.arange(days) / 7)
    monthly_seasonality = 20 * np.sin(2 * np.pi * np.arange(days) / 30)
    seasonality_strength = 1 + 0.2 * np.sin(2 * np.pi * np.arange(days) / 365)
    seasonality = seasonality_strength * (weekly_seasonality + monthly_seasonality)

    # Weekday effect
    weekday_effect = np.array([1.15 if d.weekday() < 5 else 0.9 for d in date_range])

    # Base noise
    noise = np.random.normal(0, 10, size=days)

    # Base demand
    demand = base_level + trend + seasonality
    demand *= weekday_effect
    demand += noise

    ### Injecting Structured Drifts:

    # 1. Level shift up (Day 150)
    drift1_start = 150
    demand[drift1_start:] += 50

    # 2. Level shift down (Day 300)
    drift2_start = 300
    demand[drift2_start:] -= 40

    # 3. Variance spike (Day 450 onward)
    drift3_start = 450
    high_noise = np.random.normal(0, 25, size=days - drift3_start)
    demand[drift3_start:] += high_noise

    # 4. Sudden outlier (Day 550)
    outlier_day = 550
    demand[outlier_day] += 400  # big spike

    # 5. Gradual drift (Day 600 to 700)
    gradual_drift_start = 600
    gradual_drift_days = 100
    demand[gradual_drift_start:gradual_drift_start+gradual_drift_days] += np.linspace(0, 30, gradual_drift_days)

    ### Clean up:

    # Clip negative demand (optional)
    demand = np.clip(demand, 0, None)

    # Build final DataFrame
    df = pd.DataFrame({
        "date": date_range,
        "demand": np.round(demand, 0)
    })
    df.set_index("date", inplace=True)
    df = df.asfreq("D")

    return df
