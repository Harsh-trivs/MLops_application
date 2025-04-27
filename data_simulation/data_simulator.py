import numpy as np
import pandas as pd

def simulate_single_sku_demand(start_date="2025-01-01", days=730, seed=42):
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, periods=days)

    # Base level and trend
    base_level = 120
    trend = 0.08 * np.arange(days)

    # Dual Seasonality
    weekly_seasonality = 8 * np.sin(2 * np.pi * np.arange(days) / 7)
    monthly_seasonality = 25 * np.sin(2 * np.pi * np.arange(days) / 30)
    seasonality_strength = 1 + 0.3 * np.sin(2 * np.pi * np.arange(days) / 365)  # Yearly modulation
    seasonality = seasonality_strength * (weekly_seasonality + monthly_seasonality)

    # Weekday effect (higher on weekdays)
    weekday_effect = np.array([1.1 if d.weekday() < 5 else 0.85 for d in date_range])

    # Random noise
    noise = np.random.normal(0, 12, size=days)

    # Build demand
    demand = base_level + trend + seasonality
    demand *= weekday_effect
    demand += noise

    # Holiday spikes
    holidays = pd.to_datetime(["2025-01-26", "2025-08-15", "2025-10-24", "2025-12-25"])
    for h in holidays:
        if h in date_range:
            idx = (date_range == h)
            demand[idx] += np.random.randint(80, 160)

    # Random promotions (5% of days boosted)
    promo_days = np.random.choice(days, size=int(0.05 * days), replace=False)
    demand[promo_days] *= np.random.uniform(1.2, 1.5, size=len(promo_days))

    # Sudden shocks (permanent level shifts)
    shock_days = np.random.choice(days, size=3, replace=False)
    for day in shock_days:
        shift = np.random.uniform(-20, 30)
        demand[day:] += shift

    # Rare extreme outliers
    outlier_days = np.random.choice(days, size=5, replace=False)
    demand[outlier_days] *= np.random.choice([0.5, 2, 3], size=5)

    # Missing data
    missing_days = np.random.choice(days, size=10, replace=False)
    demand[missing_days] = np.nan

    # Gradual drift (tiny daily accumulation)
    gradual_drift = np.cumsum(np.random.normal(0, 0.02, size=days))
    demand += gradual_drift

    # Build final DataFrame
    df = pd.DataFrame({
        "date": date_range,
        "demand": np.round(demand, 0)
    })
    df["demand"] = df["demand"].interpolate(method="linear")  # fill missing
    df.set_index("date", inplace=True)
    df = df.asfreq("D")
    return df
