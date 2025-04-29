from test_simulator import simulate_single_sku_demand
from data_streamer import DataStreamer
import pandas as pd
import os
import logging

# Step 1: Simulate the full dataset
df = simulate_single_sku_demand()
csv_path = "data_simulation/data.csv"

# Step 2: Detect last date if CSV exists, else initialize
def get_last_date_from_csv(path):
    
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            existing_df = pd.read_csv(path, parse_dates=['date'], index_col='date')
            return existing_df.index[-1]
        except Exception as e:
            print("⚠️ Could not read last date from CSV:", e)
    return pd.Timestamp("2025-06-01")

start_cutoff = get_last_date_from_csv(csv_path)

# If file doesn't exist or is empty, prefill until cutoff
if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
    initial_df = df[df.index <= start_cutoff]
    initial_df.to_csv(csv_path)

# Step 3: Stream remaining data every 5 seconds
streamer = DataStreamer(df, start_date=start_cutoff, interval_sec=1)
logging.basicConfig(level=logging.INFO)

for date, demand in streamer.stream():
    logging.info(f"📦 Simulated Day: {date.date()} | Demand: {demand}")
    new_row = pd.DataFrame({"date": [date], "demand": [demand]})
    new_row.to_csv(csv_path, mode='a', header=False, index=False)