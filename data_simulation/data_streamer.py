### data_streamer.py
import time

class DataStreamer:
    def __init__(self, df, start_date, interval_sec=10):
        self.data = df[df.index > start_date]
        self.interval = interval_sec
        self.pointer = 0

    def stream(self):
        while self.pointer < len(self.data):
            row = self.data.iloc[self.pointer]
            self.pointer += 1
            yield row.name, row["demand"]
            time.sleep(self.interval)