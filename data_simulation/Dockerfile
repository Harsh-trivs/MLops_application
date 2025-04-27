# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install pandas numpy

# Default command
CMD ["python", "run_streamer.py"]
