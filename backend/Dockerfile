# Use an official Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Expose the backend port
EXPOSE 8000

EXPOSE 8001

EXPOSE 5000

EXPOSE 9090

EXPOSE 9100

# Start the backend server
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]