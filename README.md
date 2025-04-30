# MLflow Application with Backend and Frontend

This project provides a complete setup for running an **MLflow server**, a **backend service**, and a **client frontend** using **Docker Compose**. It enables seamless model tracking, training orchestration, and user interaction through a web interface.

---

## 🚀 Services Overview

| Service | Description |
|--------|-------------|
| **mlflow** | MLflow Tracking Server to log experiments, metrics, parameters, and artifacts. |
| **server** | Custom backend API handling training, logging, forecasting, or model serving logic. |
| **client** | Frontend client (e.g., Streamlit or React app) for interaction and visualization. |

---

## 🐳 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Build and run the containers

```bash
docker-compose up --build
```

This will spin up the following services:
- **MLflow server** at `http://localhost:5001`
- **Backend API** at `http://localhost:8000`
- **Client frontend** at `http://localhost:8502`

---

## 🗃️ Folder Structure

```
.
├── docker-compose.yml
├── backend/            # Backend API source code and Dockerfile
├── client/             # Frontend app source code and Dockerfile
└── monitoring/        # prometheus configuration
```

---

## 📦 MLflow Configuration

- **Backend Store**: SQLite database stored in the `mlflow-data` volume (`/mlflow/mlflow.db`)
- **Artifacts**: Saved in `/mlflow/artifacts` directory inside the volume
- **Port**: MLflow UI accessible at `http://localhost:5001`

---

## 🔧 Environment Variables

Inside the `mlflow` service:
```env
MLFLOW_HOST=0.0.0.0
MLFLOW_PORT=5000
MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db
```

---

## 📂 Volumes

A named volume `mlflow-data` is used to persist:
- MLflow database (`mlflow.db`)
- Artifact storage (`artifacts/`)

This ensures data is preserved even when containers are stopped or rebuilt.

---

## 🔁 Restart Policy

All services use `restart: unless-stopped` to auto-restart unless explicitly stopped.

---

## 📡 Networking

All services are on a shared Docker network named `app-network` to allow inter-service communication.

---

## 🧪 Example Use Cases

- Run ML experiments and log results to MLflow
- Visualize experiments and metrics via MLflow UI
- Interact with training/inference workflows via the client UI
- Monitor backend and client services from local ports

---

## ✅ Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

---

## 🧼 Stopping the Application

```bash
docker-compose down
```

---
