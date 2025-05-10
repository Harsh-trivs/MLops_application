# AI Application with Backend and Frontend + Monitoring

This project provides a complete setup for running an MLflow server, a backend service, and a frontend client using Docker Compose. It also integrates **Prometheus** and **Grafana** for monitoring, enabling seamless model tracking, training orchestration, user interaction, and real-time system observability.

---

## 🚀 Services Overview

| Service | Description |
| --- | --- |
| **mlflow** | MLflow Tracking Server to log experiments, metrics, parameters, and artifacts. |
| **server** | Custom backend API handling training, logging, forecasting, or model serving logic. |
| **client** | Frontend client (e.g., Streamlit or React app) for interaction and visualization. |
| **prometheus** | Time-series metrics monitoring tool scraping data from backend and client services. |
| **grafana** | Visual dashboard for Prometheus metrics, accessible via a browser. |

---

## 🐳 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Harsh-trivs/MLops_application.git
cd MLops_application

```

### 2. Build and run all services

```bash
docker-compose up --build

```

This will spin up:

- **MLflow server** at [http://localhost:5001](http://localhost:5001/)
- **Backend API** at [http://localhost:8000](http://localhost:8000/)
- **Client frontend** at [http://localhost:8502](http://localhost:8502/)

Additional, configure prometheus using yaml file present in monitoring directory and connect it with grafana. Use the dashboard.json file to replicate intended monitoring dashboard.

- **Prometheus** at [http://localhost:9090](http://localhost:9090/)
- **Grafana** at [http://localhost:3000](http://localhost:3000/) *(login: `admin` / `admin`)*

---

## 🗃️ Folder Structure

```
.
├── docker-compose.yml
├── backend/             # Backend API source code and Dockerfile
├── client/              # Frontend app source code and Dockerfile
├── monitoring/
│   ├── prometheus.yml   # Prometheus scrape configuration
│   └── dashboard.json     

```

---

## 📦 MLflow Configuration

- **Backend Store**: SQLite database stored in `mlflow-data` volume (`/mlflow/mlflow.db`)
- **Artifacts**: Stored in `/mlflow/artifacts`
- **UI Port**: Accessible at [http://localhost:5001](http://localhost:5001/)

### Environment Variables (mlflow service):

```
MLFLOW_HOST=0.0.0.0
MLFLOW_PORT=5000
MLFLOW_BACKEND_STORE_URI=sqlite:////mlflow/mlflow.db

```

---

## 📊 Monitoring Setup (Prometheus + Grafana)

### Prometheus

- Configured to scrape metrics from:
    - `server` at `http://server:8000/metrics`
    - `client` at `http://client:8502/metrics` (if supported)
- Config file: `monitoring/prometheus.yml`

### Grafana

- Comes preconfigured with:
    - Prometheus data source (localhost:9090)
    - Default dashboards can be added under `monitoring/grafana/provisioning/`

### Login Info

- **Username**: `admin`
- **Password**: `admin` (change after first login)

---

## 📂 Volumes

- **mlflow-data** (named volume)
    - Stores:
        - MLflow SQLite DB (`mlflow.db`)
        - Artifact files (`artifacts/`)
- **grafana-data**
    - Persists Grafana configuration and dashboards

---

## 🔁 Restart Policy

All services use:

```yaml
restart: unless-stopped

```

To automatically recover after crashes or system reboots.

---

## 📡 Networking

All containers share the same Docker network: `app-network`, allowing seamless inter-service communication.

---

## ✅ Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/)

---

## 🧪 Example Use Cases

- Run ML experiments and log to MLflow
- Visualize metrics in MLflow UI
- Interact with backend and frontend services
- Monitor API & app performance in real time via Grafana dashboards
- Inspect metrics directly in Prometheus

---

## 🧼 Stopping the Application

```bash
docker-compose down

```

To remove all containers but retain volumes:

```bash
docker-compose down --volumes

```
