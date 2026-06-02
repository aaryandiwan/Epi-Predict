# Epi Predict - Deployment Guide

## Architecture Overview

Epi Predict is designed as a microservices architecture:
1. **Data Layer**: SQLite (for prediction logs) and local filesystem (for WHO data cache and model registry).
2. **Backend API**: FastAPI application serving predictions, model information, and business logic.
3. **Frontend UI**: Streamlit application providing the executive dashboard.

## Deployment Options

### Option 1: Docker Compose (Recommended)

The project includes a `Dockerfile` and `docker-compose.yml` for simplified deployment. The Dockerfile uses `supervisord` to run both the FastAPI backend and Streamlit dashboard in a single container for ease of use, though they can be separated into distinct containers if desired.

1. Install Docker and Docker Compose on your host machine.
2. Clone the repository.
3. Create your `.env` file based on `.env.example`.
4. Run: `docker-compose up -d --build`
5. The API will be available on port `8000` and the Dashboard on `8501`.

### Option 2: Cloud Platform (AWS / GCP / Azure)

#### VM Deployment (EC2 / Compute Engine)
1. Provision a VM (minimum 4GB RAM recommended for model training).
2. SSH into the instance and install Python 3.11.
3. Clone the repo and install `requirements.txt`.
4. Use `tmux` or `systemd` to run the services.

**Example systemd service for FastAPI (`/etc/systemd/system/epipredict-api.service`):**
```ini
[Unit]
Description=Epi Predict FastAPI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/epi_predict
ExecStart=/home/ubuntu/epi_predict/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

**Example systemd service for Streamlit (`/etc/systemd/system/epipredict-ui.service`):**
```ini
[Unit]
Description=Epi Predict Streamlit Dashboard
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/epi_predict
ExecStart=/home/ubuntu/epi_predict/venv/bin/streamlit run dashboard/app.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

## Production Considerations

1. **Model Retraining**: Set up a cron job to run `scripts/train_models.py` weekly to fetch new WHO data and retrain the models.
   ```bash
   0 2 * * 0 cd /path/to/epi_predict && /path/to/venv/bin/python scripts/train_models.py >> logs/cron.log 2>&1
   ```
2. **Security**: Ensure the FastAPI endpoints are secured if exposed publicly (e.g., via Nginx reverse proxy with SSL and basic authentication).
3. **Scale**: For heavy load, replace the SQLite prediction logger with PostgreSQL and use Redis for caching.
