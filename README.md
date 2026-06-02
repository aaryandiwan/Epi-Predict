# Epi Predict – Influenza Outbreak Early Warning System

A production-grade, end-to-end AI healthcare application designed to forecast seasonal influenza outbreaks globally. Built using World Health Organization (WHO) FluNet surveillance data.

## 🌟 Key Features

* **AI Prediction Engine:** Ensemble of 7 machine learning models (XGBoost, LSTM, Random Forest, ARIMA, etc.) automatically evaluating and selecting the best model.
* **Live Data Pipeline:** Automated ingestion and preprocessing of global WHO influenza surveillance data.
* **Executive Dashboard:** Stunning, dark-themed glassmorphism Streamlit UI with interactive Plotly charts.
* **Explainable AI (XAI):** Built-in SHAP (SHapley Additive exPlanations) integration to understand model feature importance.
* **Outbreak Alert System:** Dynamic risk classification and early warning generation.
* **MLOps Pipeline:** Automated retraining, model registry, and performance monitoring.

## 🌐 Live Demo
The application can be deployed for free using Streamlit Community Cloud.
Once deployed, the live link will be available here: **[[Link Live App to](https://epi-predict-m4lwjctpndijkpvuq3q7wy.streamlit.app/)]**

## 🚀 Quick Start

### 1. Installation
Clone the repository and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration
Copy the environment template:
```bash
cp .env.example .env
```

### 3. Model Training
Before starting the application, fetch data and train the AI models:
```bash
python scripts/train_models.py --country India
```
*(Use `--help` for additional training options)*

### 4. Running the Application

**Start the FastAPI Backend:**
```bash
uvicorn api.main:app --reload --port 8000
```
API Documentation available at: `http://localhost:8000/docs`

**Start the Streamlit Dashboard (In a new terminal):**
```bash
streamlit run dashboard/app.py
```
Dashboard available at: `http://localhost:8501`

## 🐳 Docker Deployment

To run the entire stack (API + Dashboard) using Docker Compose:
```bash
docker-compose up --build -d
```

## 🏗️ Architecture

```
epi_predict/
├── api/             # FastAPI backend (Routes, Schemas)
├── config/          # Centralized configuration
├── dashboard/       # Streamlit frontend UI
├── data/            # WHO data loaders and preprocessing
├── docs/            # Project documentation
├── mlops/           # Model logging, monitoring, and versioning
├── models/          # ML definitions, training, and inference
├── modules/         # Business logic (Alerts, Explainability, Risk)
└── scripts/         # CLI execution scripts
```

## 📊 Models Included
* Linear Regression
* Random Forest Regressor
* Tuned Random Forest (GridSearch)
* XGBoost Regressor
* LSTM (Deep Learning)
* ARIMA (Time Series)
* Stacking Ensemble

---
*Built as a comprehensive AI Engineering portfolio project.*
