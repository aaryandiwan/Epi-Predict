"""
Epi Predict – Model Training Pipeline

Trains all 7 ML models, evaluates them, selects the best,
and saves trained models with metadata.
"""

import json
import time
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config.settings import (
    MODELS_DIR,
    MODEL_REGISTRY_FILE,
    MODEL_PARAMS,
    RANDOM_STATE,
    TARGET_COLUMN,
    FEATURE_COLUMNS,
)
from models.model_definitions import (
    get_model,
    ALL_MODEL_NAMES,
    create_tuned_random_forest,
    create_lstm,
    create_arima,
)

logger = logging.getLogger("epi_predict.trainer")


def evaluate_model(y_true, y_pred, model_name: str = "") -> dict:
    """
    Evaluate model predictions with MAE, RMSE, and R² Score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    metrics = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2), 4),
    }

    logger.info(
        f"{model_name} | MAE={mae:.2f} | RMSE={rmse:.2f} | R²={r2:.4f}"
    )
    return metrics


def save_model(model, name: str, metrics: dict, features: list):
    """
    Save a trained model and its metadata.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"{name}.joblib"

    if name == "lstm":
        try:
            model_path = MODELS_DIR / f"{name}.keras"
            model.save(str(model_path))
        except Exception:
            model_path = MODELS_DIR / f"{name}.h5"
            model.save(str(model_path))
    elif name == "arima":
        model_path = MODELS_DIR / f"{name}.pkl"
        import pickle
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        joblib.dump(model, model_path)

    logger.info(f"Model saved: {model_path}")

    # Update registry
    registry = load_registry()
    registry[name] = {
        "path": str(model_path),
        "metrics": metrics,
        "features": features,
        "timestamp": timestamp,
        "version": registry.get(name, {}).get("version", 0) + 1,
    }
    save_registry(registry)

    return model_path


def load_registry() -> dict:
    """Load model registry from JSON file."""
    if MODEL_REGISTRY_FILE.exists():
        with open(MODEL_REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}


def save_registry(registry: dict):
    """Save model registry to JSON file."""
    with open(MODEL_REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2, default=str)
    logger.info("Model registry updated")


def train_sklearn_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    features: list,
) -> dict:
    """
    Train a scikit-learn compatible model (LR, RF, XGB, Stacking).
    """
    logger.info(f"Training {name}...")
    start_time = time.time()

    if name == "tuned_random_forest":
        model = create_tuned_random_forest(X_train, y_train)
    else:
        model = get_model(name)
        model.fit(X_train, y_train)

    train_time = time.time() - start_time
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, name)
    metrics["train_time_seconds"] = round(train_time, 2)

    model_path = save_model(model, name, metrics, features)

    return {
        "model": model,
        "predictions": y_pred,
        "metrics": metrics,
        "path": str(model_path),
    }


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    features: list,
    scaler_X=None,
    scaler_y=None,
) -> dict:
    """
    Train the LSTM model.
    Expects pre-scaled and reshaped data from prepare_lstm_data().
    """
    from models.feature_engineering import prepare_lstm_data

    logger.info("Training LSTM...")
    start_time = time.time()

    params = MODEL_PARAMS["lstm"]
    input_shape = (X_train.shape[1], X_train.shape[2])

    model = create_lstm(input_shape)

    history = model.fit(
        X_train,
        y_train,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_split=params.get("validation_split", 0.1),
        verbose=0,
    )

    train_time = time.time() - start_time

    # Predict and inverse-transform
    y_pred_scaled = model.predict(X_test, verbose=0).ravel()

    if scaler_y is not None:
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    else:
        y_pred = y_pred_scaled
        y_actual = y_test

    metrics = evaluate_model(y_actual, y_pred, "lstm")
    metrics["train_time_seconds"] = round(train_time, 2)
    metrics["epochs"] = params["epochs"]
    metrics["final_loss"] = float(history.history["loss"][-1])

    model_path = save_model(model, "lstm", metrics, features)

    # Save scalers
    if scaler_X is not None:
        joblib.dump(scaler_X, MODELS_DIR / "lstm_scaler_X.joblib")
    if scaler_y is not None:
        joblib.dump(scaler_y, MODELS_DIR / "lstm_scaler_y.joblib")

    return {
        "model": model,
        "predictions": y_pred,
        "metrics": metrics,
        "path": str(model_path),
        "history": history.history,
    }


def train_arima_model(
    train_series: pd.Series,
    test_series: pd.Series,
    features: list,
) -> dict:
    """
    Train the ARIMA model on time series data.
    """
    logger.info("Training ARIMA...")
    start_time = time.time()

    model_fit = create_arima(train_series=train_series)
    train_time = time.time() - start_time

    # Forecast
    y_pred = model_fit.forecast(steps=len(test_series))
    y_pred = np.maximum(y_pred, 0)  # Predictions can't be negative

    metrics = evaluate_model(test_series.values, y_pred, "arima")
    metrics["train_time_seconds"] = round(train_time, 2)
    metrics["aic"] = round(float(model_fit.aic), 2)

    model_path = save_model(model_fit, "arima", metrics, features)

    return {
        "model": model_fit,
        "predictions": y_pred,
        "metrics": metrics,
        "path": str(model_path),
    }


def train_all_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    features: list,
    train_series: pd.Series = None,
    test_series: pd.Series = None,
    skip_lstm: bool = False,
    skip_arima: bool = False,
) -> dict:
    """
    Train all 7 models and return results.

    Args:
        X_train, X_test: Feature arrays.
        y_train, y_test: Target arrays.
        features: List of feature names.
        train_series: Target time series for ARIMA (training portion).
        test_series: Target time series for ARIMA (test portion).
        skip_lstm: Skip LSTM training (if TensorFlow unavailable).
        skip_arima: Skip ARIMA training.

    Returns:
        Dict mapping model_name → {model, predictions, metrics, path}
    """
    results = {}
    all_metrics = []

    # 1. Sklearn-compatible models
    sklearn_models = [
        "linear_regression",
        "random_forest",
        "tuned_random_forest",
        "xgboost",
        "stacking_ensemble",
    ]

    for name in sklearn_models:
        try:
            result = train_sklearn_model(
                name, X_train, y_train, X_test, y_test, features
            )
            results[name] = result
            all_metrics.append({"model": name, **result["metrics"]})
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            continue

    # 2. LSTM
    if not skip_lstm:
        try:
            from models.feature_engineering import prepare_lstm_data

            X_tr_lstm, X_te_lstm, y_tr_lstm, y_te_lstm, scaler_X, scaler_y = (
                prepare_lstm_data(X_train, X_test, y_train, y_test)
            )
            result = train_lstm_model(
                X_tr_lstm, y_tr_lstm, X_te_lstm, y_te_lstm,
                features, scaler_X, scaler_y
            )
            results["lstm"] = result
            all_metrics.append({"model": "lstm", **result["metrics"]})
        except Exception as e:
            logger.error(f"Failed to train LSTM: {e}")

    # 3. ARIMA
    if not skip_arima and train_series is not None:
        try:
            result = train_arima_model(train_series, test_series, ["target_series"])
            results["arima"] = result
            all_metrics.append({"model": "arima", **result["metrics"]})
        except Exception as e:
            logger.error(f"Failed to train ARIMA: {e}")

    # Summary
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        summary_df = summary_df.sort_values("r2_score", ascending=False)
        logger.info("\n" + "=" * 60)
        logger.info("MODEL COMPARISON SUMMARY")
        logger.info("=" * 60)
        logger.info("\n" + summary_df.to_string(index=False))

        # Identify best model
        best = summary_df.iloc[0]
        logger.info(f"\n🏆 Best Model: {best['model']} (R²={best['r2_score']:.4f})")

        # Update registry with best model flag
        registry = load_registry()
        registry["_best_model"] = best["model"]
        registry["_comparison"] = all_metrics
        save_registry(registry)

    return results
