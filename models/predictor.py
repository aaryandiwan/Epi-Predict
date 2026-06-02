"""
Epi Predict – Prediction Engine

Loads trained models and generates predictions with confidence intervals,
future forecasts, and multi-model comparisons.
"""

import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from config.settings import (
    MODELS_DIR,
    MODEL_REGISTRY_FILE,
    FEATURE_COLUMNS,
    FORECAST_WEEKS,
)

logger = logging.getLogger("epi_predict.predictor")


class PredictionEngine:
    """
    Inference engine that loads trained models and generates predictions.
    """

    def __init__(self):
        self.models = {}
        self.registry = {}
        self.best_model_name = None
        self._load_registry()

    def _load_registry(self):
        """Load model registry."""
        if MODEL_REGISTRY_FILE.exists():
            with open(MODEL_REGISTRY_FILE, "r") as f:
                self.registry = json.load(f)
            self.best_model_name = self.registry.get("_best_model")
            logger.info(
                f"Registry loaded: {len(self.registry) - 2} models, "
                f"best={self.best_model_name}"
            )
        else:
            logger.warning("No model registry found. Train models first.")

    def load_model(self, name: str):
        """Load a specific model from disk."""
        if name in self.models:
            return self.models[name]

        if name not in self.registry:
            raise ValueError(
                f"Model '{name}' not in registry. "
                f"Available: {[k for k in self.registry if not k.startswith('_')]}"
            )

        model_path = self.registry[name]["path"]

        if name == "lstm":
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_path)
            except Exception as e:
                logger.error(f"Failed to load LSTM: {e}")
                raise
        elif name == "arima":
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        else:
            model = joblib.load(model_path)

        self.models[name] = model
        logger.info(f"Model loaded: {name} from {model_path}")
        return model

    def load_all_models(self):
        """Load all available models."""
        for name in self.registry:
            if name.startswith("_"):
                continue
            try:
                self.load_model(name)
            except Exception as e:
                logger.warning(f"Could not load {name}: {e}")

    def predict(
        self,
        X: np.ndarray,
        model_name: str = None,
    ) -> np.ndarray:
        """
        Generate predictions using a specific model.

        Args:
            X: Feature array (n_samples, n_features).
            model_name: Model to use. Defaults to best model.

        Returns:
            Predictions array.
        """
        name = model_name or self.best_model_name
        if not name:
            raise ValueError("No model specified and no best model in registry")

        model = self.load_model(name)

        if name == "lstm":
            # Need scalers for LSTM
            scaler_X_path = MODELS_DIR / "lstm_scaler_X.joblib"
            scaler_y_path = MODELS_DIR / "lstm_scaler_y.joblib"

            if scaler_X_path.exists() and scaler_y_path.exists():
                scaler_X = joblib.load(scaler_X_path)
                scaler_y = joblib.load(scaler_y_path)
                X_scaled = scaler_X.transform(X)
                X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
                y_pred_scaled = model.predict(X_lstm, verbose=0).ravel()
                y_pred = scaler_y.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                ).ravel()
            else:
                X_lstm = X.reshape(X.shape[0], 1, X.shape[1])
                y_pred = model.predict(X_lstm, verbose=0).ravel()

        elif name == "arima":
            # ARIMA uses forecast from the fitted model
            y_pred = model.forecast(steps=len(X))
        else:
            y_pred = model.predict(X)

        # Ensure non-negative predictions
        y_pred = np.maximum(y_pred, 0)

        return y_pred

    def predict_with_confidence(
        self,
        X: np.ndarray,
        model_name: str = None,
        confidence_level: float = 0.95,
    ) -> dict:
        """
        Generate predictions with confidence intervals.

        Uses ensemble variation for confidence when multiple models available,
        or bootstrap-style estimation for single models.

        Returns:
            Dict with 'predictions', 'lower_bound', 'upper_bound', 'confidence_level'
        """
        name = model_name or self.best_model_name
        predictions = self.predict(X, name)

        # Estimate uncertainty from multiple models
        all_preds = []
        for model_name_iter in self.registry:
            if model_name_iter.startswith("_"):
                continue
            try:
                pred = self.predict(X, model_name_iter)
                all_preds.append(pred)
            except Exception:
                continue

        if len(all_preds) > 1:
            all_preds_arr = np.array(all_preds)
            std = np.std(all_preds_arr, axis=0)
        else:
            # Fallback: use percentage-based uncertainty
            std = predictions * 0.15

        # z-score for confidence level
        from scipy import stats
        z = stats.norm.ppf((1 + confidence_level) / 2)

        lower = np.maximum(predictions - z * std, 0)
        upper = predictions + z * std

        return {
            "predictions": predictions.tolist(),
            "lower_bound": lower.tolist(),
            "upper_bound": upper.tolist(),
            "confidence_level": confidence_level,
            "model_used": name,
        }

    def forecast_future(
        self,
        df: pd.DataFrame,
        weeks_ahead: int = None,
        model_name: str = None,
        features: list = None,
    ) -> dict:
        """
        Generate multi-week future forecast.

        Args:
            df: Historical data with features.
            weeks_ahead: Number of weeks to forecast.
            model_name: Model to use.
            features: Feature columns to use.

        Returns:
            Dict with forecast data including dates, predictions, confidence intervals.
        """
        from models.feature_engineering import generate_future_features

        weeks_ahead = weeks_ahead or FORECAST_WEEKS
        features = features or FEATURE_COLUMNS
        name = model_name or self.best_model_name

        future_df = generate_future_features(df, weeks_ahead, features)

        # Get feature matrix
        available_features = [f for f in features if f in future_df.columns]
        X_future = future_df[available_features].values

        # Iterative prediction (each prediction updates lags for next)
        predictions = []
        conf_lower = []
        conf_upper = []

        for i in range(weeks_ahead):
            X_row = X_future[i:i+1]
            result = self.predict_with_confidence(X_row, name)

            pred = result["predictions"][0]
            predictions.append(pred)
            conf_lower.append(result["lower_bound"][0])
            conf_upper.append(result["upper_bound"][0])

            # Update future features with this prediction
            if i + 1 < weeks_ahead:
                # Update lag_1 for next row
                if "lag_1" in available_features:
                    lag1_idx = available_features.index("lag_1")
                    X_future[i + 1, lag1_idx] = pred
                # Update lag_2
                if "lag_2" in available_features and i + 2 < weeks_ahead:
                    lag2_idx = available_features.index("lag_2")
                    if i + 1 < weeks_ahead:
                        X_future[i + 1, lag2_idx] = X_future[i, lag1_idx] if "lag_1" in available_features else pred

        # Build forecast DataFrame
        forecast_data = {
            "week_number": list(range(1, weeks_ahead + 1)),
            "iso_year": future_df["ISO_YEAR"].tolist(),
            "iso_week": future_df["ISO_WEEK"].tolist(),
            "predicted_cases": predictions,
            "lower_bound": conf_lower,
            "upper_bound": conf_upper,
        }

        return {
            "forecast": forecast_data,
            "model_used": name,
            "weeks_ahead": weeks_ahead,
            "generated_at": pd.Timestamp.now().isoformat(),
        }

    def compare_models(
        self, X: np.ndarray, y_actual: np.ndarray = None
    ) -> dict:
        """
        Compare predictions from all available models.

        Returns:
            Dict with per-model predictions and optional metrics.
        """
        comparison = {}

        for name in self.registry:
            if name.startswith("_"):
                continue
            try:
                preds = self.predict(X, name)
                entry = {
                    "predictions": preds.tolist(),
                    "mean_prediction": float(np.mean(preds)),
                }

                if y_actual is not None:
                    from sklearn.metrics import mean_absolute_error, r2_score
                    entry["mae"] = float(mean_absolute_error(y_actual, preds))
                    entry["rmse"] = float(
                        np.sqrt(mean_absolute_error(y_actual, preds))
                    )
                    entry["r2_score"] = float(r2_score(y_actual, preds))

                # Include stored metrics from training
                if name in self.registry:
                    entry["training_metrics"] = self.registry[name].get(
                        "metrics", {}
                    )

                comparison[name] = entry
            except Exception as e:
                logger.warning(f"Could not compare {name}: {e}")
                comparison[name] = {"error": str(e)}

        comparison["_best_model"] = self.best_model_name
        return comparison

    def get_model_info(self, name: str = None) -> dict:
        """Get model information from registry."""
        if name:
            return self.registry.get(name, {})
        return {
            k: v for k, v in self.registry.items()
            if not k.startswith("_")
        }

    def get_all_metrics(self) -> list:
        """Get training metrics for all models."""
        return self.registry.get("_comparison", [])
