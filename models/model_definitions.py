"""
Epi Predict – Model Definitions

Factory functions for all 7 ML model architectures with exact
hyperparameters from the original notebooks.
"""

import logging
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from config.settings import MODEL_PARAMS, RANDOM_STATE

logger = logging.getLogger("epi_predict.models")


def create_linear_regression():
    """Create a Linear Regression model."""
    logger.info("Creating Linear Regression model")
    return LinearRegression()


def create_random_forest():
    """Create a Random Forest model with notebook parameters."""
    params = MODEL_PARAMS["random_forest"]
    logger.info(f"Creating Random Forest: {params}")
    return RandomForestRegressor(**params)


def create_tuned_random_forest(X_train=None, y_train=None):
    """
    Create a hyperparameter-tuned Random Forest using RandomizedSearchCV
    with TimeSeriesSplit.

    If X_train and y_train are provided, performs the search immediately.
    Otherwise returns the base estimator.
    """
    params = MODEL_PARAMS["tuned_random_forest"]

    base_rf = RandomForestRegressor(random_state=RANDOM_STATE)
    tscv = TimeSeriesSplit(n_splits=params["cv_splits"])

    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=params["param_distributions"],
        n_iter=params["n_iter"],
        cv=tscv,
        scoring=params["scoring"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    if X_train is not None and y_train is not None:
        logger.info("Running RandomizedSearchCV for Tuned RF...")
        search.fit(X_train, y_train)
        logger.info(f"Best RF params: {search.best_params_}")
        logger.info(f"Best RF score: {search.best_score_:.4f}")
        return search.best_estimator_

    logger.info("Creating Tuned RF search object (not yet fitted)")
    return search


def create_xgboost():
    """Create an XGBoost model with notebook parameters."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("XGBoost not installed. Run: pip install xgboost")

    params = MODEL_PARAMS["xgboost"]
    logger.info(f"Creating XGBoost: {params}")
    return XGBRegressor(**params)


def create_lstm(input_shape):
    """
    Create an LSTM model with Keras.

    Args:
        input_shape: Tuple of (timesteps, features) for the LSTM input.

    Returns:
        Compiled Keras Sequential model.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        # Suppress TF warnings for cleaner output
        tf.get_logger().setLevel("ERROR")
    except ImportError:
        raise ImportError(
            "TensorFlow not installed. Run: pip install tensorflow"
        )

    params = MODEL_PARAMS["lstm"]

    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(1),
    ])

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

    logger.info(f"Created LSTM model: input_shape={input_shape}")
    model.summary(print_fn=lambda x: logger.debug(x))

    return model


def create_arima(train_series=None):
    """
    Create and fit an ARIMA model.

    Args:
        train_series: Pandas Series of training data for the target variable.

    Returns:
        Fitted ARIMA model results, or ARIMA order tuple if no data provided.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        raise ImportError(
            "statsmodels not installed. Run: pip install statsmodels"
        )

    order = MODEL_PARAMS["arima"]["order"]

    if train_series is not None:
        logger.info(f"Fitting ARIMA{order} on {len(train_series)} observations...")
        model = ARIMA(train_series, order=order)
        result = model.fit()
        logger.info(f"ARIMA fitted: AIC={result.aic:.2f}")
        return result

    logger.info(f"ARIMA order configured: {order}")
    return order


def create_stacking_ensemble():
    """
    Create a Stacking Ensemble with Ridge meta-learner.

    Base models: LinearRegression, RandomForest, XGBoost
    Meta-learner: Ridge(alpha=1.0)
    """
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError("XGBoost not installed for stacking ensemble")

    params = MODEL_PARAMS["stacking_ensemble"]

    base_estimators = [
        ("lr", LinearRegression()),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=params["base_rf_n_estimators"],
                max_depth=params["base_rf_max_depth"],
                random_state=RANDOM_STATE,
            ),
        ),
        (
            "xgb",
            XGBRegressor(
                n_estimators=params["base_xgb_n_estimators"],
                max_depth=params["base_xgb_max_depth"],
                learning_rate=params["base_xgb_learning_rate"],
                random_state=RANDOM_STATE,
            ),
        ),
    ]

    meta_learner = Ridge(alpha=params["meta_learner_alpha"])

    stacking = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=TimeSeriesSplit(n_splits=params["cv_splits"]),
        n_jobs=-1,
    )

    logger.info("Created Stacking Ensemble: LR + RF + XGB → Ridge")
    return stacking


# ─── Model Registry ─────────────────────────────────────────────────────────

MODEL_FACTORY = {
    "linear_regression": create_linear_regression,
    "random_forest": create_random_forest,
    "xgboost": create_xgboost,
    "stacking_ensemble": create_stacking_ensemble,
}

# Models requiring special handling
SPECIAL_MODELS = ["tuned_random_forest", "lstm", "arima"]

ALL_MODEL_NAMES = list(MODEL_FACTORY.keys()) + SPECIAL_MODELS


def get_model(name: str, **kwargs):
    """
    Get a model instance by name.

    Args:
        name: Model name from ALL_MODEL_NAMES.
        **kwargs: Additional arguments passed to the factory.

    Returns:
        Model instance.
    """
    if name in MODEL_FACTORY:
        return MODEL_FACTORY[name]()
    elif name == "tuned_random_forest":
        return create_tuned_random_forest(**kwargs)
    elif name == "lstm":
        input_shape = kwargs.get("input_shape", (1, 6))
        return create_lstm(input_shape)
    elif name == "arima":
        return create_arima(**kwargs)
    else:
        raise ValueError(
            f"Unknown model: '{name}'. Available: {ALL_MODEL_NAMES}"
        )
