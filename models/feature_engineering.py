"""
Epi Predict – Feature Engineering Pipeline

Extracts and transforms features from cleaned WHO FluNet data,
matching the exact pipeline from the original notebooks.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import joblib
from pathlib import Path

from config.settings import (
    TARGET_COLUMN,
    FEATURE_COLUMNS,
    ALL_FEATURES,
    TRAIN_TEST_SPLIT,
    MODELS_DIR,
)

logger = logging.getLogger("epi_predict.features")


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the prediction target: Target = INF_A + INF_B
    """
    df = df.copy()

    if "INF_A" in df.columns and "INF_B" in df.columns:
        df[TARGET_COLUMN] = df["INF_A"].fillna(0) + df["INF_B"].fillna(0)
    elif "INF_ALL" in df.columns:
        # Fallback: use INF_ALL if INF_A/INF_B not available
        df[TARGET_COLUMN] = df["INF_ALL"].fillna(0)
    elif "ALL_INF" in df.columns:
        df[TARGET_COLUMN] = df["ALL_INF"].fillna(0)
    else:
        raise ValueError(
            "Cannot create target: columns INF_A, INF_B, INF_ALL, or ALL_INF not found"
        )

    logger.info(f"Target column created: mean={df[TARGET_COLUMN].mean():.2f}")
    return df


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features: Month from ISO_WEEK.
    """
    df = df.copy()

    if "ISO_WEEK" in df.columns:
        df["Month"] = ((df["ISO_WEEK"] - 1) // 4) + 1
    else:
        logger.warning("ISO_WEEK not found, cannot create Month feature")

    return df


def create_lag_features(
    df: pd.DataFrame,
    group_col: str = None,
    lags: list = None,
) -> pd.DataFrame:
    """
    Create lag features for the target variable.
    Computed per-country if group_col is provided.

    Args:
        df: DataFrame with Target column.
        group_col: Column to group by (e.g., 'COUNTRY_AREA_TERRITORY').
        lags: List of lag periods. Defaults to [1, 2, 3].
    """
    df = df.copy()
    lags = lags or [1, 2, 3]

    if TARGET_COLUMN not in df.columns:
        logger.warning(f"Target column '{TARGET_COLUMN}' not found for lag features")
        return df

    for lag in lags:
        col_name = f"lag_{lag}"
        if group_col and group_col in df.columns:
            df[col_name] = df.groupby(group_col)[TARGET_COLUMN].shift(lag)
        else:
            df[col_name] = df[TARGET_COLUMN].shift(lag)

    # Fill NaN from shifting
    for lag in lags:
        df[f"lag_{lag}"].fillna(0, inplace=True)

    logger.info(f"Created lag features: {[f'lag_{l}' for l in lags]}")
    return df


def create_rolling_features(
    df: pd.DataFrame,
    group_col: str = None,
    windows: list = None,
) -> pd.DataFrame:
    """
    Create rolling mean features for the target variable.
    Computed per-country if group_col is provided.

    Args:
        df: DataFrame with Target column.
        group_col: Column to group by.
        windows: List of rolling window sizes. Defaults to [3, 5].
    """
    df = df.copy()
    windows = windows or [3, 5]

    if TARGET_COLUMN not in df.columns:
        logger.warning(f"Target column '{TARGET_COLUMN}' not found for rolling features")
        return df

    for w in windows:
        col_name = f"roll_{w}"
        if group_col and group_col in df.columns:
            df[col_name] = df.groupby(group_col)[TARGET_COLUMN].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
            )
        else:
            df[col_name] = (
                df[TARGET_COLUMN].shift(1).rolling(window=w, min_periods=1).mean()
            )
        # Fill NA that resulted from shift
        df[col_name] = df[col_name].fillna(0)

    logger.info(f"Created rolling features: {[f'roll_{w}' for w in windows]}")
    return df


def create_positivity_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate positivity rate: (INF_A + INF_B) / SPEC_PROCESSED_NB
    """
    df = df.copy()

    if "SPEC_PROCESSED_NB" in df.columns and TARGET_COLUMN in df.columns:
        df["positivity_rate"] = df[TARGET_COLUMN] / df["SPEC_PROCESSED_NB"].replace(0, np.nan)
        df["positivity_rate"] = df["positivity_rate"].replace(
            [np.inf, -np.inf], 0
        ).fillna(0)
        logger.info(
            f"Positivity rate created: mean={df['positivity_rate'].mean():.4f}"
        )
    else:
        df["positivity_rate"] = 0
        logger.warning("Cannot compute positivity rate: missing columns")

    return df


# encode_country removed for dynamic country-specific training.

def run_feature_pipeline(
    df: pd.DataFrame,
    group_col: str = None,
    is_training: bool = True,
) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.

    Pipeline:
    1. Create target (INF_A + INF_B)
    2. Encode Country (Panel Data)
    3. Create temporal features (Month)
    4. Create lag features (1, 2, 3 weeks)
    5. Create rolling averages (3, 5 weeks)
    6. Create positivity rate

    Args:
        df: Cleaned DataFrame from data_loader.
        group_col: Optional groupby column for per-country features.
        is_training: If True, fit encoders. If False, load encoders.

    Returns:
        DataFrame with all engineered features.
    """
    logger.info("Running feature engineering pipeline...")

    # Sort by time
    sort_cols = []
    if group_col and group_col in df.columns:
        sort_cols.append(group_col)
    if "ISO_YEAR" in df.columns:
        sort_cols.append("ISO_YEAR")
    if "ISO_WEEK" in df.columns:
        sort_cols.append("ISO_WEEK")

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    df = create_target(df)
    df = create_temporal_features(df)
    df = create_lag_features(df, group_col=group_col)
    df = create_rolling_features(df, group_col=group_col)
    df = create_positivity_rate(df)

    logger.info(
        f"Feature pipeline complete: {len(df)} rows × {len(df.columns)} columns"
    )
    return df


def prepare_ml_data(
    df: pd.DataFrame,
    features: list = None,
    target: str = None,
    split_ratio: float = None,
):
    """
    Prepare data for ML model training.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    features = features or FEATURE_COLUMNS
    target = target or TARGET_COLUMN
    split_ratio = split_ratio or TRAIN_TEST_SPLIT

    # Filter to available features
    available_features = [f for f in features if f in df.columns]
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        logger.warning(f"Missing features (excluded): {missing}")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in data")

    # Drop rows with NaN in features or target
    ml_df = df[available_features + [target]].dropna()

    if len(ml_df) == 0:
        raise ValueError("No valid data after dropping NaN rows")

    X = ml_df[available_features].values
    y = ml_df[target].values

    # Time-based split (not random, preserves temporal order)
    split_idx = int(len(X) * split_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(
        f"ML data prepared: train={len(X_train)}, test={len(X_test)}, "
        f"features={len(available_features)}"
    )

    return X_train, X_test, y_train, y_test, available_features


def prepare_lstm_data(X_train, X_test, y_train, y_test):
    """
    Scale and reshape data for LSTM model.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y)
    """
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

    # Reshape for LSTM: (samples, timesteps=1, features)
    X_train_lstm = X_train_scaled.reshape(
        X_train_scaled.shape[0], 1, X_train_scaled.shape[1]
    )
    X_test_lstm = X_test_scaled.reshape(
        X_test_scaled.shape[0], 1, X_test_scaled.shape[1]
    )

    logger.info(
        f"LSTM data prepared: X_train shape={X_train_lstm.shape}"
    )

    return X_train_lstm, X_test_lstm, y_train_scaled, y_test_scaled, scaler_X, scaler_y


def generate_future_features(
    df: pd.DataFrame,
    weeks_ahead: int = 12,
    features: list = None,
) -> pd.DataFrame:
    """
    Generate feature DataFrame for future predictions.
    Rolls forward temporal features from the last known data point.
    """
    features = features or FEATURE_COLUMNS
    df = df.copy()

    last_row = df.iloc[-1]
    last_year = int(last_row.get("ISO_YEAR", 2024))
    last_week = int(last_row.get("ISO_WEEK", 1))
    last_target = float(last_row.get(TARGET_COLUMN, 0))

    future_rows = []
    prev_targets = df[TARGET_COLUMN].tail(5).tolist() if TARGET_COLUMN in df.columns else [0] * 5

    country_name = last_row.get("COUNTRY_AREA_TERRITORY", "Unknown")

    for i in range(1, weeks_ahead + 1):
        week = last_week + i
        year = last_year
        if week > 52:
            week = week - 52
            year += 1

        month = ((week - 1) // 4) + 1

        # Use rolling predictions for lag features
        lag_1 = prev_targets[-1] if len(prev_targets) >= 1 else 0
        lag_2 = prev_targets[-2] if len(prev_targets) >= 2 else 0
        lag_3 = prev_targets[-3] if len(prev_targets) >= 3 else 0
        roll_3 = np.mean(prev_targets[-3:]) if len(prev_targets) >= 3 else lag_1
        roll_5 = np.mean(prev_targets[-5:]) if len(prev_targets) >= 5 else roll_3

        row = {
            "COUNTRY_AREA_TERRITORY": country_name,
            "ISO_YEAR": year,
            "ISO_WEEK": week,
            "Month": month,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "roll_3": roll_3,
            "roll_5": roll_5,
            "positivity_rate": 0,  # Not available for future
            "week_number": i,
        }

        future_rows.append(row)
        prev_targets.append(lag_1)  # Approximate; updated during prediction

    future_df = pd.DataFrame(future_rows)
    logger.info(f"Generated {weeks_ahead} weeks of future features")

    return future_df
