"""
Epi Predict – WHO FluNet Data Loader

Handles fetching influenza surveillance data from the WHO FluNet API,
local caching, cleaning, and country/region filtering.
"""

import os
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from config.settings import (
    WHO_FLUNET_API_URL,
    DATA_CACHE_TTL_HOURS,
    LOCAL_DATA_FILE,
    PROCESSED_DATA_FILE,
    DEFAULT_COUNTRY,
    RAW_DATA_DIR,
)

logger = logging.getLogger("epi_predict.data")


def fetch_who_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch influenza data from the WHO FluNet API with local caching.

    Args:
        force_refresh: If True, bypass cache and re-download.

    Returns:
        Raw DataFrame from WHO FluNet.
    """
    cache_file = LOCAL_DATA_FILE

    # Check if cached data is still fresh
    if not force_refresh and cache_file.exists():
        file_age_hours = (
            time.time() - cache_file.stat().st_mtime
        ) / 3600
        if file_age_hours < DATA_CACHE_TTL_HOURS:
            logger.info(
                f"Loading cached WHO data (age: {file_age_hours:.1f}h)"
            )
            return pd.read_csv(cache_file, low_memory=False)

    # Fetch from WHO API
    logger.info("Fetching fresh data from WHO FluNet API...")
    try:
        df = pd.read_csv(WHO_FLUNET_API_URL, low_memory=False)
        # Cache locally
        df.to_csv(cache_file, index=False)
        logger.info(
            f"WHO data fetched and cached: {len(df)} rows × {len(df.columns)} columns"
        )
        return df
    except Exception as e:
        logger.error(f"Failed to fetch WHO data: {e}")
        # Fall back to cache if available
        if cache_file.exists():
            logger.warning("Falling back to cached data")
            return pd.read_csv(cache_file, low_memory=False)
        raise RuntimeError(
            "Cannot fetch WHO data and no local cache available. "
            "Please check your internet connection."
        ) from e


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw WHO FluNet data.

    - Fill categorical NaN with mode
    - Fill numerical NaN with year-wise median
    - Handle column name normalization
    """
    logger.info("Cleaning raw data...")
    df = df.copy()

    # Normalize column names that may differ between API versions
    column_map = {
        "Country, Area or Territory": "COUNTRY_AREA_TERRITORY",
        "COUNTRY_AREA_TERRITORY": "COUNTRY_AREA_TERRITORY",
        "WHO REGION": "WHOREGION",
        "WHOREGION": "WHOREGION",
        "WHO_REGION": "WHOREGION",
        "FLU SEASON": "FLU_SEASON",
        "FLU_SEASON": "FLU_SEASON",
    }

    for old_name, new_name in column_map.items():
        if old_name in df.columns and old_name != new_name:
            df.rename(columns={old_name: new_name}, inplace=True)

    # Identify column types
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Fill categorical missing values with mode
    for col in cat_cols:
        if df[col].isna().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col].fillna(mode_val[0], inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)

    # Fill numerical missing values with year-wise median
    if "ISO_YEAR" in df.columns and len(num_cols) > 0:
        for col in num_cols:
            if df[col].isna().any():
                df[col] = df.groupby("ISO_YEAR")[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Fill remaining NaN (years with all NaN)
                df[col].fillna(0, inplace=True)
    else:
        df[num_cols] = df[num_cols].fillna(0)

    # Convert date columns
    for date_col in ["ISO_WEEKSTARTDATE", "MMWR_WEEKSTARTDATE"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Ensure key numeric columns exist and are numeric
    for col in ["INF_A", "INF_B", "SPEC_PROCESSED_NB", "SPEC_RECEIVED_NB"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    logger.info(f"Data cleaned: {len(df)} rows × {len(df.columns)} columns")
    return df


def filter_by_country(
    df: pd.DataFrame, country: str = None
) -> pd.DataFrame:
    """Filter data for a specific country."""
    country = country or DEFAULT_COUNTRY
    col = "COUNTRY_AREA_TERRITORY"
    if col not in df.columns:
        logger.warning(f"Column '{col}' not found. Returning full dataset.")
        return df

    filtered = df[df[col].str.strip().str.lower() == country.strip().lower()].copy()

    if len(filtered) == 0:
        logger.warning(
            f"No data found for country '{country}'. "
            f"Available countries: {df[col].unique()[:10]}..."
        )
        return df

    # Sort chronologically
    if "ISO_YEAR" in filtered.columns and "ISO_WEEK" in filtered.columns:
        filtered.sort_values(["ISO_YEAR", "ISO_WEEK"], inplace=True)
        filtered.reset_index(drop=True, inplace=True)

    logger.info(f"Filtered for '{country}': {len(filtered)} rows")
    return filtered


def filter_by_region(
    df: pd.DataFrame, region: str
) -> pd.DataFrame:
    """Filter data by WHO region code."""
    col = "WHOREGION"
    if col not in df.columns:
        logger.warning(f"Column '{col}' not found. Returning full dataset.")
        return df

    filtered = df[df[col].str.strip().str.upper() == region.strip().upper()].copy()
    logger.info(f"Filtered for WHO region '{region}': {len(filtered)} rows")
    return filtered


def get_available_countries(df: pd.DataFrame) -> list:
    """Get list of all available countries in the dataset."""
    col = "COUNTRY_AREA_TERRITORY"
    if col in df.columns:
        return sorted(df[col].dropna().unique().tolist())
    return []


def get_available_regions(df: pd.DataFrame) -> list:
    """Get list of all available WHO regions in the dataset."""
    col = "WHOREGION"
    if col in df.columns:
        return sorted(df[col].dropna().unique().tolist())
    return []


def aggregate_global(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data globally by ISO_YEAR and ISO_WEEK.
    Sums up all numeric surveillance columns.
    """
    if "ISO_YEAR" not in df.columns or "ISO_WEEK" not in df.columns:
        logger.warning("Cannot aggregate: missing ISO_YEAR/ISO_WEEK")
        return df

    num_cols = [
        "INF_A", "INF_B", "SPEC_PROCESSED_NB", "SPEC_RECEIVED_NB",
        "AH1N12009", "AH1", "AH3", "AH5", "ANOTSUBTYPED",
        "INF_ALL", "INF_NEGATIVE",
    ]
    available_num = [c for c in num_cols if c in df.columns]

    agg_dict = {col: "sum" for col in available_num}
    agg_df = df.groupby(["ISO_YEAR", "ISO_WEEK"]).agg(agg_dict).reset_index()
    agg_df.sort_values(["ISO_YEAR", "ISO_WEEK"], inplace=True)
    agg_df.reset_index(drop=True, inplace=True)

    logger.info(f"Global aggregation: {len(agg_df)} weekly records")
    return agg_df


def load_and_prepare(
    country: str = None,
    force_refresh: bool = False,
    use_global: bool = False,
) -> pd.DataFrame:
    """
    Full data pipeline: fetch → clean → filter → return.

    Args:
        country: Country to filter for. Defaults to settings.DEFAULT_COUNTRY.
        force_refresh: Force re-download from WHO API.
        use_global: If True, aggregate globally instead of filtering by country.

    Returns:
        Cleaned, filtered DataFrame ready for feature engineering.
    """
    # Check for processed cache
    if not force_refresh and PROCESSED_DATA_FILE.exists():
        file_age_hours = (
            time.time() - PROCESSED_DATA_FILE.stat().st_mtime
        ) / 3600
        if file_age_hours < DATA_CACHE_TTL_HOURS:
            logger.info("Loading processed data from cache")
            df = pd.read_csv(PROCESSED_DATA_FILE, low_memory=False)
            # Re-apply country filter
            if not use_global and country:
                df = filter_by_country(df, country)
            return df

    # Full pipeline
    raw_df = fetch_who_data(force_refresh=force_refresh)
    clean_df = clean_data(raw_df)

    # Save processed data
    clean_df.to_csv(PROCESSED_DATA_FILE, index=False)

    # Apply filter
    if use_global:
        return aggregate_global(clean_df)
    else:
        return filter_by_country(clean_df, country)
