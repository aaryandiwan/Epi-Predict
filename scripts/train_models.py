"""
Epi Predict – Model Training CLI Script

End-to-end pipeline to fetch data, engineer features, and train all 7 models.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import logger
from data.data_loader import load_and_prepare
from models.feature_engineering import run_feature_pipeline, prepare_ml_data
from models.trainer import train_all_models

def run_pipeline(force_refresh=False, skip_lstm=False, skip_arima=False, country="India"):
    """Execute the full training pipeline."""
    logger.info("=" * 50)
    logger.info("EPI PREDICT - MODEL TRAINING PIPELINE")
    logger.info("=" * 50)
    
    # 1. Load Data
    logger.info("\n--- Phase 1: Data Loading ---")
    df = load_and_prepare(country=country, force_refresh=force_refresh)
    if df.empty:
        logger.error("Data loading failed. Exiting.")
        return
        
    # 2. Feature Engineering
    logger.info("\n--- Phase 2: Feature Engineering ---")
    df_features = run_feature_pipeline(df, group_col="COUNTRY_AREA_TERRITORY")
    
    # Prepare ML inputs
    X_train, X_test, y_train, y_test, features = prepare_ml_data(df_features)
    
    # Prepare ARIMA inputs (needs raw series)
    target_series = df_features["Target"] if "Target" in df_features.columns else None
    train_series = None
    test_series = None
    
    if target_series is not None:
        split_idx = int(len(target_series) * 0.8)
        train_series = target_series.iloc[:split_idx]
        test_series = target_series.iloc[split_idx:]
        
    # 3. Model Training
    logger.info("\n--- Phase 3: Model Training ---")
    results = train_all_models(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        features=features,
        train_series=train_series,
        test_series=test_series,
        skip_lstm=skip_lstm,
        skip_arima=skip_arima
    )
    
    logger.info(f"\nPipeline complete! Successfully trained {len(results)} models.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Epi Predict ML Models")
    parser.add_argument("--country", type=str, default="India", help="Country to train on (default: India)")
    parser.add_argument("--all-countries", action="store_true", help="Train Universal Panel Model on all countries")
    parser.add_argument("--force-refresh", action="store_true", help="Force refresh data from WHO API")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM model training (useful if TF not installed)")
    parser.add_argument("--skip-arima", action="store_true", help="Skip ARIMA model training")
    
    args = parser.parse_args()
    
    target_country = "ALL" if args.all_countries else args.country
    
    # ARIMA is mathematically invalid for stacked panel data, so we must skip it
    should_skip_arima = args.skip_arima or args.all_countries
    
    run_pipeline(
        force_refresh=args.force_refresh,
        skip_lstm=args.skip_lstm,
        skip_arima=should_skip_arima,
        country=target_country
    )
