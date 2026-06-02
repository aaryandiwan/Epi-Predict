import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

from config.settings import PREDICTION_LOG_DB

logger = logging.getLogger("epi_predict.prediction_logger")

class PredictionLogger:
    """Logs predictions to SQLite database for monitoring and audit."""
    
    def __init__(self):
        self.db_path = PREDICTION_LOG_DB
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database with schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            model_name TEXT NOT NULL,
            input_features_json TEXT,
            predicted_value REAL NOT NULL,
            confidence_lower REAL,
            confidence_upper REAL,
            risk_level TEXT
        )
        ''')
        
        # Index on timestamp for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions (timestamp)')
        
        conn.commit()
        conn.close()
        
    def log_prediction(self, model_name: str, input_features: dict, predicted_value: float, 
                       confidence_lower: float = None, confidence_upper: float = None, 
                       risk_level: str = None) -> int:
        """Log a prediction event."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().isoformat()
            features_json = json.dumps(input_features) if input_features else None
            
            cursor.execute('''
            INSERT INTO predictions 
            (timestamp, model_name, input_features_json, predicted_value, confidence_lower, confidence_upper, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, model_name, features_json, predicted_value, confidence_lower, confidence_upper, risk_level))
            
            row_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return row_id
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")
            return -1

    def get_recent_logs(self, limit: int = 100) -> list:
        """Retrieve recent prediction logs."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to fetch logs: {e}")
            return []
