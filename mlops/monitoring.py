import psutil
import time
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger("epi_predict.monitoring")

def get_system_metrics() -> Dict[str, Any]:
    """Get system performance metrics."""
    try:
        # CPU & Memory
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_usage_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2)
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {"error": str(e)}

def check_data_quality(df) -> dict:
    """Perform data quality checks on a DataFrame."""
    if df is None or df.empty:
        return {"status": "fail", "reason": "Empty or None DataFrame"}
        
    metrics = {
        "row_count": len(df),
        "columns": len(df.columns),
        "missing_values_percent": round((df.isna().sum().sum() / df.size) * 100, 2)
    }
    
    if "ISO_YEAR" in df.columns and "ISO_WEEK" in df.columns:
        latest_year = df["ISO_YEAR"].max()
        latest_week = df[df["ISO_YEAR"] == latest_year]["ISO_WEEK"].max()
        metrics["latest_data_point"] = f"{latest_year}-W{latest_week:02d}"
        
    return metrics
