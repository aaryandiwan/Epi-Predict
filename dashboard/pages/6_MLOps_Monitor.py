import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd
from datetime import datetime
import time

from config.settings import LOGS_DIR, DATA_CACHE_TTL_HOURS
from data.data_loader import PROCESSED_DATA_FILE
from models.predictor import PredictionEngine

st.set_page_config(page_title="MLOps Monitor | Epi Predict", layout="wide")

# Load CSS
css_path = Path(__file__).resolve().parent.parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("MLOps & System Monitor")

col1, col2 = st.columns(2)

# --- System Status ---
with col1:
    st.subheader("System Status")
    
    # Data Cache Status
    data_status_html = "<div class='epi-card'>"
    if PROCESSED_DATA_FILE.exists():
        mtime = PROCESSED_DATA_FILE.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        is_fresh = age_hours < DATA_CACHE_TTL_HOURS
        color = "#22c55e" if is_fresh else "#f59e0b"
        icon = "✅" if is_fresh else "⚠️"
        
        data_status_html += f"""
        <p><strong style="color: #94a3b8;">Data Cache:</strong> <span style="color: {color};">{icon} {'Fresh' if is_fresh else 'Stale'}</span></p>
        <p><strong style="color: #94a3b8;">Last Updated:</strong> <span style="color: #f8fafc;">{datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')}</span></p>
        <p><strong style="color: #94a3b8;">Cache Age:</strong> <span style="color: #f8fafc;">{age_hours:.1f} hours</span> (TTL: {DATA_CACHE_TTL_HOURS}h)</p>
        """
    else:
        data_status_html += "<p><strong style='color: #94a3b8;'>Data Cache:</strong> <span style='color: #ef4444;'>🔴 Missing</span></p>"
    
    data_status_html += "</div>"
    st.markdown(data_status_html, unsafe_allow_html=True)

# --- Model Registry ---
with col2:
    st.subheader("Model Registry")
    try:
        engine = PredictionEngine()
        models_info = engine.get_model_info()
        best_model = engine.best_model_name
        
        if models_info:
            registry_data = []
            for name, info in models_info.items():
                if name.startswith("_"): continue
                registry_data.append({
                    "Model": f"🌟 {name}" if name == best_model else name,
                    "Version": info.get("version", 1),
                    "R² Score": f"{info.get('metrics', {}).get('r2_score', 0):.4f}",
                    "MAE": f"{info.get('metrics', {}).get('mae', 0):.2f}",
                    "Timestamp": info.get("timestamp", "")
                })
            st.dataframe(pd.DataFrame(registry_data), use_container_width=True, hide_index=True)
        else:
            st.warning("No models found in registry.")
    except Exception as e:
        st.error(f"Failed to load registry: {e}")

st.markdown("---")

# --- Retraining Pipeline ---
st.subheader("Retraining Pipeline")
st.markdown("Manually trigger the MLOps pipeline to fetch fresh WHO data and retrain all 7 AI models.")

if st.button("Trigger Full Pipeline Retrain", type="primary"):
    st.info("Pipeline started. This may take several minutes as it downloads data and trains all models (including LSTM and GridSearch).")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        from scripts.train_models import run_pipeline
        
        status_text.text("Fetching and preparing data...")
        progress_bar.progress(10)
        
        # In a real app we'd capture progress via callbacks, here we simulate the wait
        # then run it synchronously
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            results = run_pipeline(force_refresh=True, skip_lstm=False, skip_arima=False)
            
        progress_bar.progress(100)
        status_text.text("Pipeline completed successfully!")
        
        st.success(f"Retrained {len(results)} models successfully.")
        
        with st.expander("View Training Logs"):
            st.text(f.getvalue()[-2000:]) # Show tail of logs
            
        st.button("Refresh Page to see new models")
        
    except Exception as e:
        progress_bar.progress(100)
        status_text.text("Pipeline failed!")
        st.error(f"Error during retraining: {e}")
