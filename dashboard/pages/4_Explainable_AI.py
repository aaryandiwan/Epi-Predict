import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from models.predictor import PredictionEngine
from modules.explainability import get_feature_importance, get_shap_analysis, generate_explanation_text

st.set_page_config(page_title="Explainable AI | Epi Predict", layout="wide")

# Load CSS
css_path = Path(__file__).resolve().parent.parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Explainable AI (XAI)")
st.markdown("Understand how the machine learning models make their influenza predictions.")

try:
    engine = PredictionEngine()
    if not engine.registry:
        st.error("No models found. Please train models first.")
        st.stop()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

# Get models that support feature importance (Tree based)
tree_models = [m for m in engine.registry.keys() if not m.startswith("_") and ("forest" in m.lower() or "xgb" in m.lower())]

if not tree_models:
    st.warning("No tree-based models (Random Forest, XGBoost) available for explainability analysis.")
    st.stop()

selected_model = st.selectbox("Select Model to Analyze", tree_models, index=tree_models.index(engine.best_model_name) if engine.best_model_name in tree_models else 0)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Feature Importance ({selected_model})")
    
    with st.spinner("Calculating SHAP values..."):
        importance_data = get_feature_importance(selected_model)
        
    if importance_data:
        df_imp = pd.DataFrame(importance_data)
        df_imp = df_imp.sort_values("importance", ascending=True) # Ascending for horizontal bar chart
        
        # Format labels to be more readable
        feature_labels = {
            "Month": "Month of Year",
            "lag_1": "Cases (1 Week Ago)",
            "lag_2": "Cases (2 Weeks Ago)",
            "lag_3": "Cases (3 Weeks Ago)",
            "roll_3": "3-Week Avg Cases",
            "roll_5": "5-Week Avg Cases",
            "positivity_rate": "Test Positivity Rate",
            "ISO_WEEK": "Week of Year",
            "ISO_YEAR": "Year"
        }
        df_imp["Label"] = df_imp["feature"].map(lambda x: feature_labels.get(x, x))
        
        fig = go.Figure(go.Bar(
            x=df_imp["importance"],
            y=df_imp["Label"],
            orientation='h',
            marker=dict(
                color=df_imp["importance"],
                colorscale="Tealgrn"
            ),
        ))
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Relative Importance Score",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Could not calculate feature importance for {selected_model}")

with col2:
    st.subheader("Model Decision Logic")
    
    st.markdown("""
    <div class="epi-card">
        <h4 style="color: #00d4aa; margin-top: 0;">How it works</h4>
        <p style="color: #cbd5e1; font-size: 0.9rem;">
        The AI looks at several historical factors to predict future outbreaks. 
        Features with longer bars on the left had a stronger influence on the model's overall learned patterns.
        </p>
        <hr style="border-color: rgba(255,255,255,0.1);">
        <h4 style="color: #00d4aa;">Key Drivers</h4>
        <ul style="color: #cbd5e1; font-size: 0.9rem; padding-left: 20px;">
            <li><b>Lag Features:</b> Recent case numbers are usually the strongest predictors of near-term future cases.</li>
            <li><b>Rolling Averages:</b> Help the model understand the broader trend, smoothing out weekly reporting anomalies.</li>
            <li><b>Time Features:</b> (Month, Week) allow the model to learn the seasonality of influenza.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("Cross-Model Metrics Comparison")

metrics_list = engine.get_all_metrics()
if metrics_list:
    m_df = pd.DataFrame(metrics_list)
    # Format columns
    display_cols = ["model", "r2_score", "mae", "rmse"]
    if "train_time_seconds" in m_df.columns: display_cols.append("train_time_seconds")
    
    m_df = m_df[display_cols].copy()
    m_df.columns = ["Model", "R² Score (↑)", "MAE (↓)", "RMSE (↓)", "Train Time (s)"][:len(m_df.columns)]
    
    # Highlight max R2 and min errors
    st.dataframe(
        m_df.style.highlight_max(subset=["R² Score (↑)"], color="#065f46")
                 .highlight_min(subset=["MAE (↓)", "RMSE (↓)"], color="#065f46")
                 .format(precision=4),
        use_container_width=True,
        hide_index=True
    )
