import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from data.data_loader import load_and_prepare, get_available_countries
from models.predictor import PredictionEngine
from dashboard.components.forecast_card import render_forecast_summary

st.set_page_config(page_title="AI Predictions | Epi Predict", layout="wide")

from dashboard.components.ui_helper import inject_custom_css
inject_custom_css()

st.title("AI Predictions & Forecasting")

# Setup engine
try:
    engine = PredictionEngine()
    if not engine.registry:
        st.error("No models found. Please train models first.")
        st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Sidebar Controls
st.sidebar.header("Forecast Settings")
horizon = st.sidebar.slider("Forecast Horizon (Weeks)", min_value=4, max_value=52, value=12, step=4)

available_models = [m for m in engine.registry.keys() if not m.startswith("_")]
# Add "Best Model" as an option at the top
model_options = ["Auto (Best Model)"] + available_models
selected_model_display = st.sidebar.selectbox("Select Model", model_options)
selected_model = None if selected_model_display == "Auto (Best Model)" else selected_model_display

try:
    df_full = load_and_prepare(country="ALL")
    countries = get_available_countries(df_full)
    if not countries: countries = ["India"]
except Exception as e:
    st.sidebar.error(f"Error loading countries: {e}")
    countries = ["India"]
selected_country = st.sidebar.selectbox("Select Country", countries, index=countries.index("India") if "India" in countries else 0)


@st.cache_data(ttl=300)
def get_predictions(country, weeks, model):
    df = load_and_prepare(country=country)
    forecast = engine.forecast_future(df, weeks_ahead=weeks, model_name=model)
    return df, forecast

with st.spinner("Generating AI forecast..."):
    historical_df, forecast_data = get_predictions(selected_country, horizon, selected_model)

actual_model_used = forecast_data["model_used"]

st.markdown(f"""
### {horizon}-Week Forecast for {selected_country}
Generated using **{actual_model_used}**
""")

# Render forecast summary cards
render_forecast_summary(forecast_data, limit=4)

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Forecast Trajectory")
    
    fore_data = forecast_data["forecast"]
    fore_x = [f"Week {w}" for w in fore_data["week_number"]]
    fore_y = fore_data["predicted_cases"]
    fore_lower = fore_data["lower_bound"]
    fore_upper = fore_data["upper_bound"]
    
    fig = go.Figure()
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=fore_x, y=fore_y,
        mode='lines+markers',
        name='Predicted Cases',
        line=dict(color='#00d4aa', width=3),
        marker=dict(size=8, color='#00d4aa')
    ))
    
    # Confidence Area
    fig.add_trace(go.Scatter(
        x=fore_x + fore_x[::-1],
        y=fore_upper + fore_lower[::-1],
        fill='toself',
        fillcolor='rgba(0, 212, 170, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='95% Confidence Interval'
    ))
    
    # Peak Outbreak Marker
    peak_idx = np.argmax(fore_y)
    fig.add_annotation(
        x=fore_x[peak_idx],
        y=fore_y[peak_idx],
        text=f"Projected Peak<br>{fore_y[peak_idx]:.0f} cases",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowcolor="#ef4444",
        font=dict(color="#ef4444", size=12),
        ay=-40
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Model Performance Comparison")
    
    # Get all metrics
    metrics_list = engine.get_all_metrics()
    if metrics_list:
        m_df = pd.DataFrame(metrics_list)
        # Highlight best
        best_m = engine.best_model_name
        
        # R2 comparison chart
        m_df = m_df.sort_values("r2_score", ascending=True) # Ascending for horizontal bar
        
        colors = ['#f59e0b' if m == actual_model_used else '#334155' for m in m_df['model']]
        
        fig2 = go.Figure(go.Bar(
            x=m_df['r2_score'],
            y=m_df['model'],
            orientation='h',
            marker_color=colors,
            text=m_df['r2_score'].apply(lambda x: f"{x:.3f}"),
            textposition='auto'
        ))
        
        fig2.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title="R² Score (Higher is better)",
            margin=dict(l=0, r=0, t=30, b=0),
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Comparison metrics not available.")

# Detailed Forecast Table
st.subheader("Detailed Forecast")
table_df = pd.DataFrame({
    "Week": fore_data["week_number"],
    "Year": fore_data["iso_year"],
    "ISO Week": fore_data["iso_week"],
    "Predicted Cases": [f"{x:,.0f}" for x in fore_data["predicted_cases"]],
    "Confidence Lower": [f"{x:,.0f}" for x in fore_data["lower_bound"]],
    "Confidence Upper": [f"{x:,.0f}" for x in fore_data["upper_bound"]],
})

from modules.risk_classifier import classify_batch
risk_levels = classify_batch(fore_data["predicted_cases"])
table_df["Risk Level"] = [r["label"] for r in risk_levels]

st.dataframe(table_df, use_container_width=True, hide_index=True)
