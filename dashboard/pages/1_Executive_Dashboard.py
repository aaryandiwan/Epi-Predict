import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

from data.data_loader import load_and_prepare, get_available_countries
from models.predictor import PredictionEngine
from modules.risk_classifier import classify_risk
from dashboard.components.risk_meter import render_risk_meter
from dashboard.components.metrics_card import render_metrics_row
from dashboard.components.alert_banner import render_alert_banner

st.set_page_config(page_title="Executive Dashboard | Epi Predict", layout="wide")

# Load CSS
css_path = Path(__file__).resolve().parent.parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Executive Dashboard")

# Data Loading with Cache
@st.cache_data(ttl=300)
def get_dashboard_data(country):
    try:
        df = load_and_prepare(country=country)
        engine = PredictionEngine()
        
        if not engine.registry:
            return None, df, "models_missing"
            
        forecast = engine.forecast_future(df, weeks_ahead=12)
        return forecast, df, "success"
    except Exception as e:
        return None, None, str(e)

# Sidebar controls
st.sidebar.header("Global Filters")
# Try to get countries, default to India if fails
try:
    df_full = load_and_prepare(country=None) # Load full to get countries
    countries = get_available_countries(df_full)
    if not countries: countries = ["India"]
except:
    countries = ["India"]

selected_country = st.sidebar.selectbox("Select Country", countries, index=countries.index("India") if "India" in countries else 0)

# Fetch Data
with st.spinner("Loading outbreak intelligence..."):
    forecast_data, historical_df, status = get_dashboard_data(selected_country)

if status == "models_missing":
    st.error("No trained models found. Please train models first to view the dashboard.")
    st.info("Run `python scripts/train_models.py` or use the MLOps Monitor page.")
    st.stop()
elif status != "success":
    st.error(f"Error loading data: {status}")
    st.stop()
    
if historical_df is None or historical_df.empty:
    st.warning(f"No historical data available for {selected_country}.")
    st.stop()

# Calculations
current_week_cases = historical_df["Target"].iloc[-1] if "Target" in historical_df.columns else 0
prev_week_cases = historical_df["Target"].iloc[-2] if len(historical_df) > 1 and "Target" in historical_df.columns else 0
weekly_change = ((current_week_cases - prev_week_cases) / prev_week_cases * 100) if prev_week_cases > 0 else 0

predicted_next = forecast_data["forecast"]["predicted_cases"][0]
risk_info = classify_risk(predicted_next)

# Top Alert Banner
if risk_info["level"] in ["high", "severe"]:
    render_alert_banner(
        level=risk_info["level"], 
        message=f"Forecast indicates {predicted_next:,.0f} cases next week for {selected_country}.",
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M")
    )

# Top Row: Risk Meter & KPIs
col1, col2 = st.columns([1, 2])

with col1:
    render_risk_meter(risk_info["level"], predicted_next, title="Next Week Risk")

with col2:
    # Get best model metrics
    engine = PredictionEngine()
    best_model = engine.best_model_name
    best_r2 = 0.0
    if best_model:
        metrics = engine.get_model_info(best_model).get("metrics", {})
        best_r2 = metrics.get("r2_score", 0.0)
        
    total_processed = historical_df["SPEC_PROCESSED_NB"].sum() if "SPEC_PROCESSED_NB" in historical_df.columns else 0

    metrics = [
        {
            "title": "Current Week Cases",
            "value": f"{current_week_cases:,.0f}",
            "delta": f"{weekly_change:+.1f}%",
            "delta_type": "inverse", # higher cases is bad
            "icon": "🦠"
        },
        {
            "title": "Predicted (Next Wk)",
            "value": f"{predicted_next:,.0f}",
            "delta": f"{((predicted_next - current_week_cases) / current_week_cases * 100):+.1f}%" if current_week_cases > 0 else "0%",
            "delta_type": "inverse",
            "icon": "🔮"
        },
        {
            "title": "Best Model Accuracy",
            "value": f"{best_r2:.2f} R²",
            "icon": "🧠"
        },
        {
            "title": "Total Specimens",
            "value": f"{total_processed:,.0f}",
            "icon": "🧪"
        }
    ]
    render_metrics_row(metrics[:2])
    render_metrics_row(metrics[2:])

st.markdown("---")

# Middle Row: Charts
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Historical Trend & 12-Week Forecast")
    
    # Prepare data for plot
    hist_plot_df = historical_df.tail(24).copy() # Last 24 weeks
    
    # Create date index string for plotting
    hist_x = [f"{y}-W{w:02d}" for y, w in zip(hist_plot_df["ISO_YEAR"], hist_plot_df["ISO_WEEK"])]
    hist_y = hist_plot_df["Target"].values if "Target" in hist_plot_df.columns else np.zeros(len(hist_plot_df))
    
    fore_data = forecast_data["forecast"]
    fore_x = [f"{y}-W{w:02d}" for y, w in zip(fore_data["iso_year"], fore_data["iso_week"])]
    fore_y = fore_data["predicted_cases"]
    fore_lower = fore_data["lower_bound"]
    fore_upper = fore_data["upper_bound"]
    
    # Plotly Figure
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y,
        mode='lines+markers',
        name='Historical',
        line=dict(color='#00d4aa', width=3),
        marker=dict(size=6)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=fore_x, y=fore_y,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#f59e0b', width=3, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Confidence Band
    fig.add_trace(go.Scatter(
        x=fore_x + fore_x[::-1],
        y=fore_upper + fore_lower[::-1],
        fill='toself',
        fillcolor='rgba(245, 158, 11, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name='95% Confidence'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Subtype Distribution")
    
    # Pie chart of subtypes for latest available year
    if len(historical_df) > 0:
        latest_year = historical_df["ISO_YEAR"].max()
        year_df = historical_df[historical_df["ISO_YEAR"] == latest_year]
        
        subtypes = {"A(H1N1)pdm09": "AH1N12009", "A(H3N2)": "AH3", "B/Victoria": "BVIC_2DEL", "B/Yamagata": "BYAM"}
        available_subtypes = {name: year_df[col].sum() for name, col in subtypes.items() if col in year_df.columns}
        
        if sum(available_subtypes.values()) > 0:
            fig = px.pie(
                values=list(available_subtypes.values()), 
                names=list(available_subtypes.keys()),
                hole=0.6,
                color_discrete_sequence=['#00d4aa', '#3b82f6', '#f59e0b', '#ef4444']
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=20, b=0),
                showlegend=True,
                legend=dict(orientation="h", y=-0.2)
            )
            # Add center text
            fig.add_annotation(text=f"<b>{latest_year}</b><br>Subtypes", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Subtype data not available for this region.")
