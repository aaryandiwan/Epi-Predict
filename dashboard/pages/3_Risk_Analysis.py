import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

from data.data_loader import load_and_prepare, get_available_countries
from models.predictor import PredictionEngine
from modules.risk_classifier import classify_risk, classify_batch
from modules.recommendation_engine import get_recommendations
from dashboard.components.prevention_card import render_prevention_card

st.set_page_config(page_title="Risk Analysis | Epi Predict", layout="wide")

from dashboard.components.ui_helper import inject_custom_css
inject_custom_css()

st.title("Outbreak Risk & Recommendations")

try:
    engine = PredictionEngine()
    if not engine.registry:
        st.error("No models found. Please train models first.")
        st.stop()
        
    df_full = load_and_prepare(country="ALL")
    countries = get_available_countries(df_full)
    if not countries: countries = ["India"]
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

selected_country = st.selectbox("Select Region for Risk Analysis", countries, index=countries.index("India") if "India" in countries else 0)

@st.cache_data(ttl=300)
def get_risk_data(country):
    df = load_and_prepare(country=country)
    forecast = engine.forecast_future(df, weeks_ahead=12)
    return df, forecast

with st.spinner("Analyzing risk..."):
    historical_df, forecast_data = get_risk_data(selected_country)

preds = forecast_data["forecast"]["predicted_cases"]
current_pred = preds[0]
risk_info = classify_risk(current_pred)
recs = get_recommendations(risk_info["level"])

col1, col2 = st.columns([1, 2])

with col1:
    # Large Risk Badge
    bg_colors = {"low": "rgba(34, 197, 94, 0.1)", "moderate": "rgba(245, 158, 11, 0.1)", 
                 "high": "rgba(239, 68, 68, 0.1)", "severe": "rgba(124, 45, 18, 0.2)"}
    
    st.markdown(f"""
    <div class="epi-card" style="text-align: center; border-color: {risk_info['color']}; background: {bg_colors.get(risk_info['level'], 'rgba(255,255,255,0.05)')}">
        <h2 style="color: {risk_info['color']}; margin-bottom: 0; font-size: 2.5rem;">
            {risk_info['icon']} {risk_info['label'].upper()}
        </h2>
        <p style="color: #94a3b8; font-size: 1.2rem; margin-top: 0.5rem;">Current Outbreak Risk Status</p>
        <hr style="border-color: rgba(255,255,255,0.1);">
        <h3 style="color: #f8fafc; margin: 0;">{current_pred:,.0f}</h3>
        <p style="color: #64748b; font-size: 0.9rem; margin-top: 0;">Predicted Cases Next Week</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"**AI Summary:** {recs['summary']}")

with col2:
    st.subheader("Actionable Recommendations")
    render_prevention_card(recs["level"], recs["actions"], recs["urgency"])

st.markdown("---")
st.subheader("Risk Trajectory (Next 12 Weeks)")

# Timeline of risk
risk_timeline = classify_batch(preds)

cols = st.columns(12)
for i, (col, r, p) in enumerate(zip(cols, risk_timeline, preds)):
    with col:
        st.markdown(f"""
        <div style="text-align: center; padding: 10px 5px; border-radius: 8px; background: rgba(255,255,255,0.05); border-top: 4px solid {r['color']};">
            <p style="margin: 0; font-size: 0.7rem; color: #94a3b8;">Wk {i+1}</p>
            <div style="font-size: 1.5rem; margin: 5px 0;">{r['icon']}</div>
            <p style="margin: 0; font-size: 0.8rem; font-weight: bold; color: {r['color']};">{r['level'][:3].upper()}</p>
        </div>
        """, unsafe_allow_html=True)
