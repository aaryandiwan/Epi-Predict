import sys
from pathlib import Path

# Add project root to path so imports work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from config.settings import DASHBOARD_TITLE, DASHBOARD_ICON

# Must be the first Streamlit command
st.set_page_config(
    page_title="Epi Predict",
    page_icon=DASHBOARD_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

from dashboard.components.ui_helper import inject_custom_css

# Inject CSS and Full-Screen Background
inject_custom_css()

# Sidebar Branding
st.sidebar.markdown(f"""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color: #00E5A8; margin-bottom: 0;">EpiPredict</h1>
    <p style="color: #94a3b8; font-size: 0.8rem; margin-top: 0;">Influenza Early Warning System</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Landing Page Content
st.markdown(f"""
<div class="hero-section">
    <h1 class="hero-title">EpiPredict</h1>
    <p class="hero-subtitle" style="font-weight: 600; font-size: 1.4rem; color: #FFFFFF; margin-bottom: 0.5rem;">
        AI-Powered Influenza Outbreak Early Warning System
    </p>
    <p class="hero-subtitle" style="font-size: 1.1rem;">
        Predicting Flu Trends Before They Become Public Health Emergencies.
    </p>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
        </div>
        <div class="feature-title">AI Prediction Engine</div>
        <div class="feature-desc">Powered by 7 machine learning models including XGBoost, Random Forest, and Stacking Ensembles for maximum accuracy.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
        </div>
        <div class="feature-title">Early Warning System</div>
        <div class="feature-desc">Dynamically classifies outbreak risk levels and provides actionable medical recommendations based on severity.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
        </div>
        <div class="feature-title">Explainable AI (XAI)</div>
        <div class="feature-desc">Peer into the "black box" using SHAP to understand exactly which features drive the model's forecasts.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>
        </div>
        <div class="feature-title">Global Monitoring</div>
        <div class="feature-desc">Interactive Plotly visualizations of historical trends, regional comparisons, and future outbreak forecasts.</div>
    </div>
</div>

<div style="text-align: center; margin-top: 3rem; margin-bottom: 2rem; color: #00E5A8; font-weight: 600;">
    👈 Please select a module from the sidebar to begin exploring.
</div>
""", unsafe_allow_html=True)

# Quick Stats / Status
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="epi-card">
        <h4 style="color: #00E5A8; margin:0;">7</h4>
        <p style="color: #94a3b8; margin:0; font-size:0.8rem;">ML Models</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="epi-card">
        <h4 style="color: #f59e0b; margin:0;">130+</h4>
        <p style="color: #94a3b8; margin:0; font-size:0.8rem;">Countries Supported</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="epi-card">
        <h4 style="color: #22c55e; margin:0;">12 Weeks</h4>
        <p style="color: #94a3b8; margin:0; font-size:0.8rem;">Forecast Horizon</p>
    </div>
    """, unsafe_allow_html=True)
