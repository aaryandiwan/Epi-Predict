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

# Load Custom CSS
def load_css():
    css_path = Path(__file__).parent / "styles" / "custom.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom CSS not found.")

load_css()

# Sidebar Branding
st.sidebar.markdown(f"""
<div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color: #00d4aa; margin-bottom: 0;">{DASHBOARD_ICON} Epi Predict</h1>
    <p style="color: #94a3b8; font-size: 0.8rem; margin-top: 0;">Influenza Early Warning System</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Landing Page Content
st.markdown(f"""
<div class="hero-section">
    <div style="font-size: 4rem; margin-bottom: 1rem;">🦠</div>
    <h1 class="hero-title">{DASHBOARD_TITLE}</h1>
    <p class="hero-subtitle">
        An advanced AI-powered healthcare application designed to forecast seasonal influenza outbreaks globally using WHO surveillance data. 
        Transforming raw data into actionable intelligence.
    </p>
</div>

<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">🧠</div>
        <div class="feature-title">AI Prediction Engine</div>
        <div class="feature-desc">Powered by 7 machine learning models including XGBoost, Random Forest, and Stacking Ensembles for maximum accuracy.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🚨</div>
        <div class="feature-title">Early Warning System</div>
        <div class="feature-desc">Dynamically classifies outbreak risk levels and provides actionable medical recommendations based on severity.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🔍</div>
        <div class="feature-title">Explainable AI (XAI)</div>
        <div class="feature-desc">Peer into the "black box" using SHAP to understand exactly which features drive the model's forecasts.</div>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🌍</div>
        <div class="feature-title">Global Monitoring</div>
        <div class="feature-desc">Interactive Plotly visualizations of historical trends, regional comparisons, and future outbreak forecasts.</div>
    </div>
</div>

<div style="text-align: center; margin-top: 3rem; margin-bottom: 2rem; color: #00d4aa; font-weight: 600;">
    👈 Please select a module from the sidebar to begin exploring.
</div>
""", unsafe_allow_html=True)

# Quick Stats / Status
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="epi-card">
        <h4 style="color: #00d4aa; margin:0;">7</h4>
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
