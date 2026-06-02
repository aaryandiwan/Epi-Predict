import streamlit as st
from typing import List, Dict, Any

def render_metric_card(title: str, value: str, delta: str = None, delta_type: str = "normal", icon: str = ""):
    """Render a single KPI metric card."""
    
    delta_html = ""
    if delta:
        delta_color = "#22c55e" if delta_type == "normal" and not delta.startswith("-") or delta_type == "inverse" and delta.startswith("-") else "#ef4444"
        if delta_type == "neutral": delta_color = "#94a3b8"
        
        arrow = "↑" if not delta.startswith("-") else "↓"
        clean_delta = delta.replace("-", "").replace("+", "")
        
        delta_html = f"""
        <div style="color: {delta_color}; font-size: 0.85rem; font-weight: 500; display: flex; align-items: center; margin-top: 0.5rem;">
            <span>{arrow} {clean_delta}</span>
        </div>
        """
        
    html = f"""
    <div class="epi-card" style="padding: 1.25rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div>
                <p style="color: #94a3b8; font-size: 0.8rem; margin: 0 0 0.5rem 0; text-transform: uppercase; letter-spacing: 0.05em;">{title}</p>
                <h3 style="color: #f8fafc; font-size: 1.8rem; margin: 0; font-weight: 600;">{value}</h3>
                {delta_html}
            </div>
            <div style="font-size: 1.5rem; opacity: 0.5;">
                {icon}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_metrics_row(metrics: List[Dict[str, Any]]):
    """Render a row of metric cards."""
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            render_metric_card(**metric)
