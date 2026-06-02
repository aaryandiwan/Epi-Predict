import streamlit as st
import pandas as pd
from typing import Dict, Any

def render_forecast_card(week_num: int, cases: float, lower: float, upper: float, trend: str = "stable"):
    """Render a single forecast card with HTML/CSS."""
    
    trend_map = {
        "up": {"icon": "↗", "color": "#ef4444"},
        "down": {"icon": "↘", "color": "#22c55e"},
        "stable": {"icon": "→", "color": "#94a3b8"}
    }
    t = trend_map.get(trend, trend_map["stable"])
    
    html = f"""
    <div class="epi-card" style="text-align: center;">
        <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 0.5rem; text-transform: uppercase;">Week {week_num}</p>
        <h3 style="margin: 0; font-size: 1.8rem; color: #f8fafc;">{cases:,.0f}</h3>
        <p style="color: #64748b; font-size: 0.8rem; margin-top: 0.2rem;">Range: {lower:,.0f} - {upper:,.0f}</p>
        <div style="margin-top: 0.5rem; font-size: 1.2rem; color: {t['color']}; font-weight: bold;">
            {t['icon']}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_forecast_summary(forecast_data: Dict[str, Any], limit: int = 4):
    """Render a grid of forecast cards."""
    
    if not forecast_data or "forecast" not in forecast_data:
        st.warning("No forecast data available")
        return
        
    f_data = forecast_data["forecast"]
    weeks = f_data["week_number"][:limit]
    cases = f_data["predicted_cases"][:limit]
    lowers = f_data["lower_bound"][:limit]
    uppers = f_data["upper_bound"][:limit]
    
    cols = st.columns(len(weeks))
    
    for i, (col, w, c, l, u) in enumerate(zip(cols, weeks, cases, lowers, uppers)):
        # Determine trend
        trend = "stable"
        if i > 0:
            prev_c = cases[i-1]
            if c > prev_c * 1.05: trend = "up"
            elif c < prev_c * 0.95: trend = "down"
            
        with col:
            render_forecast_card(w, c, l, u, trend)
