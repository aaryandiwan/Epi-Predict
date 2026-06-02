import streamlit as st

def render_alert_banner(level: str, message: str, timestamp: str = None):
    """Render a top-of-page alert banner."""
    
    level = level.lower()
    
    # Map colors based on level
    if level == "severe":
        bg_color = "rgba(124, 45, 18, 0.8)"
        border_color = "#fca5a5"
        icon = "🚨"
        pulse_class = "alert-pulse"
    elif level == "high":
        bg_color = "rgba(239, 68, 68, 0.8)"
        border_color = "#fca5a5"
        icon = "🔴"
        pulse_class = "alert-pulse"
    elif level == "moderate":
        bg_color = "rgba(245, 158, 11, 0.8)"
        border_color = "#fde68a"
        icon = "⚠️"
        pulse_class = ""
    else:
        # Don't show banner for low risk
        return
        
    time_str = f"<span style='font-size: 0.8rem; opacity: 0.8; margin-left: 1rem;'>{timestamp}</span>" if timestamp else ""
        
    html = f"""
    <div class="{pulse_class}" style="background-color: {bg_color}; border-left: 4px solid {border_color}; border-radius: 4px; padding: 1rem; margin-bottom: 1.5rem; display: flex; align-items: center; color: white;">
        <div style="font-size: 1.5rem; margin-right: 1rem;">{icon}</div>
        <div style="flex-grow: 1;">
            <strong style="font-size: 1.1rem; letter-spacing: 0.5px;">OUTBREAK ALERT: {level.upper()} RISK</strong><br>
            <span style="font-size: 0.95rem;">{message}</span>
            {time_str}
        </div>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)
