import streamlit as st

def render_prevention_card(level: str, actions: list, urgency: str):
    """Render a prevention recommendation card."""
    
    # Map colors based on level
    color_map = {
        "low": {"border": "#22c55e", "bg": "rgba(34, 197, 94, 0.05)", "icon": "✅"},
        "moderate": {"border": "#f59e0b", "bg": "rgba(245, 158, 11, 0.05)", "icon": "⚠️"},
        "high": {"border": "#ef4444", "bg": "rgba(239, 68, 68, 0.05)", "icon": "🔴"},
        "severe": {"border": "#7c2d12", "bg": "rgba(124, 45, 18, 0.1)", "icon": "🚨"}
    }
    
    style = color_map.get(level.lower(), color_map["low"])
    
    # Build actions list HTML
    actions_html = ""
    for action in actions:
        actions_html += f"""
        <li style="margin-bottom: 0.5rem; display: flex; align-items: flex-start;">
            <span style="color: {style['border']}; margin-right: 0.5rem;">•</span>
            <span style="color: #e2e8f0;">{action}</span>
        </li>
        """
        
    html = f"""
    <div class="epi-card" style="border-left: 4px solid {style['border']}; background: {style['bg']}; height: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h3 style="margin: 0; color: {style['border']}; font-size: 1.2rem;">
                {style['icon']} {level.upper()} RISK
            </h3>
            <span style="font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; border: 1px solid {style['border']}; color: {style['border']}; text-transform: uppercase;">
                Urgency: {urgency}
            </span>
        </div>
        <ul style="list-style-type: none; padding-left: 0; margin: 0;">
            {actions_html}
        </ul>
    </div>
    """
    
    st.markdown(html, unsafe_allow_html=True)
