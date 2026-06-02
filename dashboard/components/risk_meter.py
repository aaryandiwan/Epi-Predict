import streamlit as st
import plotly.graph_objects as go

def render_risk_meter(risk_level: str, predicted_cases: float, title: str = "Current Outbreak Risk"):
    """
    Renders a Plotly gauge chart for risk level.
    """
    # Define thresholds
    thresholds = [0, 500, 2000, 5000, 10000] # Adjust max as needed
    
    # Determine color and text based on risk level
    level_map = {
        "low": {"color": "#22c55e", "text": "LOW", "value": 250},
        "moderate": {"color": "#f59e0b", "text": "MODERATE", "value": 1250},
        "high": {"color": "#ef4444", "text": "HIGH", "value": 3500},
        "severe": {"color": "#7c2d12", "text": "SEVERE", "value": 7500}
    }
    
    level_info = level_map.get(risk_level.lower(), level_map["low"])
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_cases,
        title={'text': title, 'font': {'size': 24, 'color': '#f8fafc'}},
        number={'font': {'size': 48, 'color': level_info["color"]}, 'valueformat': '.0f', 'suffix': ' Cases'},
        gauge={
            'axis': {'range': [None, 10000], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': level_info["color"], 'thickness': 0.25},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [thresholds[0], thresholds[1]], 'color': 'rgba(34, 197, 94, 0.2)'},
                {'range': [thresholds[1], thresholds[2]], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [thresholds[2], thresholds[3]], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [thresholds[3], thresholds[4]], 'color': 'rgba(124, 45, 18, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': predicted_cases
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#e2e8f0", 'family': "Inter"},
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Add risk label annotation
    fig.add_annotation(
        x=0.5, y=0.1,
        text=f"<b>{level_info['text']} RISK</b>",
        showarrow=False,
        font=dict(size=20, color=level_info["color"])
    )
    
    st.plotly_chart(fig, use_container_width=True)
