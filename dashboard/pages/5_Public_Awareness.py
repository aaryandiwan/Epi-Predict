import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from modules.public_awareness import get_flu_info, get_preventive_measures, get_vaccination_guidance, get_emergency_contacts

st.set_page_config(page_title="Public Awareness | Epi Predict", layout="wide")

# Load CSS
css_path = Path(__file__).resolve().parent.parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Public Health Awareness")
st.markdown("Essential medical information, prevention guidelines, and emergency contacts.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("About Influenza")
    flu_info = get_flu_info()
    
    st.markdown(f"""
    <div class="epi-card">
        <p style="color: #cbd5e1; font-size: 1rem; line-height: 1.6;">{flu_info['description']}</p>
        <p style="color: #cbd5e1; font-size: 1rem; line-height: 1.6;"><b>Transmission:</b> {flu_info['transmission']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Common Symptoms")
    symptoms_html = "<div class='epi-card'><ul style='color: #cbd5e1;'>"
    for sym in flu_info["symptoms"]:
        symptoms_html += f"<li style='margin-bottom: 0.5rem;'>{sym}</li>"
    symptoms_html += "</ul></div>"
    st.markdown(symptoms_html, unsafe_allow_html=True)

with col2:
    st.subheader("High-Risk Groups")
    hr_html = "<div class='epi-card' style='border-left: 4px solid #ef4444;'><ul style='color: #cbd5e1;'>"
    for group in flu_info["high_risk_groups"]:
        hr_html += f"<li style='margin-bottom: 0.5rem;'>{group}</li>"
    hr_html += "</ul><p style='font-size:0.8rem; color:#94a3b8; margin-top:1rem;'>*People in these groups should seek medical attention promptly if symptoms develop.</p></div>"
    st.markdown(hr_html, unsafe_allow_html=True)
    
    st.subheader("Vaccination Guidance")
    vax_info = get_vaccination_guidance()
    st.markdown(f"""
    <div class="epi-card" style="border-left: 4px solid #00d4aa;">
        <p style="color: #cbd5e1; margin-bottom: 1rem;"><b>Recommendation:</b> {vax_info['recommendation']}</p>
        <p style="color: #cbd5e1; margin-bottom: 1rem;"><b>Eligibility:</b> {vax_info['eligibility']}</p>
        <p style="color: #cbd5e1; margin-bottom: 1rem;"><b>Timing:</b> {vax_info['timing']}</p>
        <p style="color: #cbd5e1; font-size: 0.9rem;"><i>{vax_info['effectiveness']}</i></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

col_prev, col_emerg = st.columns([2, 1])

with col_prev:
    st.subheader("Everyday Preventive Actions")
    prev_measures = get_preventive_measures()
    
    cols = st.columns(2)
    for i, measure in enumerate(prev_measures):
        col_idx = i % 2
        with cols[col_idx]:
            st.markdown(f"""
            <div class="epi-card" style="padding: 1rem; display: flex; align-items: center; min-height: 80px;">
                <div style="font-size: 1.5rem; margin-right: 1rem; color: #00d4aa;">🛡️</div>
                <div style="color: #f8fafc; font-size: 0.95rem;">{measure}</div>
            </div>
            """, unsafe_allow_html=True)

with col_emerg:
    st.subheader("Emergency Contacts & Resources")
    contacts = get_emergency_contacts()
    
    contact_html = "<div class='epi-card'>"
    for name, value in contacts.items():
        if value.startswith("http"):
            contact_html += f"<p style='margin-bottom: 1rem;'><strong style='color:#00d4aa;'>{name}:</strong><br><a href='{value}' target='_blank' style='color: #cbd5e1; text-decoration: none;'>Visit Website ↗</a></p>"
        else:
            contact_html += f"<p style='margin-bottom: 1rem;'><strong style='color:#f59e0b;'>{name}:</strong><br><span style='color: #f8fafc; font-size: 1.2rem;'>{value}</span></p>"
    contact_html += "</div>"
    st.markdown(contact_html, unsafe_allow_html=True)
