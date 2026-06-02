import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from modules.public_awareness import get_flu_info, get_preventive_measures, get_vaccination_guidance, get_emergency_contacts

st.set_page_config(page_title="Public Awareness | Epi Predict", layout="wide")

from dashboard.components.ui_helper import inject_custom_css
inject_custom_css()

st.title("Public Health Awareness")
st.markdown("Essential medical information, prevention guidelines, and emergency contacts curated from the **World Health Organization (WHO)** and other health authorities.")

col1, col2 = st.columns([1, 1])

flu_info = get_flu_info()

with col1:
    st.subheader("About Influenza")
    
    st.markdown(f"""
    <div class="epi-card">
        <p style="color: #cbd5e1; font-size: 1rem; line-height: 1.6;">{flu_info['overview']}</p>
        <hr style="border-color: #334155;">
        <h5 style="color: #00d4aa; margin-top: 10px;">Transmission</h5>
        <ul style="color: #cbd5e1;">
            {''.join([f"<li>{item}</li>" for item in flu_info['transmission']['primary_routes']])}
        </ul>
        <p style="color: #94a3b8; font-size: 0.9rem;"><em>{flu_info['transmission']['survival_on_surfaces']}</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Common Symptoms")
    symptoms_html = "<div class='epi-card'><ul style='color: #cbd5e1;'>"
    for sym in flu_info["symptoms"]["common"]:
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
        <p style="color: #cbd5e1; margin-bottom: 1rem;"><b>Overview:</b> {vax_info['overview']}</p>
        <p style="color: #cbd5e1; margin-bottom: 1rem;"><b>Eligibility:</b> Recommended for {vax_info['eligibility']['recommended_for'][0].lower()} and other high-risk individuals.</p>
        <p style="color: #cbd5e1; font-size: 0.9rem;"><i>Note: {vax_info['side_effects']['note']}</i></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

col_prev, col_emerg = st.columns([2, 1])

with col_prev:
    st.subheader("Everyday Preventive Actions (WHO Guidelines)")
    prev_measures = get_preventive_measures()
    
    cols = st.columns(2)
    for i, item in enumerate(prev_measures):
        col_idx = i % 2
        with cols[col_idx]:
            st.markdown(f"""
            <div class="epi-card" style="padding: 1rem; display: flex; align-items: start; min-height: 120px; margin-bottom: 10px;">
                <div style="font-size: 1.5rem; margin-right: 1rem; color: #00d4aa; padding-top: 0.2rem;">🛡️</div>
                <div>
                    <strong style="color: #f8fafc; font-size: 1rem;">{item['measure']}</strong>
                    <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.3rem;">{item['description']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

with col_emerg:
    st.subheader("Emergency Contacts & Resources")
    contacts = get_emergency_contacts()
    
    contact_html = "<div class='epi-card'>"
    for org in contacts.get("international", []):
        contact_html += f"<p style='margin-bottom: 1.5rem;'><strong style='color:#00d4aa;'>{org['organisation']}</strong><br><span style='color: #cbd5e1; font-size: 0.9rem;'>{org['description']}</span><br><a href='{org['website']}' target='_blank' style='color: #38bdf8; text-decoration: none; font-size: 0.9rem;'>Visit Website ↗</a></p>"
        
    contact_html += "<hr style='border-color: #334155;'><h5 style='color: #f59e0b;'>Helplines</h5>"
    for help in contacts.get("helplines", []):
        contact_html += f"<p style='margin-bottom: 1rem;'><strong style='color:#f8fafc;'>{help['name']}</strong><br><span style='color: #f59e0b; font-size: 1.2rem; font-weight: bold;'>{help['number']}</span><br><span style='color: #94a3b8; font-size: 0.8rem;'>{help['description']}</span></p>"
        
    contact_html += "</div>"
    st.markdown(contact_html, unsafe_allow_html=True)
