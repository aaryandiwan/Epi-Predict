import base64
import streamlit as st
from pathlib import Path

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def inject_custom_css():
    """Injects custom CSS and the base64 background image into the app."""
    # Read the custom.css file
    css_path = Path(__file__).resolve().parent.parent / "styles" / "custom.css"
    css_content = ""
    if css_path.exists():
        with open(css_path, "r") as f:
            css_content = f.read()

    # Read and encode the background image
    bg_path = Path(__file__).resolve().parent.parent / "assets" / "epidemic_bg.png"
    bg_css = ""
    if bg_path.exists():
        img_base64 = get_base64_of_bin_file(bg_path)
        bg_css = f"""
        .stApp {{
            background: linear-gradient(rgba(10, 22, 40, 0.85), rgba(10, 22, 40, 0.85)), url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        """

    # Inject everything
    st.markdown(f"<style>{css_content}\n{bg_css}</style>", unsafe_allow_html=True)
