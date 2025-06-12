import streamlit as st
import base64
from PIL import Image
import os
from pathlib import Path


st.set_page_config(
    page_title="Orbital Response - Home",
    page_icon="🌍",
    layout="wide"
)

# Header
st.markdown("<h1 style='text-align: center'>Orbital Response</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center'>Optimising Humanitarian Aid Distribution with AI</h3>", unsafe_allow_html=True)

st.write("")
st.markdown("---")

st.markdown("""<p style='text-align: center'>
Our Mission: Harness the power of CNNs and high-frequency satellite imagery to revolutionize humanitarian aid.""", unsafe_allow_html=True)
st.markdown("""<p style='text-align: center'>
High-quality, high-frequency satellite imagery is a novel resource. </p>""", unsafe_allow_html=True)

st.markdown("<p style='text-align: center'>INSERT TEXT HERE</p>", unsafe_allow_html=True)
st.write("")

st.markdown("---")
st.markdown("<h3 style='text-align:center; color:#004080;'>The Team</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

team_profiles = [
    {"name": "Ben Shaw", "url": "https://www.linkedin.com/in/bencshaw/", "img": "presentation_images/Ben.png"},
    {"name": "Felix Pless", "url": "https://www.linkedin.com/in/felixpless/", "img": "presentation_images/felix.jpg"},
    {"name": "Christian Miro", "url": "https://www.linkedin.com/in/christianmiro/", "img": "presentation_images/Christian.jpeg"},
]

def make_avatar_tag(path: Path, width: int = 120) -> str:
    """Return a base64-embedded circular avatar suitable for HTML."""
    img_format = "png" if path.suffix.lower() == ".png" else "jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<img src="data:image/{img_format};base64,{b64}" '
        f'style="width:{width}px;height:{width}px;border-radius:50%;display:block;margin:auto;" />'
    )

cols = st.columns(len(team_profiles))
for col, member in zip(cols, team_profiles):
    p = Path(member["img"])
    with col:
        if p.exists():
            avatar_html = make_avatar_tag(p, width=140)
            st.markdown(
                f"""
                <a href="{member["url"]}" target="_blank">
                    {avatar_html}
                    <p style="text-align:center; margin-top:0.5rem;">{member["name"]}</p>
                </a>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f'<p style="text-align:center;"><a href="{member["url"]}">{member["name"]}</a></p>', unsafe_allow_html=True)
