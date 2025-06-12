import streamlit as st
import base64
from PIL import Image
import os


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

# --- Image Carousel ---
st.markdown("---")

image_files = [
    'images/carousel_image1.jpg',
    'images/carousel_image2.jpg',
    'images/carousel_image3.jpg'
]
for img_path in image_files:
    if not os.path.exists(img_path):
        try:
            dummy_image = Image.new('RGB', (800, 400), color='skyblue')
            dummy_image.save(img_path)
            st.toast(f"Created dummy image: {img_path}", icon="ℹ️")
        except Exception as e:
            st.warning(f"Could not create dummy image {img_path}: {e}")

if 'current_image_index' not in st.session_state:
    st.session_state.current_image_index = 0

if image_files and os.path.exists(image_files[st.session_state.current_image_index]):
    st.image(image_files[st.session_state.current_image_index], use_container_width=True)
elif image_files:
    st.warning(f"Image not found: {image_files[st.session_state.current_image_index]}")

prev_col, _, next_col = st.columns([1, 8, 1])
with prev_col:
    if st.button("⬅️ Previous", use_container_width=True) and image_files:
        st.session_state.current_image_index = (st.session_state.current_image_index - 1) % len(image_files)
        st.rerun()
with next_col:
    if st.button("Next ➡️", use_container_width=True) and image_files:
        st.session_state.current_image_index = (st.session_state.current_image_index + 1) % len(image_files)
        st.rerun()
st.write("")

st.markdown("---")
col1, col2, col3 = st.columns(3)
with col2:
    try:
        st.image('images/LeWagon_logo.png', width=300)
        st.write("Created by LeWagon Data Science, Batch #2012")
    except FileNotFoundError:
        st.info("📷 Image not found")
