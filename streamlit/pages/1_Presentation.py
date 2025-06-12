import streamlit as st
from pathlib import Path

# Set page configuration
st.set_page_config(page_title="Orbital Response - Presentation", layout="wide")

st.title("Orbital Response")

# Define the slides
slides = [
    {
        "title": "The Initiative",
        "content": """
            <h3 style='font-size: 28px;'>Crisis Response CNN</h3>
            <p style='font-size: 18px;'>
            <em>Harnessing the power of CNNs and high-frequency satellite imagery to revolutionize humanitarian aid.</em><br><br>

            <strong>As of 02 May 2025:</strong><br>
            - <strong>174,486</strong> damaged buildings in Gaza<br>
            - <strong>70%</strong> of the total structures in the Gaza Strip<br>
            - <strong>1.9 million</strong> people displaced<br><br>

            Air strikes are <strong>sporadic</strong> and <strong>unpredictable</strong><br><br>

            <strong>Data-driven, optimised humanitarian response requires:</strong><br>
            <h2 style='color: red;'>SATELLITE IMAGERY</h2>
            </p>
        """,
        "image": None
    },
    {
        "title": "Primary Data Source",
        "content": """
            <h3 style='font-size: 28px; font-weight: 700;'> xView2 Dataset</h3>
            <p style='font-size: 18px;'>
            <b>Manually annotated</b> with polygons and corresponding damage scores<br><br>

            <b>18,336</b> high-resolution satellite images<br>
            <b>850,000</b> polygons covering over <b>45,000 km²</b><br><br>

            Covers <b>six types</b> of natural disasters worldwide:
            <ul style='font-size: 18px; margin-left: 20px;'>
                <li><b>Earthquakes</b></li>
                <li><b>Hurricanes</b></li>
                <li><b>Monsoons</b></li>
                <li>Wildfires</li>
                <li>Volcanic Eruptions</li>
                <li>Floods</li>
            </ul>

            <hr style='border: 1px solid #ddd;'>

            <b>Damage Classification:</b>
            <ul style='font-size: 18px; margin-left: 20px;'>
                <li><b>0</b> — un-classified</li>
                <li><b>1</b> — no-damage</li>
                <li><b>2</b> — minor-damage</li>
                <li><b>3</b> — major-damage</li>
                <li><b>4</b> — destroyed</li>
            </ul>
            </p>
        """,
        "image": "presentation_images/primary_data_img.png"
    },
    {
        "title": "Secondary Data Source",
        "content": "Describe use of Google Static Maps, Mapbox imagery, and Roboflow annotations...",
        "image": None
    },
    {
        "title": "The Models - U-Net",
        "content": "Detail the U-Net model with ResNet34 encoder, input channels, and output segmentation...",
        "image": None
    },
    {
        "title": "The Models - YOLOv8-seg",
        "content": "Explain the YOLOv8-segmentation model used for instance-based detection...",
        "image": None
    }
]

# Set or get the current slide index
if "slide_index" not in st.session_state:
    st.session_state.slide_index = 0

# Get the current slide
current_slide = slides[st.session_state.slide_index]

# --- Slide Content Layout ---
st.markdown(f"## {current_slide['title']}")

if current_slide["image"]:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(current_slide["content"], unsafe_allow_html=True)
    with col2:
        st.image(current_slide["image"], use_container_width=True)
else:
    st.markdown(current_slide["content"], unsafe_allow_html=True)

st.markdown("---")

# --- Navigation Buttons ---
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    if st.button("Previous", disabled=st.session_state.slide_index == 0):
        st.session_state.slide_index -= 1

with col3:
    if st.button("Next", disabled=st.session_state.slide_index == len(slides) - 1):
        st.session_state.slide_index += 1
