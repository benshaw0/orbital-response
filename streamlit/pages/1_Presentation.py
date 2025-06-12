import streamlit as st

# Set page configuration
st.set_page_config(page_title="Orbital Response - Presentation", layout="wide")

st.title("Orbital Response")

# Define the slides
slides = [
    {
        "title": "The Initiative",
        "content": """
        #### **Crisis Response CNN**
        *Harnessing the power of CNNs and high-frequency satellite
        imagery to revolutionize humanitarian aid.*

        --_

        *As of 02 May 2025:*

        - **174,486** damaged buildings in Gaza
        - **70%** of the total structures in the Gaza Strip
        - **1.9 million** people displaced

        Air strikes are **sporadic** and **unpredictable**

        ---

        **Data-driven, optimised humanitarian response requires:**

        ## **SATELLITE IMAGERY**
        """,
        # "image_path": "images/initiative_slide.png"
    },
    {
        "title": "Primary Data Source",
        "content": "Explain your use of xView2 data, damage classes, and masks...",
        # "image_path": "images/xview2_example.png"
    },
    {
        "title": "Secondary Data Source",
        "content": "Describe use of Google Static Maps, Mapbox imagery, and Roboflow annotations...",
        # "image_path": "images/gaza_grids.png"
    },
    {
        "title": "The Models - U-Net",
        "content": "Detail the U-Net model with ResNet34 encoder, input channels, and output segmentation...",
        # "image_path": "images/unet_architecture.png"
    },
    {
        "title": "The Models - YOLOv8-seg",
        "content": "Explain the YOLOv8-segmentation model used for instance-based detection...",
        # "image_path": "images/yolo_output.png"
    }
]


# Set or get the current slide index
if "slide_index" not in st.session_state:
    st.session_state.slide_index = 0

# Get the current slide
current_slide = slides[st.session_state.slide_index]

# --- Slide Content ---
st.markdown(f"## {current_slide['title']}")
st.markdown(current_slide["content"])
st.markdown("---")

# Optional image (if it exists)
if "image_path" in current_slide:
    try:
        st.image(current_slide["image_path"], use_column_width=True)
    except FileNotFoundError:
        st.warning(f"Image not found: {current_slide['image_path']}")

# --- Navigation Buttons ---
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    if st.button("Previous", disabled=st.session_state.slide_index == 0):
        st.session_state.slide_index -= 1

with col3:
    if st.button("Next", disabled=st.session_state.slide_index == len(slides) - 1):
        st.session_state.slide_index += 1
