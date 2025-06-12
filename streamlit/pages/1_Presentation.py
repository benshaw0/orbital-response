import streamlit as st
from streamlit_image_comparison import image_comparison
from PIL import Image
import base64

# Set page configuration
st.set_page_config(page_title="Orbital Response - Presentation", layout="wide")

# Define the slides
slides = [
        {
        "title": "The Initiative",
        "content": """
        <h3 style='color:#004080;'>Crisis Response CNN</h3>
        <p style='font-size:18px;'>
        <i>Harnessing the power of CNNs and high-frequency satellite imagery to revolutionize humanitarian aid</i>
        </p>

        <hr style='border:1px solid #DDD;'>

        <p style='font-size:18px;'>
        <b>As of <span style='color:#004080;'>02 May 2025</span>:</b><br><br>
        <b><span style='color:#004080;'>174,486</span></b> damaged buildings in Gaza<br>
        <b><span style='color:#004080;'>70%</span></b> of all structures in the Gaza Strip<br>
        <b><span style='color:#004080;'>1.9 million</span></b> people displaced
        </p>

        <p style='font-size:18px;'>
        <b>Air strikes are <span style='color:#004080;'>sporadic</span> and <span style='color:#004080;'>unpredictable</span></b>
        </p>

        <hr style='border:1px solid #DDD;'>

        <p style='font-size:16px;'>
        <b>Data-driven, optimised humanitarian response required</b>
        </p>

        <p style='font-size:22px; font-weight:bold; background-color:#d6f5d6; padding:10px; border-radius:5px; display:inline-block;'>
        SATELLITE IMAGERY
        </p>
        """,
        "image_path": "presentation_images/initiative_image.jpg"
    },
    {
        "title": "Primary Data Source",
        "content": """
        **xBD Dataset**

        *The largest, manually labelled, building damage assessment dataset*

         **18,336** high-resolution satellite images
         **850,000** polygons covering over **45,000 km²**

        Covers **six types** of natural disasters worldwide:
        - **Earthquakes**
        - **Hurricanes**
        - **Monsoons**
        - <span style='color:grey;'>Wildfires</span>
        - <span style='color:grey;'>Volcanic Eruptions</span>
        - <span style='color:grey;'>Floods</span>

        ---

        **Damage Classification:**
        - **0** — un-classified
        - **1** — no-damage
        - **2** — minor-damage
        - **3** — major-damage
        - **4** — destroyed
        """,
        "image_paths": [
            "presentation_images/primary_data_pre.png",
            "presentation_images/primary_data_post.png"
        ]
    },
    {
        "title": "Secondary Data Source",
        "content": """
        Intended to calibrate the models with a manually labelled dataset from Gaza.

        Six locations deliberately selected (shown below), targeting urban areas with high levels of building destruction.

        150 images labelled with **Roboflow**:

        - undamaged - 0
        - damaged - 1
        """,
        "image_static": "presentation_images/secondary_image_static.png",
        "image_slider_before": "presentation_images/secondary_image_unlabelled.png",
        "image_slider_after": "presentation_images/secondary_image_labelled.png"
    },
    {
        "title": "The Models - U-Net",
        "content": """
        **Semantic Segmentation** with a **U-Net CNN**

        - **Pretrained Encoder** *(ResNet18)* — Feature compression
        - **Trained Decoder** — Segmentation map

        Total Number of Parameters: **25,383,347**

        ---

        **Baseline approach:** Building localisation and damage classification in a single U-Net model

        **Final approach:** Separate localisation and damage classification models
        """,
        "image_path": "presentation_images/unet_model_architecture.png"
    },
    {
        "title": "The Models - YOLOv8-seg",
        "content": """
        **YOLOv8** family by **Ultralytics**, tailored for **instance segmentation**

        - **Pretrained** on the COCO dataset (80+ object classes)
        - **Fine-Tuned** to Gaza with the manually labelled secondary dataset

        Number of Parameters: **~11 million**
        """,
        "image_path": "presentation_images/yolo_model_img.png"
    }
]

# Sidebar for navigation
st.sidebar.title("Presentation Slides")
selected_title = st.sidebar.radio("Navigate to:", [slide["title"] for slide in slides])
st.session_state.slide_index = [i for i, s in enumerate(slides) if s["title"] == selected_title][0]

# Get the current slide
current_slide = slides[st.session_state.slide_index]

# --- Slide Header as Title ---
st.title(current_slide['title'])

# Display layout depending on slide content
if current_slide['title'] == "The Initiative" and "image_path" in current_slide:
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.markdown(current_slide["content"], unsafe_allow_html=True)
    with right_col:
        try:
            st.markdown("""
                <style>
                    .slide-image-container {
                        display: flex;
                        align-items: flex-start;
                        margin-top: -3.5rem;
                    }
                    .slide-image-container img {
                        margin-top: 0;
                        border-radius: 0 !important;
                    }
                </style>
                <div class='slide-image-container'>
            """, unsafe_allow_html=True)
            st.image(current_slide["image_path"], use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning(f"Image not found: {current_slide['image_path']}")
elif current_slide['title'] == "Primary Data Source" and "image_paths" in current_slide:
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.markdown(current_slide["content"], unsafe_allow_html=True)
    with right_col:
        for img_path in current_slide["image_paths"]:
            try:
                st.image(img_path, use_container_width=True)
            except FileNotFoundError:
                st.warning(f"Image not found: {img_path}")
elif current_slide['title'] == "Secondary Data Source":
    st.markdown("""
        <style>
            .side-by-side-container {
                display: flex;
                justify-content: space-between;
                gap: 2rem;
            }
            .side-by-side-container .img-box {
                flex: 1;
            }
            .element-container .image-comparison__img-wrapper img,
            .element-container img {
                border-radius: 0 !important;
                display: block;
                width: 100%;
                height: auto;
            }
        </style>
    """, unsafe_allow_html=True)
    st.markdown(current_slide["content"], unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        try:
            st.image(current_slide["image_static"], use_container_width=True)
        except FileNotFoundError:
            st.warning(f"Static image not found: {current_slide['image_static']}")
    with col2:
        try:
            image_comparison(
                img1=current_slide["image_slider_before"],
                img2=current_slide["image_slider_after"],
                label1="Unlabelled",
                label2="Labelled"
            )
        except FileNotFoundError:
            st.warning("Comparison images not found.")
elif "image_path" in current_slide:
    left_col, right_col = st.columns([1.5, 1.5])
    with left_col:
        st.markdown(current_slide["content"], unsafe_allow_html=True)
    with right_col:
        try:
            st.markdown("""
                <style>
                    .element-container img {
                        border-radius: 0 !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            st.image(current_slide["image_path"], use_container_width=True)
        except FileNotFoundError:
            st.warning(f"Image not found: {current_slide['image_path']}")
else:
    st.markdown(current_slide["content"], unsafe_allow_html=True)
