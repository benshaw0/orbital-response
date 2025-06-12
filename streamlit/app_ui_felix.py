import streamlit as st
import pandas as pd
import torch
import os
import folium
import requests
import base64
import numpy as np
from pathlib import Path
from PIL import Image
from streamlit_folium import st_folium
from streamlit_image_comparison import image_comparison
from dotenv import load_dotenv

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Local imports
from google_static_map_api import google_api
from mapbox_api import mapbox_api

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# -----------------------------------------------------------------------------
# IMPORT OF YOLO MODEL
# -----------------------------------------------------------------------------

# from ultralytics import YOLO
# model = YOLO('path/to/best.pt')
# results = model.predict('path/to/image.jpg')

# -----------------------------------------------------------------------------
# DATA & CONSTANTS
# -----------------------------------------------------------------------------
CSV_PATH = os.path.join(os.path.dirname(__file__), "conflicts.csv")
conflict_data = pd.read_csv(CSV_PATH)

warzone = {
    "-": ["-"],
    "Europe": ["-", "Spain", "Ukraine", "Turkey"],
    "Middle East": ["-", "Lebanon", "Syria", "Israel", "Iran", "Yemen"],
    "Africa": ["-", "Libya", "Sudan"]
}

region_centers = {
    "-": (0, 0),
    "Middle East": (29.0, 45.0),
    "Europe": (54.5, 15.3),
    "Asia": (34.0, 100.0),
    "Africa": (1.0, 20.0),
    "South America": (-15.6, -60.0)
}

# -----------------------------------------------------------------------------
# STREAMLIT CONFIG & GLOBAL STYLE
# -----------------------------------------------------------------------------

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
        /* Reduce left/right padding */
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        /* Images & widgets take full width of their parent */
        .element-container img { width: 100% !important; height: auto !important; }
        /* Tighten vertical spacing between elements */
        .element-container { margin-bottom: 0rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------

st.sidebar.title("Orbital Response Damage Prediction")
st.sidebar.header("Choose Region or Search")

region = st.sidebar.selectbox("Select a Region", list(region_centers.keys()))
st.sidebar.markdown("**Or search for a city/region below:**")

# -------------------- SEARCH BAR (GOOGLE GEOCODING) ---------------------------
search_query = st.sidebar.text_input("Search Location")
search_coords = None

if search_query:
    try:
        params = {"address": search_query, "key": GOOGLE_API_KEY}
        response = requests.get("https://maps.googleapis.com/maps/api/geocode/json", params=params)
        data = response.json()

        if data["status"] == "OK":
            result = data["results"][0]
            lat = result["geometry"]["location"]["lat"]
            lon = result["geometry"]["location"]["lng"]
            formatted_address = result["formatted_address"]
            search_coords = (lat, lon)
            st.sidebar.success(f"Found: {formatted_address}")
        else:
            st.sidebar.error(f"Geocoding failed: {data['status']}")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# -------------------- TEAM PROFILES ------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("Team")

team_profiles = [
    {"name": "Ben Shaw", "url": "https://www.linkedin.com/in/bencshaw/", "img": "images/Ben.png"},
    {"name": "Felix Pless", "url": "https://www.linkedin.com/in/felixpless/", "img": "images/felix.jpg"},
    {"name": "Christian Miro", "url": "https://www.linkedin.com/in/christianmiro/", "img": "images/Christian.jpeg"},
    ]

def make_avatar_tag(path: Path, width: int = 120) -> str:
    """Return a base64-embedded circular avatar suitable for HTML."""
    img_format = "png" if path.suffix.lower() == ".png" else "jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<img src="data:image/{img_format};base64,{b64}" '
        f'style="width:{width}px;height:{width}px;border-radius:50%;display:block;margin:auto;" />'
    )

for member in team_profiles:
    p = Path(member["img"])
    if p.exists():
        avatar_html = make_avatar_tag(p, width=120)
        st.sidebar.markdown(
            f'<a href="{member["url"]}" target="_blank">{avatar_html}'
            f'<p style="text-align:center;margin-top:0.4rem;">{member["name"]}</p></a>',
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(f'- [{member["name"]}]({member["url"]})')
# -----------------------------------------------------------------------------
# MAP SECTION
# -----------------------------------------------------------------------------

# Determine map center
if search_coords:
    lat, lon = search_coords
    map_title = f"Map of: {search_query}"
elif region != "-":
    lat, lon = region_centers[region]
    map_title = f"Map of {region}"
else:
    st.warning("Please select a region or search for a location.")
    st.stop()

# Wrap map into centred column layout
left_spacer, map_col, right_spacer = st.columns([1, 2, 1])

with map_col:
    st.subheader(map_title)
    m = folium.Map(location=[lat, lon], zoom_start=6 if search_coords else 4)
    folium.LatLngPopup().add_to(m)
    map_data = st_folium(m, width=700, height=500)

# -----------------------------------------------------------------------------
# HANDLE MAP CLICK & SHOW SATELLITE IMAGES
# -----------------------------------------------------------------------------

if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]
    st.success(f"Clicked Location: Latitude: {clicked_lat:.5f}, Longitude: {clicked_lon:.5f}")

    # Store for later use
    st.session_state["lat"] = clicked_lat
    st.session_state["lon"] = clicked_lon

    # Fetch satellite imagery from APIs
    mapbox_api(clicked_lat, clicked_lon)
    google_api(clicked_lat, clicked_lon)

    # Paths to the locally saved images
    pre_image = "images/pre_disaster.png"
    post_image = "images/post_disaster.png"

    # Two columns right underneath the map with a SMALL gap to minimise whitespace
    col1, col2 = st.columns([1, 1], gap="small")

    with col1:
        if os.path.exists(pre_image) and os.path.exists(post_image):
            st.subheader("Pre vs Post Disaster")
            image_comparison(
                img1=pre_image,
                img2=post_image,
                label1="Pre-Disaster",
                label2="Post-Disaster",
                width=700,
            )
        else:
            st.warning("Both images must be available for comparison.")

    with col2:
        if os.path.exists(post_image):
            st.subheader("Post-Disaster Focus")
            image_comparison(
                img1=post_image,
                img2=post_image,
                label1="Post-Disaster",
                label2="Mask",
                width=700,
            )
        else:
            st.error("Post-disaster image not found. Check your API or connection.")
    # -----------------------------------------------------------------------------
    # MODEL INFERENCE SECTION
    # -----------------------------------------------------------------------------
    # post_image = "images/post_disaster.png"
    # img=mpimg.imread(post_image)
    # from ultralytics import YOLO
    # model = YOLO('yolo11n.pt')
    # results = model.predict(post_image)
    # st.write(results)
