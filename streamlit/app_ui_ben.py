import streamlit as st
import numpy as np
import pandas as pd

import sys
import os
from pathlib import Path

import folium
import requests
from dotenv import load_dotenv

import torch
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image

from PIL import Image

from streamlit_folium import st_folium
from streamlit_image_comparison import image_comparison

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from ultralytics import YOLO

# Importing Google API
from goog_static_map_api import google_api

# Importing MapBox API
from mapbox_api import mapbox_api

#Importing img_transform and model module
from ml_logic.model.model import get_model, get_model_destruction

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Dictionary of Warzones

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

mapbox_api(30, 31)


if map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]
    st.success(f"Clicked Location: Latitude: {clicked_lat:.5f}, Longitude: {clicked_lon:.5f}")

    # if st.button("Fetch Satellite Images"):
    #     mapbox_api(clicked_lat, clicked_lon)
    #     google_api(clicked_lat, clicked_lon)

    st.session_state["lat"] = clicked_lat
    st.session_state["lon"] = clicked_lon

    st.write("lat")

    st.write(st.session_state["lat"])

    st.write("lon")

    st.write(st.session_state["lon"])


    # Fetch satellite imagery from APIs
    #google_api(st.session_state["lat"], st.session_state["lon"])

    mapbox_api(st.session_state["lat"], st.session_state["lon"])


    # Paths to the locally saved images

    image_dir = Path(__file__).resolve().parent / "images_masks" / "satellite_images"
    pre_image_path = image_dir / "pre_disaster.png"
    post_image_path = image_dir / "post_disaster.png"

    # Two columns right underneath the map with a SMALL gap to minimise whitespace
    col1, col2 = st.columns([1, 1], gap="small")

if st.button('show me'):
    with col1:
        if os.path.exists(pre_image_path) and os.path.exists(post_image_path):
            st.subheader("Pre vs Post Disaster")
            image_comparison(
                img1=str(pre_image_path),
                img2=str(post_image_path),
                label1="Pre-Disaster",
                label2="Post-Disaster",
                width=700,
            )
        else:
            st.warning("Both images must be available for comparison.")

    with col2:
        if os.path.exists(post_image_path):
            st.subheader("Post-Disaster Focus")
            image_comparison(
                img1=str(post_image_path),
                img2=str(post_image_path),
                label1="Post-Disaster",
                label2="Mask",
                width=700,
            )
        else:
            st.error("Post-disaster image not found. Check your API or connection.")


    #     st.markdown("---")
    #     st.info("Generating model prediction for the selected location...")

    device = "cuda" if torch.cuda.is_available() else "cpu"


    def load_models():
        model = get_model()
        model_destruction = get_model_destruction()
        model.load_state_dict(torch.load("models/Gaza_1st_BestModel_70.pth", map_location=torch.device("cpu")))
        model_destruction.load_state_dict(torch.load("models/AllData_v1_Destruction_Model_37.pth", map_location=torch.device("cpu")))
        model.eval()
        return model, model_destruction

    pre_img = Image.open(pre_image_path).convert("RGB")
    post_img = Image.open(post_image_path).convert("RGB")

    # Resize
    resize = T.Resize((1024, 1024))
    pre_img = resize(pre_img)
    post_img = resize(post_img)

    # Tensor
    pre_tensor = T.ToTensor()(pre_img)
    post_tensor = T.ToTensor()(post_img)

    # Concat and normalize
    input_tensor = torch.cat([pre_tensor, post_tensor], dim=0)
    normalize = T.Normalize(mean=[0.485]*6, std=[0.229]*6)
    input_tensor = normalize(input_tensor).unsqueeze(0).to(device)

    #Loading the models
    model, model_destruction = load_models()
    model = model.to(device)
    model_destruction = model_destruction.to(device)

    # Inference - model_buildings
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)  # [1, 1, H, W]
        probs = torch.sigmoid(output)[0, 0]  # [H, W]
        building_mask = (probs > 0.6).int()  #Adjust the threshold here if you need to

    mask_img = to_pil_image(torch.tensor(building_mask))
    building_mask_output_dir = "images_masks/masks"
    mask_img.save(os.path.join(building_mask_output_dir, "building_mask.png"))

    #Running - model_destruction

    building_mask_path = "images_masks/masks/building_mask.png"

    def predict_and_show_from_paths(model_destruction, pre_path, post_path, building_mask_path, device):
        model_destruction.eval()
        pre = np.array(Image.open(pre_path).convert("RGB").resize((224, 224)))
        post = np.array(Image.open(post_path).convert("RGB").resize((224, 224)))
        building = np.array(Image.open(building_mask_path).resize((224, 224), resample=Image.NEAREST))
        building_mask_binary_for_input = (building > 127).astype(np.float32)[..., np.newaxis]
        building_mask_original_binary_for_dice = (building > 127).astype(np.float32)
        image = np.concatenate([pre, post, building_mask_binary_for_input], axis=-1)  #[H, W, 7]
        mean = np.array([0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.5])
        std = np.array([0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.25])
        image_normalized = (image.astype(np.float32) / 255.0 - mean) / std
        image_tensor = torch.tensor(image_normalized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        # Inference
        with torch.no_grad():
            output = model_destruction(image_tensor)  # [1, 1, H, W]
            probs = torch.sigmoid(output.squeeze(1))
            pred_mask_raw = (probs > 0.5).float().squeeze().cpu().numpy()

        pred_mask_final = pred_mask_raw * building_mask_original_binary_for_dice

        mask_img = to_pil_image(torch.tensor(pred_mask_final))
        final_mask_output_dir = "images_masks/masks"
        mask_img.save(os.path.join(final_mask_output_dir, f"final_mask.png"))

        return mask_img

    predict_and_show_from_paths(model_destruction, pre_image_path, post_image_path, building_mask_path, device)

    #YOLO Model

    model = YOLO("models/yolo_50ep.pt")
    results = model.predict(source=post_image_path, save=False)

    results[0].save(filename="temp_yolo_output.jpg")

    output_path = Path("images_masks/masks/yolo_output.png")
    img = Image.open("temp_yolo_output.jpg").convert("RGB")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

    yolo_mask_path = "images_masks/masks/yolo_output.png"

    #OUTPUTTING AN IMAGE

    final_mask_path = "images_masks/masks/final_mask.png"

    #PRODUCING THE FINAL OUTPUT

    #Visualising the two masks on the interface:

    building_mask_img = Image.open(building_mask_path).convert("RGB")
    destruction_mask_img = Image.open(final_mask_path).convert("RGB")
    yolo_mask = Image.open(yolo_mask_path)

    st.image(building_mask_img)
    st.image(destruction_mask_img)
    st.image(yolo_mask)

    # CLASS_COLORS = {
    #     0: (0, 0, 0),         # un-classified (black)
    #     1: (0, 255, 0),       # no-damage (green)
    #     2: (0, 0, 255),       # minor-damage (blue)
    #     3: (255, 165, 0),     # major-damage (orange)
    #     4: (255, 0, 0),       # destroyed (red)
    # }
    # def mask_to_rgb(mask_tensor):
    #     mask_np = mask_tensor.cpu().numpy()
    #     h, w = mask_np.shape
    #     rgb = np.zeros((h, w, 3), dtype=np.uint8)
    #     for class_idx, color in CLASS_COLORS.items():
    #         rgb[mask_np == class_idx] = color
    #     return Image.fromarray(rgb)

    # with torch.no_grad():
    #     output = model(input_tensor)  # [1, 5, H, W]
    #     pred_mask = torch.argmax(output, dim=1)[0]  # [H, W]

    # rgb_mask = mask_to_rgb(pred_mask)

    # original_img = post_img.resize((224, 224))

    # blended_img = Image.blend(original_img, rgb_mask, alpha=0.5)

    # st.image(blended_img, width=800, clamp=True)
