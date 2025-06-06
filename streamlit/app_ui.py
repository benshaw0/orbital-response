import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import folium
import requests

# Importing Google API
from orbital_response.ml_logic.api.goog_static_map_api import google_api

# Importing MapBox API
from orbital_response.ml_logic.api.mapbox_api import mapbox_api

# Importing Bens Model
from WHATEVER import BENS MODEL

from dotenv import load_dotenv

from geopy.distance import geodesic
from streamlit_folium import st_folium



CSV_PATH = os.path.join(os.path.dirname(__file__), "conflicts.csv")
conflict_data = pd.read_csv(CSV_PATH)
#
#
# Page SetUp
#
#
st.set_page_config(layout="wide")
st.title("Orbital Response Damage Predictor")
#
#
# Dictionary of Warzones
# 
# 
warzone = {
    "-": ["-"],
    "Europe": ["-", "Spain","Ukraine", "Turkey"],
    "Middle East": ["-", "Lebanon", "Syria", "Israel", "Iran", "Yemen"],
    "Africa": ["-", "Libya", "Sudan"]
}
#
#
# Select Region and Country
#
#
region = st.selectbox("Select your region", list(warzone.keys()))
country = st.selectbox("Select your country", warzone[region])

if region == "-" and country == "-":
    st.warning("Please select a region and a country to view data.")
elif country == "-":
    st.warning("Please select a country to view data.")

if region != "-" and country != "-":
    st.success(f"Your Damage Predictor will be based on: ***{country}*** in ***{region}***")
#
#
# Get coordinates from REST Countries API using the selected country name
# Load country borders GeoJSON data from GitHub
#
# 
api_url = f"https://restcountries.com/v3.1/name/{country}?fullText=true"
response = requests.get(api_url)

if response.status_code == 200:
    data = response.json()
    center_lat, center_lon = data[0]["latlng"]

    geojson_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    geojson_data = requests.get(geojson_url).json()

    country_shape = next(
        (f for f in geojson_data["features"] if f["properties"]["name"].lower() == country.lower()),
        None
    )
#
#
# Initiating Columns
# 
#    
#if country_shape:
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")
#
#
# 1st Column
#
#
    with col1:
        map_height = 800
        map_width = 800
        user_map = folium.Map(zoom_start=6)
        bounds = folium.GeoJson(country_shape).get_bounds()
        user_map.fit_bounds(bounds)

        folium.GeoJson(
            country_shape,
            style_function=lambda x: {
                'fillOpacity': 0,
                'color': 'red',
                'weight': 3
            }
        ).add_to(user_map)
        user_map.add_child(folium.LatLngPopup())
        click_data = st_folium(user_map, width=map_width, height=map_height, returned_objects=["last_clicked"]) # After clicking the "returned object" of lat and lon is stored in last_clicked
#
#
# 2nd Column
#
#
    with col2:
        if click_data and click_data["last_clicked"]:
            lat = click_data["last_clicked"]["lat"]
            lon = click_data["last_clicked"]["lng"]
            st.session_state["lat"] = lat
            st.session_state["lon"] = lon

            url = mapbox_api(lat, lon)
            image_path = "/home/felix/code/benshaw0/orbital-response/streamlit/pre_disaster.png"
            if os.path.exists(image_path):
                st.image(image_path, width=800, clamp=True)
#
#
# 3rd Column
#
#
    with col3:
            google_api(lat, lon)
            image_path = "/home/felix/code/benshaw0/orbital-response/streamlit/post_disaster.png"
            if os.path.exists(image_path):
                st.image(image_path, width=800, clamp=True)
            else:
                st.error("Satellite image could not be loaded. Check your API key or network connection.")
#
# 
# Input of the Model of Ben / Masking etc.
# 
#                    
    if click_data and click_data["last_clicked"]:
        st.markdown("---")
        st.info("Here is supposed to be displayed the Model of Ben")


        trained_model_1 = UNetModel(n_classes=5).to(device) 
        trained_model_1.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

        BENS FUNCTION(lat, lon)
        concacinated_images = "path"
        st.image(concacinated_images, width=800, clamp=True)


# google_api(lat, lon)
# image_path = "/home/felix/code/benshaw0/orbital-response/streamlit/post_disaster.png"
# if os.path.exists(image_path):
# st.image(image_path, width=800, clamp=True)



###################################################################################################################
#model_pth = '/Users/BenedictShaw/code/benshaw0/orbital-response/orbital_response/models/unet_bs001.pth'
#trained_model_1 = UNetModel(n_classes=5).to(device) /// calls the model
#trained_model_1.load_state_dict(torch.load(model_pth, map_location=torch.device(device))) /// params
######################################################################################################################
###### images for test the model
# pre_path = "/home/felix/code/benshaw0/orbital-response/streamlit/pre_disaster.png"
# post_path = /home/felix/code/benshaw0/orbital-response/streamlit/post_disaster.png"
# pre_img = Image.open(pre_path)
# post_img = Image.open(post_path)
# img_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])
# # Apply transform and concat
# pre_tensor = img_transform(pre_img)
# post_tensor = img_transform(post_img)
# input_tensor = torch.cat([pre_tensor, post_tensor], dim=0).unsqueeze(0).to(device)  # [1, 6, 224, 224]
# output_tensor = trained_model_1(input_tensor)
# ##############################################################################################
# CLASS_COLORS = {
#     0: (0, 0, 0),         # un-classified (black)
#     1: (0, 255, 0),       # no-damage (green)
#     2: (0, 0, 255),     # minor-damage (blue)
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
##################################################
######### Dispalying images
# model.eval()
# with torch.no_grad():
#     output = model(input_tensor)  # [1, 5, H, W]
#     pred_mask = torch.argmax(output, dim=1)[0]  # [H, W]
# # prediction to RGB
# rgb_mask = mask_to_rgb(pred_mask)