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

from dotenv import load_dotenv  # Make sure this is imported
# ✅ Load environment variables before any getenv call
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

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
    "Europe": ["-", "Ukraine", "Turkey"],
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
    if country_shape:
        #col1, col2 = st.columns([1, 1], gap="medium")
#
#
# 1st Column
#
#
        #with col1:
            map_height = 550
            map_width = 550
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
            click_data = st_folium(user_map, width=map_width, height=map_height, returned_objects=["last_clicked"])
#
#
# 2nd Column
#
#
        #with col2:
            if click_data and click_data["last_clicked"]:
                lat = click_data["last_clicked"]["lat"]
                lon = click_data["last_clicked"]["lng"]
                st.session_state["lat"] = lat
                st.session_state["lon"] = lon

                mapbox_api(lat, lon)

                image_path = "streamlit/pre_disaster.png"
                if os.path.exists(image_path):
                    st.image(image_path, width=800, clamp=True)
    #            else:
    #                 st.error("Satellite image could not be loaded. Check your API key or network connection.")
    #         else:
    #             st.markdown("### Satellite image will appear here after map interaction.")
    # else:
    #     st.error("Could not find shape data for this country.")
    
    # st.success("WHATEVER YOU LITTLE BITCH")

            if click_data and click_data["last_clicked"]:
                lat = click_data["last_clicked"]["lat"]
                lon = click_data["last_clicked"]["lng"]
                st.session_state["lat"] = lat
                st.session_state["lon"] = lon

                mapbox_api(lat, lon)

                image_path = "streamlit/pre_disaster.png"
                if os.path.exists(image_path):
                    st.image(image_path, width=800, clamp=True)
                else:
                    st.error("Satellite image could not be loaded. Check your API key or network connection.")
            else:
                st.markdown("### Satellite image will appear here after map interaction.")
    else:
        st.error("Could not find shape data for this country.")