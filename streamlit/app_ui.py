import streamlit as st
import folium
import requests
import pandas as pd
from geopy.distance import geodesic
from streamlit_folium import st_folium

# Load conflict data
conflict_data = pd.read_csv("conflicts.csv")

st.set_page_config(layout="wide")
st.title("Orbital Response Damage Predictor")
st.text("This is our Website created by Ben, Christian & Felix")

# Region-country dictionary
warzone = {
    "-": ["-"],
    "Europe": ["-", "Ukraine", "Turkey"],
    "Middle East": ["-", "Lebanon", "Syria", "Israel", "Iran", "Yemen"],
    "Africa": ["-", "Libya", "Sudan"]
}

# Step 1: Select Region and Country
region = st.selectbox("Select your region", list(warzone.keys()))
country = st.selectbox("Select your country", warzone[region])

if region == "-" and country == "-":
    st.warning("Please select a region and a country to view data.")
elif country == "-":
    st.warning("Please select a country to view data.")

if region != "-" and country != "-":
    st.success(f"Your Damage Predictor will be based on: ***{country}*** in ***{region}***")

# Get coordinates from API 
    api_url = f"https://restcountries.com/v3.1/name/{country}?fullText=true"
    response = requests.get(api_url)
# Fetching the API Data
    if response.status_code == 200:
        data = response.json()
        center_lat, center_lon = data[0]["latlng"]
# / Load and show Map of selected Country
        # Load country border GeoJSON
        geojson_url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
        geojson_data = requests.get(geojson_url).json()

        country_shape = next(
            (f for f in geojson_data["features"] if f["properties"]["name"].lower() == country.lower()),
            None
        )

        col1, col2 = st.columns([1, 1], gap="medium")

        with col1:
            st.markdown("### Step 1: Click your location")
            user_map = folium.Map(location=[center_lat, center_lon], zoom_start=6)
            folium.GeoJson(country_shape, style_function=lambda x: {
                'fillOpacity': 0,
                'color': 'red',
                'weight': 3
            }).add_to(user_map)
            user_map.add_child(folium.LatLngPopup())
            click_data = st_folium(user_map, width="100%", height=500, returned_objects=["last_clicked"])

        with col2:
            st.markdown("### Step 2: Nearest Conflict Summary")
            if click_data and click_data["last_clicked"]:
                lat = click_data["last_clicked"]["lat"]
                lon = click_data["last_clicked"]["lng"]

                # Calculate distances
                def calc_dist(row):
                    return geodesic((lat, lon), (row["latitude"], row["longitude"])).km

                conflict_data["distance"] = conflict_data.apply(calc_dist, axis=1)
                nearest = conflict_data.loc[conflict_data["distance"].idxmin()]

                # Conflict description block
                conflict_html = f"""
                <div style='background-color:#262730;padding:15px;border-radius:10px'>
                    <h5 style='color:#eee;margin-bottom:10px;'>⚠️ {nearest['name']}</h5>
                    <p style='color:#aaa;margin:0;'>📍 Distance: {nearest['distance']:.1f} km</p>
                    <p style='color:#aaa;margin-top:4px;'>📝 {nearest['description']}</p>
                </div>
                """
                st.markdown(conflict_html, unsafe_allow_html=True)

                # Then show second map with red/green markers
                st.markdown("### Step 3: Click inside conflict area")
                conflict_map = folium.Map(location=[lat, lon], zoom_start=6)
                folium.Marker([lat, lon], tooltip="Your Click", icon=folium.Icon(color='green')).add_to(conflict_map)
                folium.Marker([nearest["latitude"], nearest["longitude"]],
                            tooltip=nearest["name"],
                            icon=folium.Icon(color='red')).add_to(conflict_map)
                conflict_map.add_child(folium.LatLngPopup())
                second_click = st_folium(conflict_map, width="100%", height=500, key="conflict_map", returned_objects=["last_clicked"])

            # second_click = None
            # if click_data and click_data["last_clicked"]:
            #     lat = click_data["last_clicked"]["lat"]
            #     lon = click_data["last_clicked"]["lng"]

            #     def calc_dist(row):
            #         return geodesic((lat, lon), (row['latitude'], row['longitude'])).kilometers

            #     conflict_data["distance"] = conflict_data.apply(calc_dist, axis=1)
            #     nearest = conflict_data.loc[conflict_data["distance"].idxmin()]

            #     st.markdown("### Step 2: Click inside conflict area")
            #     conflict_map = folium.Map(location=[lat, lon], zoom_start=6)
            #     folium.Marker([lat, lon], tooltip="Your Click", icon=folium.Icon(color='green')).add_to(conflict_map)
            #     folium.Marker(
            #         [nearest["latitude"], nearest["longitude"]],
            #         tooltip=nearest["name"],
            #         icon=folium.Icon(color='red')
            #     ).add_to(conflict_map)
            #     conflict_map.add_child(folium.LatLngPopup())

            #     second_click = st_folium(conflict_map, width="100%", height=500, key="second_map", returned_objects=["last_clicked"])

        # # 🧾 Conflict Summary
        # if click_data and click_data["last_clicked"]:
        #     st.markdown("### Nearest Conflict Summary")
        #     conflict_html = f"""
        #         <div style='background-color:#262730;padding:15px;border-radius:10px;'>
        #             <h5 style='color:#eee;margin-bottom:10px;'>{nearest['name']}</h5>
        #             <p style='color:#aaa;margin:0;'>Distance: {nearest['distance']:.1f} km</p>
        #             <p style='color:#aaa;margin-top:4px;'>{nearest['description']}</p>
        #         </div>
        #     """
        #     st.markdown(conflict_html, unsafe_allow_html=True)

        # 🛰️ Step 3: Satellite image of clicked conflict spot
        if second_click and second_click["last_clicked"]:
            sec_lat = second_click["last_clicked"]["lat"]
            sec_lon = second_click["last_clicked"]["lng"]

            st.markdown("### Step 3: Satellite Image (~600m x 600m)")
            sat_map = folium.Map(location=[sec_lat, sec_lon], zoom_start=18)

            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery",
                name="Esri Satellite",
                overlay=True,
                control=True
            ).add_to(sat_map)

            folium.Marker([sec_lat, sec_lon], tooltip="Selected Point").add_to(sat_map)
            st_folium(sat_map, width="100%", height=500, key="sat_map")

        # 📥 Step 4: Input
        st.markdown("### Step 4: Prediction Values")
        attack = st.number_input("Enter attack value", value=10000.0)
        no_attack = st.number_input("Enter no_attack value", value=287.738)

    else:
        st.error("Could not retrieve country coordinates.")
