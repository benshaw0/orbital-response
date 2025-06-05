# Initialize session state
if 'zoom' not in st.session_state:
    st.session_state.update({
        'center': [9.082, 8.675],
        'zoom': 6,
        'coordinates': None,
	})
	
m = folium.Map(
    location=st.session_state['center'],
    zoom_start=st.session_state['zoom'],
    tiles='OpenStreetMap',
    control_scale=True
)

marker = folium.LatLngPopup()
m.add_child(marker)

Draw(export=True).add_to(m)
folium.LayerControl().add_to(m)

map_data = st_folium(
    m, key='map', height=400, width='100%'
)
sel = map_data.get('last_clicked')
if sel:
    st.session_state['coordinates'] = [sel['lng'], sel['lat']]

# Display selected coordinates
if st.session_state['coordinates']:
    lon, lat = st.session_state['coordinates']
    st.subheader(f"📍 Lat: {lat:.4f}, Lon: {lon:.4f}")
    lon, lat = st.session_state['coordinates']
    # Get bounding box and demographics
    _, polygon = create_bounding_box(lon, lat, mode='display')
    # Get satellite image
    img_path = get_mapbox_image(lat, lon)
    img = Image.open(img_path)

    # Display styled satellite image
    st.subheader("🛰️ Styled Satellite View")
    # Apply a cool filter
    img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    st.image(img, use_container_width=True)

    os.remove(img_path)

else:
    st.info("Click on the map to choose a location.")


def create_bounding_box(longitude, latitude, mode):
    # Define the point and square side length
    point = Point(longitude, latitude)
    side_length = 0.02  # 0.01 degrees is approximately 1 km at the equator

    # Create a square bounding box around the point
    xmin = point.x - side_length / 2
    xmax = point.x + side_length / 2
    ymin = point.y - side_length / 2
    ymax = point.y + side_length / 2

    # Create bbox
    bboxes = []
    bboxes.append([(xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin), (xmax, ymin)])

    #Create polygon
    # Create polygon for gdf
    polygon = Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]) ######Creating a polygon within the function

    if mode == 'Model':
        return xmin, ymax
    else:
		return bboxes, polygon
	
def get_mapbox_image(latitude = 7.13, longitude = 7.66):

    # Set up your access token and the Mapbox API URL
    access_token = 'YOUR TOKEN HERE'
    # Replace with your Mapbox access token
    base_url = "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static"

    # Coordinates (latitude, longitude) and zoom level
    latitude = latitude
    longitude = longitude

    # Calculate the pixel size for a 1 km² area
    # At zoom level 17, 1 pixel is approximately 1 meter.
    # Therefore, 1 km (1000 meters) would be 1000 pixels.

    # Adjust zoom level for the 1 km² coverage, but reduce the zoom to fit the 512x512 image
    # In this case, for a 1km² area, adjust zoom level slightly (from 17 to 16.5 or 16)
    zoom_level = 16.5 #ADJUST  # Experimenting with fractional zoom level for better fit

    pixel_width = 512 #ADJUST
    pixel_height = 512 #ADJUST

    # Construct the request URL
    request_url = f"{base_url}/{longitude},{latitude},{zoom_level}/{pixel_width}x{pixel_height}?access_token={access_token}"

    # Make the request
    response = requests.get(request_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open the image
        image = Image.open(BytesIO(response.content))
        # Display the image
        #image.show()
        # Save the image

        #notice here same you need to start from solarodyssey since uvicorn starts from solarodyssey
        image_path = f"solarodyssey/mapbox_user_images/satellite_image_lat:{latitude}_lon:{longitude}.png"
        image.save(image_path)
    else:
        print(f"Failed to retrieve the image. Status code: {response.status_code}")

    return image_path
