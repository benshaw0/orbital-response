from dotenv import load_dotenv
import os
import requests
from PIL import Image
from io import BytesIO

#load_dotenv(dotenv_path=".env")
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../../.env"))


def mapbox_api(lat, lon):
    zoom = 17
    size = (1024, 1024)
    mapbox_token = os.getenv("MAPBOX_API_KEY")

    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{zoom}/{size[0]}x{size[1]}?access_token={mapbox_token}"
    )

    r = requests.get(url)
    if r.ok:
        img = Image.open(BytesIO(r.content))
        os.makedirs("streamlit", exist_ok=True)
        img.save("streamlit/pre_disaster.png")
        print("Saved pre image .png")
    else:
        print("Error when downloading pre image:", r.status_code)

mapbox_api(31.55195, 34.49948)
