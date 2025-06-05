from dotenv import load_dotenv
import os
import requests
from PIL import Image
from io import BytesIO

load_dotenv(dotenv_path=".env")

def google_api(lat, lon):
    zoom = 17
    size="512x512"
    scale=2
    api_key = os.getenv("GOOGLE_API_KEY")

    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lon}&zoom={zoom}&size={size}&scale={scale}&maptype=satellite&key={api_key}"
    )


    r = requests.get(url)
    if r.ok:
        img = Image.open(BytesIO(r.content))
        img.save("post_disaster.png")
        print("Saved post image.png")
    else:
        print("Error when downloading post image:", r.status_code)

google_api(31.55195, 34.49948)
