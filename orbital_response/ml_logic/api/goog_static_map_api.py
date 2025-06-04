from dotenv import load_dotenv
import os
import requests
from PIL import Image
from io import BytesIO

load_dotenv()

lat, lon = 31.551953, 34.49948
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
    img.save("orbital_response/ml_logic/api/images/post_disaster.png")
    print("Saved post image.png")
else:
    print("Error when downloading post image:", r.status_code)
