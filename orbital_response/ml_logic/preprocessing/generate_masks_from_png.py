import os
import json
import numpy as np
from shapely import wkt
from PIL import Image
from affine import Affine
from rasterio.features import rasterize
from tqdm import tqdm

DAMAGE_MAPPING = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,
}

COLOR_MAPPING = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (255, 255, 0),
    3: (255, 165, 0),
    4: (255, 0, 0),
}

def load_shapes_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    shapes = []
    if "features" in data and "xy" in data["features"]:
        for item in data["features"]["xy"]:
            try:
                polygon = wkt.loads(item["wkt"])
                subtype = item.get("properties", {}).get("subtype", "no-damage")
                class_id = DAMAGE_MAPPING.get(subtype, 0)
                shapes.append((polygon, class_id))
            except Exception as e:
                print(f"Error at json file: {json_path}: {e}")
                continue
    return shapes

def generate_masks_from_png(png_dir, json_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rgb_masks"), exist_ok=True)

    png_files = [f for f in os.listdir(png_dir) if f.endswith(".png")]
    for file in tqdm(png_files, desc="Generating masks"):
        base = file.replace(".png", "")
        png_path = os.path.join(png_dir, file)

        # Find the Json to image
        json_path = None
        for jd in json_dirs:
            candidate = os.path.join(jd, base + ".json")
            if os.path.exists(candidate):
                json_path = candidate
                break

        img = Image.open(png_path)
        width, height = img.size

        shapes = load_shapes_from_json(json_path) if json_path else []

        if shapes:
            mask = rasterize(
                ((geom, val) for geom, val in shapes),
                out_shape=(height, width),
                transform=Affine.translation(0, 0) * Affine.scale(1, 1),
                fill=0,
                dtype="uint8"
            )
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        # Grey Scale Masks
        mask_path = os.path.join(output_dir, base + "_mask.png")
        Image.fromarray(mask).save(mask_path)

        # RGB Checking masks
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        for class_id, color in COLOR_MAPPING.items():
            rgb[mask == class_id] = color
        rgb_path = os.path.join(output_dir, "rgb_masks", base + "_mask_rgb.png")
        Image.fromarray(rgb).save(rgb_path)
