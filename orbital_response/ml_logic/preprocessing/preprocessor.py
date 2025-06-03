import os
import shutil
import rasterio
import numpy as np
from rasterio.features import rasterize
from PIL import Image
from tqdm import tqdm

from orbital_response.ml_logic.preprocessing.geojson_to_mask import load_custom_json

SELECTED_DISASTERS = {
    "hurricane-michael",
    "mexico-earthquake",
    "hurricane-matthew",
    "hurricane-harvey",
    "hurricane-florence",
    "joplin-tornado",
    "moore-tornado",
    "tuscaloosa-tornado",
}

COLOR_MAPPING = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (255, 255, 0),
    3: (255, 165, 0),
    4: (255, 0, 0),
}

def geojson_to_mask(json_path, image_path, output_shape):
    shapes = load_custom_json(json_path)
    with rasterio.open(image_path) as src:
        transform = src.transform

    return rasterize(
        shapes,
        out_shape=output_shape,
        transform=transform,
        fill=0,
        dtype='uint8'
    )

def process_and_filter(source_root, filtered_dir):
    image_dir = os.path.join(source_root, "images")
    label_dir = os.path.join(source_root, "labels")

    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(os.path.join(filtered_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(filtered_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(filtered_dir, "masks"), exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]

    for image_file in tqdm(image_files, desc=f"Procesando {source_root}"):
        base_name = image_file.replace(".tif", "")
        disaster = base_name.split("_")[0]

        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, base_name + ".json")

        if disaster not in SELECTED_DISASTERS:
            # 🧹 Eliminar archivos no deseados
            if os.path.exists(image_path):
                os.remove(image_path)
            if os.path.exists(label_path):
                os.remove(label_path)
            continue

        dest_img = os.path.join(filtered_dir, "images", image_file)
        dest_label = os.path.join(filtered_dir, "labels", os.path.basename(label_path))
        mask_path = os.path.join(filtered_dir, "masks", base_name + "_mask.png")
        rgb_path = mask_path.replace("_mask.png", "_mask_rgb.png")

        if not os.path.exists(label_path):
            print(f"⚠️ Falta el label de {image_file}")
            continue

        try:
            shutil.copy(image_path, dest_img)
            shutil.copy(label_path, dest_label)

            with rasterio.open(image_path) as src:
                shape = (src.height, src.width)

            mask = geojson_to_mask(label_path, image_path, shape)

            rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for class_id, color in COLOR_MAPPING.items():
                rgb_mask[mask == class_id] = color

            Image.fromarray(rgb_mask).save(rgb_path)
            Image.fromarray(mask.astype(np.uint8)).save(mask_path)

        except Exception as e:
            print(f"❌ Error at {base_name}: {e}")


if __name__ == "__main__":
    # Filter good files
    filtered_dir = "data/filtered"
    for tier in ["tier1", "tier3"]:
        source_root = os.path.join("data", tier)
        process_and_filter(source_root, filtered_dir)
