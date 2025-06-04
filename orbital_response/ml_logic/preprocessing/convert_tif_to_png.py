
import os
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm ## Progress meter bars for tasks

# Preselected disasters we want to feed the model with
SELECTED_DISASTERS = {
    "hurricane-michael",
    "mexico-earthquake",
    "hurricane-matthew",
    "hurricane-harvey",
    "hurricane-florence",
    "joplin-tornado",
    "moore-tornado",
    "tuscaloosa-tornado"
}

## Convert .tif images to .png
def convert_all_tifs(source_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for tier in ["tier1", "tier3"]:
        image_dir = os.path.join(source_root, tier, "images")
        if not os.path.exists(image_dir):
            continue

        tif_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]
        for tif_file in tqdm(tif_files, desc=f"converting {tier}"):
            base_name = tif_file.replace(".tif", "")
            disaster = base_name.split("_")[0]
            if disaster not in SELECTED_DISASTERS:
                continue

            tif_path = os.path.join(image_dir, tif_file)
            png_path = os.path.join(output_dir, base_name + ".png")

            try:
                with rasterio.open(tif_path) as src:
                    img = src.read([1, 2, 3])
                    img = np.transpose(img, (1, 2, 0)).astype(np.uint8)
                Image.fromarray(img).save(png_path)
            except Exception as e:
                print(f"Error on {tif_file}: {e}")
