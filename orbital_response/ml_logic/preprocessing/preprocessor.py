import shutil


from orbital_response.ml_logic.preprocessing.convert_tif_to_png import convert_all_tifs
from orbital_response.ml_logic.preprocessing.generate_masks_from_png import generate_masks_from_png
from orbital_response.ml_logic.preprocessing.split_dataset import split_dataset


if __name__ == "__main__":

    ##### GET DATASET TIF AND CONVERT THEM TO PNG
    print("Converting .tif to .png")
    convert_all_tifs(
        source_root="data",
        output_dir="data/processed/images"
    )

    ##### GET THE RGB MASKS FROM JSON USING IMAGES 1024x1024 FORMAT
    print("using Json files to create the masks")
    generate_masks_from_png(
        png_dir="data/processed/images",
        json_dirs=["data/tier1/labels", "data/tier3/labels"],
        output_dir="data/processed/masks"
    )

    ##### REMOVE SPLIT OLD FOLDER IF ANY #####
    print("Cleaning old split files")
    try:
        shutil.rmtree("data/processed/split")
    except FileNotFoundError:
        pass

    ##### SPLIT HOLDOUT METHOD #####
    print("Splitting the data in train/test/val")
    split_dataset(
        image_dir="data/processed/images",
        mask_dir="data/processed/masks",
        output_dir="data/processed/split",
        dataset_fraction=0.15,
        train_pct=0.7,
        val_pct=0.15,
        test_pct=0.15
    )
    print("Preprocess pipe done")
