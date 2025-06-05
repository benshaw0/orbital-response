import shutil


from orbital_response.ml_logic.preprocessing.convert_tif_to_png import convert_all_tifs
from orbital_response.ml_logic.preprocessing.generate_masks_from_png import generate_masks_from_png
from orbital_response.ml_logic.preprocessing.split_dataset import split_dataset


def ask_yes_no(prompt):
    while True:
        answer = input(f"{prompt} (y/n): ").strip().lower()
        if answer in {"y", "yes"}:
            return True
        elif answer in {"n", "no"}:
            return False
        else:
            print("Please enter 'y' or 'n'.")

def ask_float(prompt):
    while True:
        try:
            return float(input(f"{prompt}: ").strip())
        except ValueError:
            print("Please enter a valid number (e.g., 0.15 for 15%).")

if __name__ == "__main__":

    ##### GET DATASET TIF AND CONVERT THEM TO PNG
    if ask_yes_no("Do you want to convert .tif files to .png?"):
        print("Converting .tif to .png")
        convert_all_tifs(
            source_root="data",
            output_dir="data/processed/images"
        )

    ##### GET THE RGB MASKS FROM JSON USING IMAGES 1024x1024 FORMAT
    if ask_yes_no("Do you want to generate masks from JSON files?"):
        print("Using JSON files to create the masks")
        generate_masks_from_png(
            png_dir="data/processed/images",
            json_dirs=["data/tier1/labels", "data/tier3/labels"],
            output_dir="data/processed/masks"
        )

    ##### SPLIT HOLDOUT METHOD #####
    if ask_yes_no("Do you want to split the dataset into train/val/test?"):
        total_split_images = ask_float("write the % of the dataset you want to use from 0.1 (10%) to 1.0 (100%)")
        ##### REMOVE SPLIT OLD FOLDER IF ANY #####
        print("Cleaning old split files")
        try:
            shutil.rmtree("data/processed/split")
        except FileNotFoundError:
            pass

        print("Splitting the data in train/test/val")
        split_dataset(
            image_dir="data/processed/images",
            mask_dir="data/processed/masks",
            output_dir="data/processed/split",
            dataset_fraction=total_split_images,
            train_pct=0.7,
            val_pct=0.15,
            test_pct=0.15
        )

    print("preprocessing completed")
