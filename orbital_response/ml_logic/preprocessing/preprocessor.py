from src.preprocessing.convert_tif_to_png import convert_all_tifs
from src.preprocessing.generate_masks_from_png import generate_masks_from_png
from src.preprocessing.split_dataset import split_dataset


if __name__ == "__main__":
    print("Converting .tif to .png")
    convert_all_tifs(
        source_root="data",
        output_dir="data/processed/images"
    )

    print("using Json files to create the masks")
    generate_masks_from_png(
        png_dir="data/processed/images",
        json_dirs=["data/tier1/labels", "data/tier3/labels"],
        output_dir="data/processed/masks"
    )

    print("Splitting the data in train/test/val")
    split_dataset(
        image_dir="data/processed/images",
        mask_dir="data/processed/masks",
        output_dir="data/processed/split",
        dataset_fraction=1.0,
        train_pct=0.7,
        val_pct=0.15,
        test_pct=0.15
    )

    print("Preprocess pipe done")
