import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(
    image_dir,
    mask_dir,
    output_dir,
    dataset_fraction=1.0,
    train_pct=0.7,
    val_pct=0.15,
    test_pct=0.15,
    seed=42
    ):

    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-5, "percentages needs to sum 1.0"

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    image_paths = sorted(image_dir.glob("*_post_disaster.tif"))
    valid_pairs = [(img, mask_dir / img.name.replace(".tif", "_mask.png")) for img in image_paths if (mask_dir / img.name.replace(".tif", "_mask.png")).exists()]

    print(f"🔍 Amount of valid pairs: {len(valid_pairs)}")

    # LIMIT THE % OF THE DATASET
    subset_size = int(len(valid_pairs) * dataset_fraction)
    random.seed(seed)
    selected_pairs = random.sample(valid_pairs, subset_size)
    print(f"✅ Using {subset_size} image pairs ({dataset_fraction*100:.0f}%)")

    # Split
    train_val, test = train_test_split(selected_pairs, test_size=test_pct, random_state=seed)
    val_rel_pct = val_pct / (train_pct + val_pct)
    train, val = train_test_split(train_val, test_size=val_rel_pct, random_state=seed)

    split_map = {"train": train, "val": val, "test": test}

    for split, pairs in split_map.items():
        img_out = output_dir / split / "images"
        msk_out = output_dir / split / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        msk_out.mkdir(parents=True, exist_ok=True)

        for img_path, mask_path in pairs:
            shutil.copy(img_path, img_out / img_path.name)
            shutil.copy(mask_path, msk_out / mask_path.name)

        print(f"📁 {split}: {len(pairs)} images pairs sent to {img_out.parent}")

if __name__ == "__main__":
    split_dataset(
        image_dir="data/filtered/images",
        mask_dir="data/filtered/masks",
        output_dir="data/processed_dataset",
        dataset_fraction=1,   # TOTAL % IN THE SPLIT
        train_pct=0.7,
        val_pct=0.15,
        test_pct=0.15
    )
