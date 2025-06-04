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
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-5, "percentages must sum to 1.0"

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    post_images = sorted(image_dir.glob("*_post_disaster.png"))
    valid_triplets = []

    for post_img in post_images:
        base = post_img.name.replace("_post_disaster.png", "")
        pre_img = image_dir / f"{base}_pre_disaster.png"
        mask_path = mask_dir / f"{base}_post_disaster_mask.png"  # Ajusta si tus máscaras tienen otro nombre

        if pre_img.exists() and mask_path.exists():
            valid_triplets.append((pre_img, post_img, mask_path))

    print(f"Image triplets found: {len(valid_triplets)}")

    subset_size = int(len(valid_triplets) * dataset_fraction)
    random.seed(seed)
    selected_triplets = random.sample(valid_triplets, subset_size)
    print(f"Using {subset_size} image triplets ({dataset_fraction*100:.0f}%)")

    train_val, test = train_test_split(selected_triplets, test_size=test_pct, random_state=seed)
    val_rel_pct = val_pct / (train_pct + val_pct)
    train, val = train_test_split(train_val, test_size=val_rel_pct, random_state=seed)

    split_map = {"train": train, "val": val, "test": test}

    for split, triplets in split_map.items():
        img_out = output_dir / split / "images"
        msk_out = output_dir / split / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        msk_out.mkdir(parents=True, exist_ok=True)

        for pre_path, post_path, mask_path in triplets:
            shutil.copy(pre_path, img_out / pre_path.name)
            shutil.copy(post_path, img_out / post_path.name)
            shutil.copy(mask_path, msk_out / mask_path.name)

        print(f"{split}: {len(triplets)} triplets copied to {img_out.parent}")
