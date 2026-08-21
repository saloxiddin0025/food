"""Builds a fixed, stratified train/val split shared by every model we train,
so ResNet and (later) CLIP get evaluated on the exact same held-out images."""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

CHEESE_DIR = Path(r"c:\Users\OEM\Downloads\cheese pizzas\plain cheese pizzass")
MEAT_DIR = Path(r"c:\Users\OEM\Downloads\meat pizzas\plain meat pizzass")

CLASSES = ["cheese_pizza", "meat_pizza"]
VAL_FRACTION = 0.15
SEED = 42

OUTPUT_PATH = Path(__file__).parent / "split.json"


def collect(dir_path: Path, label: int):
    exts = {".jpg", ".jpeg", ".png"}
    return [(str(p), label) for p in sorted(dir_path.iterdir()) if p.suffix.lower() in exts]


def main():
    samples = collect(CHEESE_DIR, 0) + collect(MEAT_DIR, 1)
    paths = [p for p, _ in samples]
    labels = [l for _, l in samples]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=VAL_FRACTION, random_state=SEED, stratify=labels
    )

    split = {
        "classes": CLASSES,
        "train": list(zip(train_paths, train_labels)),
        "val": list(zip(val_paths, val_labels)),
    }

    OUTPUT_PATH.write_text(json.dumps(split, indent=2))

    print(f"Total images: {len(samples)} (cheese={labels.count(0)}, meat={labels.count(1)})")
    print(f"Train: {len(train_paths)}  Val: {len(val_paths)}")
    print(f"Split written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
