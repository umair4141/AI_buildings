#!/usr/bin/env python3
"""Prepare Kaggle oral COCO zips into Ultralytics YOLO format."""

from __future__ import annotations

import json
import random
import shutil
import zipfile
from pathlib import Path

from PIL import Image

# Kaggle input layout from user
KAGGLE_ROOT = Path("/kaggle/input/datasets/xxccfffffffffff/oral-dataset/oral dataset")
OUT_ROOT = Path("/kaggle/working/oral_yolo")

# Fallback local paths for non-Kaggle debugging
LOCAL_ROOT = Path("data/oral_raw")
LOCAL_OUT = Path("data/oral_yolo")

CLASS_NAMES_FALLBACK = [
    "Apical Periodontitis",
    "Decay",
    "Wisdom Tooth",
    "Missing Tooth",
    "Dental Filling",
    "Root Canal Filling",
    "Implant",
    "Porcelain Crown",
    "Ceramic Bridge",
]


def pick_roots() -> tuple[Path, Path]:
    if KAGGLE_ROOT.exists():
        return KAGGLE_ROOT, OUT_ROOT
    return LOCAL_ROOT, LOCAL_OUT


def unzip_if_needed(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".unzipped"
    if marker.exists():
        return
    print(f"Unzipping {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    marker.write_text("ok\n", encoding="utf-8")


def find_coco_json(ann_dir: Path, split: str) -> Path:
    # Common COCO names
    candidates = [
        ann_dir / f"instances_{split}2017.json",
        ann_dir / f"instances_{split}.json",
        ann_dir / f"{split}.json",
        ann_dir / f"instances_{split}2017" / f"instances_{split}2017.json",
    ]
    # Also search recursively
    hits = list(ann_dir.rglob(f"*{split}*.json"))
    for c in candidates + hits:
        if c.is_file():
            return c
    raise FileNotFoundError(f"No COCO json for split={split} under {ann_dir}")


def find_image_dir(img_root: Path, split: str) -> Path:
    direct = img_root / f"{split}2017"
    if direct.is_dir():
        return direct
    # zip may extract with nested folder
    hits = [p for p in img_root.rglob("*") if p.is_dir() and split in p.name.lower()]
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No image dir for split={split} under {img_root}")


def coco_to_yolo(coco_json: Path, image_dir: Path, label_dir: Path, names: list[str]) -> list[str]:
    label_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(coco_json.read_text(encoding="utf-8"))

    cats = sorted(data["categories"], key=lambda c: c["id"])
    # Remap arbitrary COCO category ids -> contiguous 0..nc-1
    id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names_from_json = [c["name"] for c in cats]
    if len(names_from_json) == 9:
        names[:] = names_from_json

    images = {im["id"]: im for im in data["images"]}
    anns_by_img: dict[int, list] = {i: [] for i in images}
    skipped_cat = 0
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann["category_id"] not in id_to_idx:
            skipped_cat += 1
            continue
        anns_by_img[ann["image_id"]].append(ann)

    for img_id, im in images.items():
        file_name = im["file_name"]
        src = image_dir / file_name
        if not src.exists():
            # sometimes file_name has subdirs stripped in zip
            src = image_dir / Path(file_name).name
        if not src.exists():
            print(f"WARN missing image: {file_name}")
            continue

        w = im.get("width")
        h = im.get("height")
        if not w or not h:
            with Image.open(src) as pic:
                w, h = pic.size

        lines = []
        for ann in anns_by_img[img_id]:
            x, y, bw, bh = ann["bbox"]  # COCO xywh absolute
            if bw <= 0 or bh <= 0:
                continue
            cx = (x + bw / 2.0) / w
            cy = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            cls = id_to_idx[ann["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        stem = Path(file_name).stem
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    print(f"skipped_unknown_category_id={skipped_cat}")
    return names


def link_or_copy_images(image_dir: Path, out_img_dir: Path) -> None:
    out_img_dir.mkdir(parents=True, exist_ok=True)
    for src in image_dir.rglob("*"):
        if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            continue
        dst = out_img_dir / src.name
        if dst.exists():
            continue
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def split_val_into_val_test(out_root: Path, seed: int = 0, test_ratio: float = 0.5) -> None:
    rng = random.Random(seed)
    img_val, lab_val = out_root / "images/val", out_root / "labels/val"
    img_test, lab_test = out_root / "images/test", out_root / "labels/test"
    img_test.mkdir(parents=True, exist_ok=True)
    lab_test.mkdir(parents=True, exist_ok=True)
    if any(img_test.iterdir()):
        return
    stems = sorted({p.stem for p in lab_val.glob("*.txt")})
    rng.shuffle(stems)
    test_stems = set(stems[: max(1, int(len(stems) * test_ratio))])
    for stem in test_stems:
        for src in list(img_val.glob(stem + ".*")) + list(lab_val.glob(stem + ".*")):
            dst_dir = img_test if src.parent.name == "val" and src.suffix.lower() != ".txt" else lab_test
            if src.suffix.lower() == ".txt":
                dst_dir = lab_test
            else:
                dst_dir = img_test
            shutil.move(str(src), str(dst_dir / src.name))


def write_data_yaml(out_root: Path, names: list[str]) -> Path:
    yaml_path = out_root / "oral.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    text = f"""# Auto-generated oral detection dataset
path: {out_root.resolve()}
train: images/train
val: images/val
test: images/test

names:
{names_block}
"""
    yaml_path.write_text(text, encoding="utf-8")
    return yaml_path


def main() -> None:
    root, out = pick_roots()
    print(f"RAW={root}")
    print(f"OUT={out}")

    ann_zip = root / "annotations.zip"
    train_zip = root / "train2017.zip"
    val_zip = root / "val2017.zip"
    for z in (ann_zip, train_zip, val_zip):
        if not z.exists():
            raise FileNotFoundError(f"Missing {z}")

    raw = out / "_raw"
    unzip_if_needed(ann_zip, raw / "annotations")
    unzip_if_needed(train_zip, raw / "train2017")
    unzip_if_needed(val_zip, raw / "val2017")

    ann_dir = raw / "annotations"
    train_img = find_image_dir(raw / "train2017", "train")
    val_img = find_image_dir(raw / "val2017", "val")
    train_json = find_coco_json(ann_dir, "train")
    val_json = find_coco_json(ann_dir, "val")

    names = list(CLASS_NAMES_FALLBACK)
    names = coco_to_yolo(train_json, train_img, out / "labels/train", names)
    coco_to_yolo(val_json, val_img, out / "labels/val", names)

    link_or_copy_images(train_img, out / "images/train")
    link_or_copy_images(val_img, out / "images/val")
    split_val_into_val_test(out, seed=0, test_ratio=0.5)

    yaml_path = write_data_yaml(out, names)
    print(f"Done. Dataset YAML: {yaml_path}")
    print(f"Classes ({len(names)}): {names}")


if __name__ == "__main__":
    main()
