# ============================================================
# OralXrays-9 + YOLO26n + IFSA (IFS + SwiftFormer additive attn)
# Feature-level IFS at neck P3/P4 | train FROM SCRATCH | 150 epochs
# ============================================================
# Kaggle: Run > Restart session, then run this script once from the top.
# Do not use importlib.reload (breaks DetectionModel pickle identity).
# ============================================================

import os
import sys
import io
import ast
import json
import random
import shutil
import inspect
import subprocess
from pathlib import Path
from collections import Counter, defaultdict

# ============================================================
# 1. SETTINGS
# ============================================================

PASSWORD = "CVPR2024-OralXrays-9"

DATASET_URL = (
    "https://drive.google.com/drive/folders/"
    "1_y7ERcFicnOYY2DMR6Qe1W4KGdCsoQ1n"
)

RAW_DIR = Path("/kaggle/temp/oralxrays9_raw")
EXTRACT_DIR = Path("/kaggle/temp/oralxrays9_extracted")
YOLO_DIR = Path("/kaggle/working/oralxrays9_ifsa_seed42_v1")

DATA_YAML = Path("/kaggle/working/oralxrays9.yaml")
MODEL_YAML = Path("/kaggle/working/yolo26n-ifsa.yaml")
SPLIT_FILE = Path("/kaggle/working/oralxrays9_split_ids.json")

EPOCHS = 150
IMAGE_SIZE = 640
BATCH_SIZE = 8
SEED = 42
VALIDATION_FRACTION = 0.20

# IFS / IFSA hyperparameters
IFS_LAMBDA = 0.5          # hesitation penalty in score S = mu - nu - lambda * pi
IFSA_AT_P3 = True         # main feature-level IFS (small lesions)
IFSA_AT_P4 = True         # multi-scale fusion path
IFSA_AT_P5 = False        # keep False for FL/edge cost unless ablating

for directory in (RAW_DIR, EXTRACT_DIR, YOLO_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. INSTALL AND IMPORT PACKAGES
# ============================================================

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "ultralytics",
        "gdown",
        "pyzipper",
        "pyyaml",
    ],
    check=True,
)

import gdown
import pyzipper
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import ultralytics
import ultralytics.nn.tasks as tasks

from ultralytics import YOLO

print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("GPU unavailable: training will use CPU.")

# ============================================================
# 3. DOWNLOAD AND EXTRACT DATASET
# ============================================================

existing_train_json = list(EXTRACT_DIR.rglob("instances_train2017.json"))

if not existing_train_json:
    raw_files = [path for path in RAW_DIR.rglob("*") if path.is_file()]

    if not raw_files:
        print("\nDownloading OralXrays-9...")
        gdown.download_folder(
            url=DATASET_URL,
            output=str(RAW_DIR),
            quiet=False,
            remaining_ok=True,
        )

    raw_files = [path for path in RAW_DIR.rglob("*") if path.is_file()]
    if not raw_files:
        raise RuntimeError(
            "No files downloaded. Check Kaggle Internet and Drive access."
        )

    archives = [
        path
        for path in raw_files
        if path.name.lower().endswith((".zip", ".7z", ".rar", ".001"))
    ]

    for archive in archives:
        print("\nExtracting:", archive.name)

        if archive.name.lower().endswith(".zip"):
            try:
                with pyzipper.AESZipFile(archive) as zip_file:
                    zip_file.pwd = PASSWORD.encode()
                    zip_file.extractall(EXTRACT_DIR)
                continue
            except Exception as exc:
                print("ZIP extraction failed:", exc)
                print("Trying 7z...")

        if shutil.which("7z") is None:
            raise RuntimeError(
                f"7z is required to extract {archive.name}. "
                "Install p7zip or extract the archive separately."
            )

        subprocess.run(
            [
                "7z",
                "x",
                str(archive),
                f"-p{PASSWORD}",
                f"-o{EXTRACT_DIR}",
                "-y",
            ],
            check=True,
        )

# ============================================================
# 4. FIND AND LOAD COCO ANNOTATIONS
# ============================================================

SEARCH_ROOTS = [EXTRACT_DIR, RAW_DIR]


def find_unique(filename):
    matches = [path for root in SEARCH_ROOTS for path in root.rglob(filename)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {filename}; found: {matches}")
    return matches[0]


train_json_path = find_unique("instances_train2017.json")
test_json_path = find_unique("instances_val2017.json")

with open(train_json_path, "r", encoding="utf-8") as file:
    train_coco = json.load(file)

with open(test_json_path, "r", encoding="utf-8") as file:
    test_coco = json.load(file)

print("\nTrain JSON:", train_json_path)
print("Test JSON:", test_json_path)
print("Original training images:", len(train_coco["images"]))
print("Original test images:", len(test_coco["images"]))

# ============================================================
# 5. VERIFY NINE-CLASS CATEGORY MAPPING
# ============================================================

categories = sorted(train_coco["categories"], key=lambda category: category["id"])

train_categories = {category["id"]: category["name"] for category in categories}
test_categories = {
    category["id"]: category["name"] for category in test_coco["categories"]
}

if len(categories) != 9:
    raise ValueError(f"Expected 9 named categories; found {len(categories)}.")

if train_categories != test_categories:
    raise ValueError("Train and test category IDs or names do not match.")

if [category["id"] for category in categories] != list(range(1, 10)):
    raise ValueError("Expected named category IDs 1-9. Inspect the JSON.")

annotation_to_yolo = {
    category["id"]: index for index, category in enumerate(categories)
}
names = {index: category["name"] for index, category in enumerate(categories)}

train_counts = Counter(
    annotation["category_id"] for annotation in train_coco["annotations"]
)
test_counts = Counter(
    annotation["category_id"] for annotation in test_coco["annotations"]
)

observed_ids = set(train_counts) | set(test_counts)
unexpected_ids = observed_ids - set(annotation_to_yolo) - {0}
if unexpected_ids:
    raise ValueError(f"Unexpected category IDs: {sorted(unexpected_ids)}")

print("\nClass mapping:")
for original_id, yolo_id in annotation_to_yolo.items():
    print(f"COCO {original_id} -> YOLO {yolo_id} -> {names[yolo_id]}")

print("\nTrain annotation counts:", dict(sorted(train_counts.items())))
print("Test annotation counts:", dict(sorted(test_counts.items())))
print(
    "\nUnknown ID 0 annotations excluded:",
    "train =",
    train_counts.get(0, 0),
    "test =",
    test_counts.get(0, 0),
)

# ============================================================
# 6. INDEX IMAGES
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
image_index = defaultdict(list)

for root in SEARCH_ROOTS:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            image_index[path.name].append(path)

print(
    "\nIndexed image files:",
    sum(len(paths) for paths in image_index.values()),
)


def locate_image(image_info, expected_folder):
    filename = Path(image_info["file_name"]).name
    candidates = image_index.get(filename, [])

    if len(candidates) == 1:
        return candidates[0]

    matching_split = [path for path in candidates if expected_folder in path.parts]
    if len(matching_split) == 1:
        return matching_split[0]

    raise FileNotFoundError(
        f"Cannot uniquely locate {filename!r}. Candidates: {candidates[:10]}"
    )


# ============================================================
# 7. CREATE TRAIN / VALIDATION / TEST SPLITS
# ============================================================
# Provisional image-level split.
# Replace with patient-level grouping if patient IDs are available.

rng = random.Random(SEED)
training_images = list(train_coco["images"])
rng.shuffle(training_images)

number_validation = max(1, round(len(training_images) * VALIDATION_FRACTION))
validation_images = training_images[:number_validation]
fit_images = training_images[number_validation:]
test_images = list(test_coco["images"])

print("\nSplit sizes:")
print("Train:", len(fit_images))
print("Validation:", len(validation_images))
print("Test:", len(test_images))
print(
    "WARNING: The train/validation split is image-level and "
    "has not been verified as patient-level."
)

# ============================================================
# 8. CONVERT COCO BOXES TO YOLO FORMAT
# ============================================================


def convert_split(coco, images, split_name, source_folder):
    images_dir = YOLO_DIR / "images" / split_name
    labels_dir = YOLO_DIR / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    all_image_ids = {image["id"] for image in coco["images"]}
    if len(all_image_ids) != len(coco["images"]):
        raise ValueError(f"Duplicate image IDs in source for {split_name}.")

    selected_ids = {image["id"] for image in images}
    annotations_by_image = defaultdict(list)
    excluded_unknown = 0

    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]

        if image_id not in all_image_ids:
            raise ValueError(f"Annotation references unknown image ID: {image_id}")

        if category_id == 0:
            if image_id in selected_ids:
                excluded_unknown += 1
            continue

        if category_id not in annotation_to_yolo:
            raise ValueError(f"Unexpected category ID: {category_id}")

        if annotation.get("iscrowd", 0):
            raise ValueError(
                "Found iscrowd annotations. Inspect them and define "
                "a handling policy before conversion."
            )

        annotations_by_image[image_id].append(annotation)

    invalid_boxes = 0

    for image in images:
        source = locate_image(image, source_folder)
        image_id = image["id"]
        destination = images_dir / f"{image_id}{source.suffix.lower()}"

        if not destination.exists():
            os.symlink(source.resolve(), destination)

        image_width = float(image["width"])
        image_height = float(image["height"])
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid dimensions for image {image_id}.")

        label_lines = []
        for annotation in annotations_by_image[image_id]:
            x, y, width, height = map(float, annotation["bbox"])

            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(image_width, x + width)
            y2 = min(image_height, y + height)

            if x2 <= x1 or y2 <= y1:
                invalid_boxes += 1
                continue

            x_center = (x1 + x2) / (2.0 * image_width)
            y_center = (y1 + y2) / (2.0 * image_height)
            box_width = (x2 - x1) / image_width
            box_height = (y2 - y1) / image_height
            class_id = annotation_to_yolo[annotation["category_id"]]

            label_lines.append(
                f"{class_id} {x_center:.8f} {y_center:.8f} "
                f"{box_width:.8f} {box_height:.8f}"
            )

        label_path = labels_dir / f"{image_id}.txt"
        label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )

    print(
        f"{split_name}: {len(images)} images | "
        f"ID 0 excluded: {excluded_unknown} | "
        f"invalid boxes skipped: {invalid_boxes}"
    )


convert_split(train_coco, fit_images, "train", "train2017")
convert_split(train_coco, validation_images, "val", "train2017")
convert_split(test_coco, test_images, "test", "val2017")

# ============================================================
# 9. CREATE DATASET YAML AND SAVE SPLIT IDS
# ============================================================

dataset_config = {
    "path": str(YOLO_DIR),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": names,
}

with open(DATA_YAML, "w", encoding="utf-8") as file:
    yaml.safe_dump(dataset_config, file, sort_keys=False, allow_unicode=True)

split_ids = {
    "train": [image["id"] for image in fit_images],
    "val": [image["id"] for image in validation_images],
    "test": [image["id"] for image in test_images],
}

with open(SPLIT_FILE, "w", encoding="utf-8") as file:
    json.dump(split_ids, file, indent=2)

print("\nDataset YAML:")
print(DATA_YAML.read_text(encoding="utf-8"))

# ============================================================
# 10. DEFINE IFSA = IFS + SwiftFormer-style additive attention
# ============================================================


class IFSA(nn.Module):
    """
    Intuitionistic Fuzzy SwiftFormer-style Additive attention for YOLO neck.

    Input/Output: (B, C, H, W)
    - Additive attention mixes spatial tokens (linear in N; edge/FL friendly)
    - IFS (mu, nu, pi) with mu + nu <= 1 gates features:
        mu = membership (attend / keep lesion evidence)
        nu = non-membership (suppress background)
        pi = hesitation (epistemic ambiguity)
    """

    def __init__(self, c1, c2=None, lambda_hes=IFS_LAMBDA):
        super().__init__()
        c2 = c1 if c2 is None else c2
        if c1 != c2:
            raise ValueError(
                f"IFSA expects c1 == c2 for residual use; got c1={c1}, c2={c2}"
            )

        self.c = c1
        self.lambda_hes = float(lambda_hes)

        # SwiftFormer-style efficient additive attention (channel as token dim)
        self.to_query = nn.Conv2d(c1, c1, 1, bias=False)
        self.to_key = nn.Conv2d(c1, c1, 1, bias=False)
        self.w_g = nn.Parameter(torch.randn(c1, 1) * 0.02)
        self.proj = nn.Conv2d(c1, c1, 1, bias=False)
        self.scale = c1**-0.5

        # IFS heads: predict unconstrained logits then enforce mu+nu<=1
        self.mu_head = nn.Sequential(
            nn.Conv2d(c1, c1 // 4 if c1 >= 4 else c1, 1),
            nn.SiLU(),
            nn.Conv2d(c1 // 4 if c1 >= 4 else c1, 1, 1),
        )
        self.nu_head = nn.Sequential(
            nn.Conv2d(c1, c1 // 4 if c1 >= 4 else c1, 1),
            nn.SiLU(),
            nn.Conv2d(c1 // 4 if c1 >= 4 else c1, 1, 1),
        )

        self.act = nn.SiLU()
        self.last_mu = None
        self.last_nu = None
        self.last_pi = None

    def _additive_attention(self, x):
        # x: B,C,H,W  -> treat spatial positions as tokens with dim C
        b, c, h, w = x.shape
        q = self.to_query(x)
        k = self.to_key(x)

        # Normalize along channel (token feature) dimension
        q_n = F.normalize(q.flatten(2), dim=1)  # B,C,N
        k_n = F.normalize(k.flatten(2), dim=1)  # B,C,N

        # query global weight: (B,C,N)^T @ (C,1) -> B,N,1 style via einsum
        # w_g: C,1
        qw = torch.einsum("bcn,cj->bnj", q_n, self.w_g) * self.scale  # B,N,1
        a = F.normalize(qw, dim=1)  # over tokens

        # Global query context G: sum_n a_n * q_n -> B,C
        g = torch.einsum("bnj,bcn->bc", a, q_n)  # B,C
        g = g.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)  # B,C,H,W

        k_map = k_n.view(b, c, h, w)
        out = self.proj(g * k_map) + q
        return out

    def _ifs_gate(self, x):
        # Proper IFS: mu in [0,1], nu in [0,1-mu], pi = 1-mu-nu >= 0
        mu = torch.sigmoid(self.mu_head(x))
        nu = (1.0 - mu) * torch.sigmoid(self.nu_head(x))
        pi = (1.0 - mu - nu).clamp_min(0.0)

        # Score in roughly [-1,1]; map to positive gate
        score = mu - nu - self.lambda_hes * pi
        gate = torch.sigmoid(score)

        self.last_mu = mu.detach()
        self.last_nu = nu.detach()
        self.last_pi = pi.detach()
        return gate

    def forward(self, x):
        y = self._additive_attention(x)
        gate = self._ifs_gate(y)
        y = y * gate
        return self.act(x + y)


# ============================================================
# 11. WRITE YOLO26n + IFSA NECK YAML
# ============================================================

# Build Detect indices from the exact layer sequence (safe when flags change).
head_lines = [
    '  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]',
    "  - [[-1, 6], 1, Concat, [1]] # cat backbone P4",
    "  - [-1, 2, C3k2, [512, True]] # 13",
]
idx = 13
p4_fuse = idx
if IFSA_AT_P4:
    idx += 1
    head_lines.append(f"  - [-1, 1, IFSA, [512]] # {idx} IFSA on P4 upsample path")
    p4_fuse = idx

idx += 1
head_lines.append('  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]')
idx += 1
head_lines.append("  - [[-1, 4], 1, Concat, [1]] # cat backbone P3")
idx += 1
head_lines.append(f"  - [-1, 2, C3k2, [256, True]] # {idx} P3/8-small")
p3_out = idx
if IFSA_AT_P3:
    idx += 1
    head_lines.append(f"  - [-1, 1, IFSA, [256]] # {idx} IFSA @ P3 (main)")
    p3_out = idx

idx += 1
head_lines.append("  - [-1, 1, Conv, [256, 3, 2]]")
idx += 1
head_lines.append(f"  - [[-1, {p4_fuse}], 1, Concat, [1]] # cat head P4")
idx += 1
head_lines.append(f"  - [-1, 2, C3k2, [512, True]] # {idx} P4/16-medium")
p4_out = idx
if IFSA_AT_P4:
    idx += 1
    head_lines.append(f"  - [-1, 1, IFSA, [512]] # {idx} IFSA @ P4 out")
    p4_out = idx

idx += 1
head_lines.append("  - [-1, 1, Conv, [512, 3, 2]]")
idx += 1
head_lines.append("  - [[-1, 10], 1, Concat, [1]] # cat head P5")
idx += 1
head_lines.append(
    f"  - [-1, 1, C3k2, [1024, True, 0.5, True]] # {idx} P5/32-large"
)
p5_out = idx
if IFSA_AT_P5:
    idx += 1
    head_lines.append(f"  - [-1, 1, IFSA, [1024]] # {idx} IFSA @ P5")
    p5_out = idx

idx += 1
head_lines.append(
    f"  - [[{p3_out}, {p4_out}, {p5_out}], 1, Detect, [nc]] # Detect(P3,P4,P5)"
)

model_yaml_text = """# YOLO26n + IFSA neck (IFS + SwiftFormer additive attention)
# OralXrays-9 | feature-level IFS at P3/P4
# Backbone unchanged; SwiftFormer is NOT a full backbone replace.

nc: 9
end2end: True
reg_max: 1
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]] # 9
  - [-1, 2, C2PSA, [1024]] # 10

head:
""" + "\n".join(head_lines) + "\n"

MODEL_YAML.write_text(model_yaml_text, encoding="utf-8")
print("\nModel YAML written:", MODEL_YAML)
print(model_yaml_text)

# ============================================================
# 12. REGISTER IFSA WITH ULTRALYTICS parse_model
# ============================================================

# Expose IFSA where parse_model resolves module names.
IFSA.__module__ = tasks.__name__
tasks.IFSA = IFSA

# Also attach on nn.modules if present (some Ultralytics versions import from there).
try:
    import ultralytics.nn.modules as nn_modules

    IFSA.__module__ = nn_modules.__name__
    nn_modules.IFSA = IFSA
    # Prefer tasks namespace for pickle stability with DetectionModel.
    IFSA.__module__ = tasks.__name__
    tasks.IFSA = IFSA
except Exception as exc:
    print("Note: could not attach IFSA to nn.modules:", exc)

# Ensure parse_model can resolve "IFSA" and treat it like Conv (channel scaling).
# Without base_modules membership, YOLO26n width=0.25 would call IFSA(256) on
# 64-ch feature maps and crash.
tasks_file = Path(inspect.getfile(tasks))
tasks_source = tasks_file.read_text(encoding="utf-8")
syntax_tree = ast.parse(tasks_source, filename=str(tasks_file))
parser_node = next(
    (
        node
        for node in syntax_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "parse_model"
    ),
    None,
)
if parser_node is None:
    raise RuntimeError("parse_model was not found in tasks.py")

parser_source = ast.get_source_segment(tasks_source, parser_node)
if parser_source is None:
    raise RuntimeError("Could not extract parse_model source.")

# Insert IFSA into the base_modules frozenset literal if present.
old_base = "C2PSA,"
new_base = "C2PSA,\n            IFSA,"
if "IFSA," not in parser_source and old_base in parser_source:
    parser_source = parser_source.replace(old_base, new_base, 1)
elif "IFSA," not in parser_source:
    raise RuntimeError(
        "Could not patch base_modules for IFSA; inspect Ultralytics parse_model."
    )

# Re-exec parse_model in tasks dict so IFSA in tasks.__dict__ is visible.
exec(compile(parser_source, str(tasks_file), "exec"), tasks.__dict__)
print("\nIFSA registered with Ultralytics tasks.parse_model (base_modules).")

# ============================================================
# 13. BUILD MODEL FROM SCRATCH
# ============================================================

model = YOLO(str(MODEL_YAML), task="detect")
model.info()

net_cls = type(model.model)
net_module = sys.modules.get(net_cls.__module__)
if net_module is not None and getattr(net_module, net_cls.__qualname__, None) is not net_cls:
    print(
        f"Re-binding {net_cls.__module__}.{net_cls.__qualname__} "
        "to the class in use (stale kernel state detected)."
    )
    setattr(net_module, net_cls.__qualname__, net_cls)

if getattr(tasks, "IFSA", None) is not IFSA:
    IFSA.__module__ = tasks.__name__
    tasks.IFSA = IFSA

try:
    buffer = io.BytesIO()
    torch.save(model.model, buffer)
    del buffer
except Exception as exc:
    raise RuntimeError(
        "Checkpoint serialization test failed. Restart the Kaggle session "
        "(Run > Restart session) and run the script once from the top, "
        "without importlib.reload. Original error: " + str(exc)
    ) from exc

print("Checkpoint serialization smoke test passed.")
print("Training from scratch: no pretrained weights loaded.")
print(
    f"IFSA placement: P3={IFSA_AT_P3}, P4={IFSA_AT_P4}, P5={IFSA_AT_P5}, "
    f"lambda={IFS_LAMBDA}"
)

# ============================================================
# 14. TRAIN FOR 150 EPOCHS
# ============================================================

device = 0 if torch.cuda.is_available() else "cpu"

results = model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    patience=0,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=device,
    workers=2,
    seed=SEED,
    pretrained=False,
    project="/kaggle/working/runs/oralxrays9",
    name="yolo26n_ifsa_p3p4_scratch",
    plots=True,
)

# ============================================================
# 15. OUTPUT LOCATIONS
# ============================================================

save_dir = model.trainer.save_dir
print("\nTraining completed.")
print("Results directory:", save_dir)
print("Best weights:", save_dir / "weights" / "best.pt")
print("Last weights:", save_dir / "weights" / "last.pt")
print("The original test split is reserved for final evaluation.")
print(
    "\nNext ablations: (1) YOLO26n baseline yaml without IFSA, "
    "(2) IFSA@P3 only, (3) + confidence-level IFS on Detect scores."
)
