# Paste this WHOLE file into ONE Kaggle notebook cell.
# Run one experiment at a time by changing RUN, then re-run the cell.
#
# RUN = 1  -> Exp-1 proposed: Gaussian q-rung gate (learnable q)
# RUN = 2  -> Exp-2 control: plain feature gate (no q)
#
# No Exp-0 / plain YOLO26n baseline.
# From scratch, patience=0.

# ===================== SETTINGS =====================
RUN = 1          # 1=q-rung proposed | 2=plain-gate control
IMGSZ = 1280
EPOCHS = 100
BATCH = 16
SEED = 0

# Confirm which Kaggle dataset slug you attached:
KAGGLE_ROOT = "/kaggle/input/datasets/xxccfffffffffff/oral-dataset/oral dataset"
# KAGGLE_ROOT = "/kaggle/input/datasets/xainab22/oral-dataset/oral dataset"

OUT_ROOT = "/kaggle/working/oral_yolo"
CFG_DIR = "/kaggle/working/cfgs"
RUNS = "/kaggle/working/runs"

# ===================== INSTALL =====================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pillow"])

# ===================== IMPORTS =====================
import json, random, shutil, zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

KAGGLE_ROOT = Path(KAGGLE_ROOT)
OUT_ROOT = Path(OUT_ROOT)
CFG_DIR = Path(CFG_DIR)
RUNS = Path(RUNS)
CFG_DIR.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

# ===================== DATA PREP =====================
CLASS_NAMES_FALLBACK = [
    "Apical Periodontitis", "Decay", "Wisdom Tooth", "Missing Tooth",
    "Dental Filling", "Root Canal Filling", "Implant", "Porcelain Crown", "Ceramic Bridge",
]

def unzip_if_needed(zip_path, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".unzipped"
    if marker.exists():
        return
    print("Unzipping", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    marker.write_text("ok")

def find_coco_json(ann_dir, split):
    hits = sorted(Path(ann_dir).rglob("*.json"))
    preferred = [p for p in hits if split in p.name.lower()]
    if preferred:
        return preferred[0]
    if hits:
        return hits[0]
    raise FileNotFoundError(ann_dir)

def find_image_dir(img_root, split):
    img_root = Path(img_root)
    direct = img_root / f"{split}2017"
    if direct.is_dir():
        return direct
    for p in img_root.rglob("*"):
        if p.is_dir() and split in p.name.lower():
            return p
    return img_root

def coco_to_yolo(coco_json, image_dir, label_dir, names):
    """Convert COCO boxes to YOLO txt. Skips unknown category_id (e.g. id 0 not in categories)."""
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(coco_json).read_text())
    cats = sorted(data["categories"], key=lambda c: c["id"])
    id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names[:] = [c["name"] for c in cats]

    images = {im["id"]: im for im in data["images"]}
    anns_by_img = {i: [] for i in images}
    skipped_cat = 0
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        if ann["category_id"] not in id_to_idx:
            skipped_cat += 1
            continue
        anns_by_img[ann["image_id"]].append(ann)

    n_ok = 0
    for img_id, im in images.items():
        file_name = im["file_name"]
        src = Path(image_dir) / file_name
        if not src.exists():
            src = Path(image_dir) / Path(file_name).name
        if not src.exists():
            continue
        w, h = im.get("width"), im.get("height")
        if not w or not h:
            with Image.open(src) as pic:
                w, h = pic.size
        lines = []
        for ann in anns_by_img[img_id]:
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            cx, cy, nw, nh = (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h
            cls = id_to_idx[ann["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (label_dir / f"{Path(file_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        n_ok += 1
    print(f"coco_to_yolo: images={n_ok}, skipped_unknown_category_id={skipped_cat}")
    return n_ok

def link_images(image_dir, out_img_dir):
    out_img_dir = Path(out_img_dir)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in Path(image_dir).rglob("*"):
        if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
            continue
        dst = out_img_dir / src.name
        if not dst.exists():
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
        n += 1
    return n

def split_val_into_val_test(out_root: Path, seed: int = 0, test_ratio: float = 0.5):
    """Hold out part of val as test so final score is not the same as model selection."""
    rng = random.Random(seed)
    img_val = out_root / "images/val"
    lab_val = out_root / "labels/val"
    img_test = out_root / "images/test"
    lab_test = out_root / "labels/test"
    img_test.mkdir(parents=True, exist_ok=True)
    lab_test.mkdir(parents=True, exist_ok=True)

    # If test already populated, keep it
    existing = list(img_test.glob("*"))
    if existing:
        print("test split already exists:", len(existing))
        return

    stems = sorted({p.stem for p in lab_val.glob("*.txt")})
    rng.shuffle(stems)
    n_test = max(1, int(len(stems) * test_ratio))
    test_stems = set(stems[:n_test])

    moved = 0
    for stem in test_stems:
        for folder, dst_folder in (
            (img_val, img_test),
            (lab_val, lab_test),
        ):
            matches = list(folder.glob(stem + ".*"))
            for src in matches:
                dst = dst_folder / src.name
                if src.exists():
                    shutil.move(str(src), str(dst))
                    moved += 1
    print(f"Moved {len(test_stems)} images to test (files touched={moved})")

assert KAGGLE_ROOT.exists(), f"Dataset not found: {KAGGLE_ROOT}"
raw = OUT_ROOT / "_raw"
unzip_if_needed(KAGGLE_ROOT / "annotations.zip", raw / "annotations")
unzip_if_needed(KAGGLE_ROOT / "train2017.zip", raw / "train2017")
unzip_if_needed(KAGGLE_ROOT / "val2017.zip", raw / "val2017")

train_img = find_image_dir(raw / "train2017", "train")
val_img = find_image_dir(raw / "val2017", "val")
train_json = find_coco_json(raw / "annotations", "train")
val_json = find_coco_json(raw / "annotations", "val")
print("JSON/IMG", train_json, val_json, train_img, val_img)

names = list(CLASS_NAMES_FALLBACK)
print("train", coco_to_yolo(train_json, train_img, OUT_ROOT / "labels/train", names))
print("val", coco_to_yolo(val_json, val_img, OUT_ROOT / "labels/val", names))
print("train images", link_images(train_img, OUT_ROOT / "images/train"))
print("val images", link_images(val_img, OUT_ROOT / "images/val"))
split_val_into_val_test(OUT_ROOT, seed=SEED, test_ratio=0.5)

data_yaml = OUT_ROOT / "oral.yaml"
names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
data_yaml.write_text(
    f"path: {OUT_ROOT.resolve()}\n"
    f"train: images/train\n"
    f"val: images/val\n"
    f"test: images/test\n\n"
    f"names:\n{names_block}\n"
)
print("DATA", data_yaml, "classes", names)

# ===================== FUSION MODULES (Concat-compatible) =====================
def _soft_sigma(raw, eps=1e-4):
    return F.softplus(raw) + eps


class _CrossStreamGate(nn.Module):
    """
    Drop-in Concat replacement.
    __init__(dimension=1) matches Ultralytics Concat.
    Supports unequal channels via lazy init.
    Output = cat(gated_a, gated_b) along dim — same rank as Concat.
    Gate range is always ~[0, 2] via 2*sigmoid(...), init near 1 (identity).
    """

    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self._inited = False
        self.logit_w = nn.Parameter(torch.tensor(0.0))  # stream mix weight

    def _lazy_init(self, fa, fb):
        ca, cb = fa.shape[1], fb.shape[1]
        self.ev_a = nn.Conv2d(ca, 1, kernel_size=1, bias=True)
        self.ev_b = nn.Conv2d(cb, 1, kernel_size=1, bias=True)
        self.cha = nn.Conv2d(1, ca, kernel_size=1, bias=True)
        self.chb = nn.Conv2d(1, cb, kernel_size=1, bias=True)
        # bias 0 => 2*sigmoid(0)=1 => identity gate at start
        nn.init.zeros_(self.ev_a.weight)
        nn.init.zeros_(self.ev_b.weight)
        nn.init.zeros_(self.ev_a.bias)
        nn.init.zeros_(self.ev_b.bias)
        nn.init.zeros_(self.cha.weight)
        nn.init.zeros_(self.chb.weight)
        nn.init.zeros_(self.cha.bias)
        nn.init.zeros_(self.chb.bias)
        self._init_extra(ca, cb)
        self.to(fa.device)
        self._inited = True

    def _init_extra(self, ca, cb):
        pass

    def fused_evidence(self, fa, fb):
        raise NotImplementedError

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            raise ValueError(f"{self.__class__.__name__} expects 2 feature maps")
        fa, fb = x[0], x[1]
        if not self._inited:
            self._lazy_init(fa, fb)
        S = self.fused_evidence(fa, fb)  # [B,1,H,W] in (~0,1) or logits->mapped inside
        ga = 2.0 * torch.sigmoid(self.cha(S))
        gb = 2.0 * torch.sigmoid(self.chb(S))
        return torch.cat((fa * ga, fb * gb), self.d)


class GqROFConcat(_CrossStreamGate):
    """Exp-1: cross-stream Gaussian q-rung aggregation, then matched 2*sigmoid gate."""

    def _init_extra(self, ca, cb):
        self.c_mu = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.c_nu = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.s_mu = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.s_nu = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.rho = nn.Parameter(torch.tensor(-1.3863))  # q≈2

    def _orthopair(self, e):
        q = 1.0 + 5.0 * torch.sigmoid(self.rho)
        mu = torch.exp(-((e - self.c_mu) ** 2) / (2.0 * _soft_sigma(self.s_mu) ** 2)).clamp(1e-6, 1.0)
        nu = (1.0 - mu.pow(q)).clamp(min=0.0).pow(1.0 / q) * torch.exp(
            -((e - self.c_nu) ** 2) / (2.0 * _soft_sigma(self.s_nu) ** 2)
        )
        nu = nu.clamp(0.0, 1.0)
        s = 0.5 * (mu.pow(q) - nu.pow(q) + 1.0)
        return mu, nu, s

    def fused_evidence(self, fa, fb):
        ea = self.ev_a(fa)
        eb = self.ev_b(fb)
        mu_a, nu_a, _ = self._orthopair(ea)
        mu_b, nu_b, _ = self._orthopair(eb)
        w = torch.sigmoid(self.logit_w)
        # Algebraic weighted mix of orthopair degrees, then score (cross-stream fusion)
        mu = w * mu_a + (1.0 - w) * mu_b
        nu = w * nu_a + (1.0 - w) * nu_b
        q = 1.0 + 5.0 * torch.sigmoid(self.rho)
        # Renormalize lightly to keep numerical stability inside orthopair ball
        norm = (mu.pow(q) + nu.pow(q)).clamp(min=1e-6)
        scale = torch.clamp(norm, max=1.0) / norm
        mu = (mu * scale.pow(1.0 / q)).clamp(1e-6, 1.0)
        nu = (nu * scale.pow(1.0 / q)).clamp(0.0, 1.0)
        S = 0.5 * (mu.pow(q) - nu.pow(q) + 1.0)
        return S


class PlainGateConcat(_CrossStreamGate):
    """Exp-2 Control A: same cross-stream mix + 2*sigmoid gates, no q-rung."""

    def fused_evidence(self, fa, fb):
        ea = self.ev_a(fa)
        eb = self.ev_b(fb)
        w = torch.sigmoid(self.logit_w)
        # Map mixed evidence to (0,1) so cha/chb see same scale family as S
        return torch.sigmoid(w * ea + (1.0 - w) * eb)


# ===================== BUILD OFFICIAL YOLO26n + REPLACE CONCAT =====================
YOLO26_YAML = f"""
# Official YOLO26 topology; nc=9 for oral dataset. Concat kept on purpose.
nc: {len(names)}
end2end: True
reg_max: 1
scale: n
scales:
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5, 3, True]]
  - [-1, 2, C2PSA, [1024]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, True]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 1, C3k2, [1024, True, 0.5, True]]

  - [[16, 19, 22], 1, Detect, [nc]]
""".lstrip()

yaml_path = CFG_DIR / "yolo26n_oral.yaml"
yaml_path.write_text(YOLO26_YAML)

from ultralytics import YOLO
from ultralytics.nn.modules import Concat

def copy_ultralytics_meta(dst, src):
    for key in ("f", "i", "type", "np"):
        if hasattr(src, key):
            setattr(dst, key, getattr(src, key))
    return dst

def replace_concats(model, mode: int) -> int:
    """Replace Ultralytics Concat with gated fusion. mode=1 q-rung, mode=2 plain."""
    seq = model.model.model  # nn.Sequential of DetectionModel
    n = 0
    for i, layer in enumerate(seq):
        if isinstance(layer, Concat):
            dim = getattr(layer, "d", 1)
            new = GqROFConcat(dimension=dim) if mode == 1 else PlainGateConcat(dimension=dim)
            seq[i] = copy_ultralytics_meta(new, layer)
            n += 1
    return n

def smoke_forward(model):
    model.model.eval()
    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        _ = model.model(x)
    print("smoke_forward OK")

run_names = {
    1: "exp1_qrof_gate",
    2: "exp2_plain_gate",
}
assert RUN in run_names, "RUN must be 1 or 2"

print(f"Building YOLO26n + gate (from scratch), RUN={RUN} ({run_names[RUN]})")
model = YOLO(str(yaml_path))  # architecture only — NO pretrained load
n_rep = replace_concats(model, RUN)
print(f"Replaced {n_rep} Concat modules")
smoke_forward(model)

model.train(
    data=str(data_yaml),
    imgsz=IMGSZ,
    epochs=EPOCHS,
    batch=BATCH,
    seed=SEED,
    workers=2,
    project=str(RUNS),
    name=run_names[RUN],
    optimizer="auto",
    patience=0,
    pretrained=False,
    save=True,
    plots=True,
)
print("TRAIN DONE:", RUNS / run_names[RUN])

# Final score on held-out test (not used for early stopping — patience=0 anyway)
best = RUNS / run_names[RUN] / "weights" / "best.pt"
if best.exists():
    m = YOLO(str(best))
    print("TEST metrics:")
    m.val(data=str(data_yaml), split="test", imgsz=IMGSZ, plots=False)

# One-by-one:
# 1) RUN=1 proposed q-rung
# 2) RUN=2 plain-gate control
