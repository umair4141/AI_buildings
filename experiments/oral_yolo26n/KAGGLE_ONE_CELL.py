# Paste this WHOLE file into ONE Kaggle notebook cell.
# Then run Exp-1 and Exp-2 one-by-one by setting RUN = 1 or RUN = 2 at the bottom.

# ===================== SETTINGS =====================
RUN = 1          # 1 = Exp-1 q-rung | 2 = Exp-2 plain gate | 0 = prepare only
IMGSZ = 1280
EPOCHS = 100
BATCH = 16
SEED = 0

KAGGLE_ROOT = "/kaggle/input/datasets/xxccfffffffffff/oral-dataset/oral dataset"
OUT_ROOT = "/kaggle/working/oral_yolo"
CFG_DIR = "/kaggle/working/cfgs"
RUNS = "/kaggle/working/runs"

# ===================== INSTALL =====================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "pillow"])

# ===================== IMPORTS =====================
import copy, json, shutil, zipfile
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
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".unzipped"
    if marker.exists():
        return
    print("Unzipping", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    marker.write_text("ok")

def find_coco_json(ann_dir, split):
    hits = sorted(Path(ann_dir).rglob("*.json"))
    for p in hits:
        if split in p.name.lower():
            return p
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
    label_dir = Path(label_dir); label_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(coco_json).read_text())
    cats = sorted(data["categories"], key=lambda c: c["id"])
    id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
    names[:] = [c["name"] for c in cats]
    images = {im["id"]: im for im in data["images"]}
    anns_by_img = {i: [] for i in images}
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) == 1:
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
            lines.append(f"{id_to_idx[ann['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        (label_dir / f"{Path(file_name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_ok += 1
    return n_ok

def link_images(image_dir, out_img_dir):
    out_img_dir = Path(out_img_dir); out_img_dir.mkdir(parents=True, exist_ok=True)
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

assert KAGGLE_ROOT.exists(), KAGGLE_ROOT
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
print("train labels", coco_to_yolo(train_json, train_img, OUT_ROOT / "labels/train", names))
print("val labels", coco_to_yolo(val_json, val_img, OUT_ROOT / "labels/val", names))
print("train images", link_images(train_img, OUT_ROOT / "images/train"))
print("val images", link_images(val_img, OUT_ROOT / "images/val"))

data_yaml = OUT_ROOT / "oral.yaml"
names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
data_yaml.write_text(f"path: {OUT_ROOT.resolve()}\ntrain: images/train\nval: images/val\n\nnames:\n{names_block}\n")
print("DATA", data_yaml, "classes", names)

# ===================== MODULES =====================

def _soft_sigma(raw, eps=1e-4):
    return F.softplus(raw) + eps

class _BaseDualGate(nn.Module):
    def __init__(self, c1, c2=None):
        super().__init__()
        self.c = int(c1) if c2 is None else int(c2)
        if c2 is None and c1 % 2 == 0:
            self.c = c1 // 2
        self.proj_a = nn.Conv2d(self.c, self.c, 1, bias=False)
        self.proj_b = nn.Conv2d(self.c, self.c, 1, bias=False)
        self.out = nn.Conv2d(2 * self.c, 2 * self.c, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def gate_from_evidence(self, ea, eb):
        raise NotImplementedError

    def forward(self, x):
        fa, fb = x
        ea, eb = self.proj_a(fa), self.proj_b(fb)
        ga, gb = self.gate_from_evidence(ea, eb)
        fused = torch.cat((fa * ga, fb * gb), 1)
        base = torch.cat((fa, fb), 1)
        return base + self.gamma * self.out(fused)

class GqROFConcat(_BaseDualGate):
    """Exp-1: Gaussian q-rung gate, learnable q."""
    def __init__(self, c1, c2=None):
        super().__init__(c1, c2)
        self.c_mu = nn.Parameter(torch.zeros(1, self.c, 1, 1))
        self.c_nu = nn.Parameter(torch.zeros(1, self.c, 1, 1))
        self.s_mu = nn.Parameter(torch.ones(1, self.c, 1, 1))
        self.s_nu = nn.Parameter(torch.ones(1, self.c, 1, 1))
        self.rho = nn.Parameter(torch.tensor(-1.3863))  # q≈2

    def _score(self, x):
        q = 1.0 + 5.0 * torch.sigmoid(self.rho)
        mu = torch.exp(-((x - self.c_mu) ** 2) / (2 * _soft_sigma(self.s_mu) ** 2)).clamp(1e-6, 1)
        nu = (1.0 - mu.pow(q)).clamp(min=0).pow(1.0 / q) * torch.exp(
            -((x - self.c_nu) ** 2) / (2 * _soft_sigma(self.s_nu) ** 2)
        )
        nu = nu.clamp(0, 1)
        return 0.5 * (mu.pow(q) - nu.pow(q) + 1.0)

    def gate_from_evidence(self, ea, eb):
        return 0.5 + self._score(ea), 0.5 + self._score(eb)

class PlainGateConcat(_BaseDualGate):
    """Exp-2: plain feature gate, NO q."""
    def __init__(self, c1, c2=None):
        super().__init__(c1, c2)
        self.gate_a = nn.Sequential(nn.Conv2d(self.c, self.c, 1), nn.Sigmoid())
        self.gate_b = nn.Sequential(nn.Conv2d(self.c, self.c, 1), nn.Sigmoid())

    def gate_from_evidence(self, ea, eb):
        return self.gate_a(ea), self.gate_b(eb)

# ===================== REGISTER =====================
import ultralytics.nn.tasks as tasks
from ultralytics.utils.torch_utils import make_divisible

tasks.GqROFConcat = GqROFConcat
tasks.PlainGateConcat = PlainGateConcat

if not getattr(tasks, "_gqrof_parse_wrapped", False):
    orig = tasks.parse_model

    def wrapped(d, ch, verbose=True):
        dd = copy.deepcopy(d)
        scale = dd.get("scale")
        depth, width, max_channels = 0.50, 0.25, 1024
        if dd.get("scales") and scale in dd["scales"]:
            depth, width, max_channels = dd["scales"][scale]
        ch0 = ch if isinstance(ch, list) else [ch]
        ch_track = [ch0[-1]]
        bb, hd = dd["backbone"], dd["head"]
        all_layers = bb + hd
        for i, (f, n, m, args) in enumerate(all_layers):
            m_name = m if isinstance(m, str) else getattr(m, "__name__", str(m))
            if m_name in {"GqROFConcat", "PlainGateConcat"}:
                c_in = ch_track[f[0]] if isinstance(f, list) else ch_track[f]
                all_layers[i] = [f, n, m_name, [c_in]]
                c2 = c_in * 2
            elif m_name == "Concat":
                c2 = sum(ch_track[x] for x in f)
            elif m_name in {"Detect"}:
                c2 = ch_track[-1]
            elif m_name.endswith("Upsample") or m_name == "nn.Upsample":
                c2 = ch_track[-1]
            elif args and isinstance(args[0], int):
                c2 = make_divisible(min(args[0], max_channels) * width, 8)
            else:
                c2 = ch_track[-1]
            ch_track.append(c2)
        dd["backbone"] = all_layers[: len(bb)]
        dd["head"] = all_layers[len(bb) :]
        g = orig.__globals__
        g["GqROFConcat"] = GqROFConcat
        g["PlainGateConcat"] = PlainGateConcat
        if "base_modules" in g:
            g["base_modules"] = frozenset(set(g["base_modules"]) | {GqROFConcat, PlainGateConcat})
        return orig(dd, ch, verbose)

    tasks.parse_model = wrapped
    tasks._gqrof_parse_wrapped = True

print("Registered GqROFConcat + PlainGateConcat")

# ===================== YAMLs =====================

def yolo26_yaml(merge_module: str) -> str:
    return f"""
nc: 9
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
  - [[-1, 6], 1, {merge_module}, []]
  - [-1, 2, C3k2, [512, True]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, {merge_module}, []]
  - [-1, 2, C3k2, [256, True]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, {merge_module}, []]
  - [-1, 2, C3k2, [512, True]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, {merge_module}, []]
  - [-1, 1, C3k2, [1024, True, 0.5, True]]

  - [[16, 19, 22], 1, Detect, [nc]]
""".lstrip()

exp1_yaml = CFG_DIR / "yolo26n_exp1_qrof.yaml"
exp2_yaml = CFG_DIR / "yolo26n_exp2_plain_gate.yaml"
exp1_yaml.write_text(yolo26_yaml("GqROFConcat"))
exp2_yaml.write_text(yolo26_yaml("PlainGateConcat"))

# ===================== TRAIN ONE EXPERIMENT =====================
from ultralytics import YOLO

if RUN == 0:
    print("Prepare only. Set RUN=1 or RUN=2 and re-run cell.")
elif RUN in (1, 2):
    yaml_path = str(exp1_yaml if RUN == 1 else exp2_yaml)
    run_name = "exp1_qrof_gate" if RUN == 1 else "exp2_plain_gate"
    print(f"Starting {run_name} with {yaml_path}")
    model = YOLO(yaml_path)
    try:
        model.load("yolo26n.pt")
    except Exception as e:
        print("load warning:", e)
    model.train(
        data=str(data_yaml),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        seed=SEED,
        workers=2,
        project=str(RUNS),
        name=run_name,
        optimizer="auto",
        patience=0,
        save=True,
        plots=True,
    )
    print("DONE:", RUNS / run_name)
else:
    raise ValueError("RUN must be 0, 1, or 2")

# How to run one by one:
# 1) Set RUN = 1  -> run cell  (Exp-1 q-rung, learnable q)
# 2) Set RUN = 2  -> run cell again (Exp-2 plain gate, no q)
