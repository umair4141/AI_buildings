# ==========================================================================================
# Oral detection — YOLO26n + neck gate
# Paste into ONE Kaggle cell. Run Exp-1, then change RUN and run Exp-2.
#
# RUN = 1  -> Exp-1: Gaussian q-rung gate (learnable q)     = proposed method
# RUN = 2  -> Exp-2: plain Conv+sigmoid gate (no q)         = control
#
# Both: from scratch, same data/seed, no early stopping.
# Dataset (Kaggle Input): annotations.zip, train2017.zip, val2017.zip
# ==========================================================================================

# -------------------- 1) SETTINGS --------------------
RUN = 1                 # 1 = Exp-1 q-rung | 2 = Exp-2 plain gate
IMGSZ = 1280
EPOCHS = 100
BATCH = 4
SEED = 0
Q_INIT = 2.0            # start q (Pythagorean)
LEARN_Q = True
Q_MAX = 6.0

# Put your Kaggle dataset path here (whichever exists in Add Input)
DATA_CANDIDATES = [
    "/kaggle/input/datasets/xainab22/oral-dataset",
    "/kaggle/input/datasets/xxccfffffffffff/oral-dataset/oral dataset",
]
PASSWORD = "CVPR2024-OralXrays-9"   # zip password from OralXrays-9

WORK = "/kaggle/working"
RAW = f"{WORK}/raw"
EXTRACT = f"{WORK}/extracted"
YOLO_DATA = f"{WORK}/oral_yolo"
RESULTS = f"{WORK}/results"
RUNS = f"{WORK}/runs"

# -------------------- 2) INSTALL --------------------
import subprocess, sys
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.4", "pyzipper", "pyyaml", "pandas", "matplotlib"],
    check=True,
)

import json, math, os, random, shutil, zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyzipper
import torch
import torch.nn as nn
import torch.nn.functional as F
import ultralytics
import yaml
from ultralytics import YOLO

print("Ultralytics", ultralytics.__version__, "| GPU:",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORK, RAW, EXTRACT, YOLO_DATA, RESULTS, RUNS = map(Path, [WORK, RAW, EXTRACT, YOLO_DATA, RESULTS, RUNS])
for p in (RAW, EXTRACT, YOLO_DATA, RESULTS, RUNS):
    p.mkdir(parents=True, exist_ok=True)

# Find attached dataset folder
DATASET = next((Path(p) for p in DATA_CANDIDATES if Path(p).exists()), None)
if DATASET is None:
    # fallback: search whole /kaggle/input for the three zips
    hits = list(Path("/kaggle/input").rglob("annotations.zip"))
    DATASET = hits[0].parent if hits else None
assert DATASET is not None, "Attach oral dataset Input (annotations/train2017/val2017 zips)."
print("Dataset folder:", DATASET)

RUN_NAMES = {1: "exp1_qrof_gate", 2: "exp2_plain_gate"}
YAML_FILES = {
    1: WORK / "yolo26n_exp1_qrof.yaml",
    2: WORK / "yolo26n_exp2_plain.yaml",
}
assert RUN in RUN_NAMES, "Set RUN to 1 or 2 only."

# -------------------- 3) GATE MODULES --------------------
# Neck merge: two maps fa, fb -> gated concat (same shape as Concat: ca+cb channels)
# Exp-1 gate from Gaussian q-rung score; Exp-2 gate from plain sigmoid.
# gamma starts at 0 => first step equals plain Concat.


def soft_sigma(x, eps=1e-4):
    return F.softplus(x) + eps


class DualGateConcat(nn.Module):
    """Replace Concat([fa, fb]). Builds 1x1 layers on first forward (when channels known)."""

    def __init__(self, dimension=1):
        super().__init__()
        self.dim = dimension
        self.ready = False

    def make_gate_params(self, ca, cb):
        raise NotImplementedError

    def compute_gates(self, ea, eb):
        raise NotImplementedError

    def build(self, fa, fb):
        ca, cb = fa.shape[1], fb.shape[1]
        self.proj_a = nn.Conv2d(ca, ca, 1, bias=False)
        self.proj_b = nn.Conv2d(cb, cb, 1, bias=False)
        self.out = nn.Conv2d(ca + cb, ca + cb, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))  # 0 => identity Concat
        self.make_gate_params(ca, cb)
        self.to(device=fa.device, dtype=torch.float32)
        self.ready = True

    def forward(self, xs):
        fa, fb = xs
        if not self.ready:
            self.build(fa, fb)
        ga, gb = self.compute_gates(self.proj_a(fa), self.proj_b(fb))
        gated = torch.cat((fa * ga.to(fa.dtype), fb * gb.to(fb.dtype)), self.dim)
        base = torch.cat((fa, fb), self.dim)
        return base + self.gamma.to(base.dtype) * self.out(gated).to(base.dtype)


class GqROFConcat(DualGateConcat):
    """Exp-1: Gaussian q-rung orthopair gate. YAML: [dim, q_init, learn_q, q_max]"""

    def __init__(self, dimension=1, q_init=2.0, learn_q=1, q_max=6.0):
        super().__init__(dimension)
        self.q_max = float(q_max)
        # q = 1 + (q_max-1)*sigmoid(rho); choose rho so q starts at q_init
        r = (float(q_init) - 1.0) / (self.q_max - 1.0)
        rho0 = torch.tensor(math.log(r / (1.0 - r)))
        if int(learn_q):
            self.rho = nn.Parameter(rho0)
        else:
            self.register_buffer("rho", rho0)

    @property
    def q(self):
        return 1.0 + (self.q_max - 1.0) * torch.sigmoid(self.rho.float())

    def make_gate_params(self, ca, cb):
        for tag, c in (("a", ca), ("b", cb)):
            for name, val in (("c_mu", 0.0), ("c_nu", 0.0), ("s_mu", 1.0), ("s_nu", 1.0)):
                setattr(self, f"{name}_{tag}", nn.Parameter(torch.full((1, c, 1, 1), val)))

    def score(self, e, tag):
        e = e.float()
        q = self.q
        get = lambda n: getattr(self, f"{n}_{tag}").float()
        mu = torch.exp(-((e - get("c_mu")) ** 2) / (2 * soft_sigma(get("s_mu")) ** 2)).clamp(1e-6, 1)
        nu = (1.0 - mu.pow(q)).clamp(min=0).pow(1.0 / q)
        nu = nu * torch.exp(-((e - get("c_nu")) ** 2) / (2 * soft_sigma(get("s_nu")) ** 2))
        nu = nu.clamp(0, 1)
        # score in [0,1]; gate = 0.5 + score => [0.5, 1.5]
        return 0.5 * (mu.pow(q) - nu.pow(q) + 1.0)

    def compute_gates(self, ea, eb):
        with torch.autocast(device_type=ea.device.type, enabled=False):
            return 0.5 + self.score(ea, "a"), 0.5 + self.score(eb, "b")


class PlainGateConcat(DualGateConcat):
    """Exp-2: plain gate, no q. Same residual / same gate range as Exp-1."""

    def make_gate_params(self, ca, cb):
        self.gate_a = nn.Conv2d(ca, ca, 1)
        self.gate_b = nn.Conv2d(cb, cb, 1)

    def compute_gates(self, ea, eb):
        return 0.5 + torch.sigmoid(self.gate_a(ea)), 0.5 + torch.sigmoid(self.gate_b(eb))


def register_gate(run):
    """Tell Ultralytics about our modules. Also alias Concat so channel math stays ca+cb."""
    import ultralytics.nn.tasks as tasks

    tasks.GqROFConcat = GqROFConcat
    tasks.PlainGateConcat = PlainGateConcat
    tasks.Concat = GqROFConcat if run == 1 else PlainGateConcat


def unregister_gate():
    import ultralytics.nn.tasks as tasks
    from ultralytics.nn.modules import Concat

    tasks.Concat = Concat
    for name in ("GqROFConcat", "PlainGateConcat"):
        if hasattr(tasks, name):
            delattr(tasks, name)


def gate_layers(model):
    return [m for m in model.modules() if isinstance(m, DualGateConcat)]


# -------------------- 4) MODEL YAML (YOLO26n, only merge row differs) --------------------
YAML_BODY = """# YOLO26n oral | {title}
nc: 9
end2end: True
reg_max: 1
scale: n
scales:
  n: [0.50, 0.25, 1024]

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
  - [[-1, 6], 1, {merge}]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, {merge}]
  - [-1, 2, C3k2, [256, True]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, {merge}]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, {merge}]
  - [-1, 1, C3k2, [1024, True, 0.5, True]]
  - [[16, 19, 22], 1, Detect, [nc]]
"""

merge_rows = {
    1: f"GqROFConcat, [1, {Q_INIT}, {int(LEARN_Q)}, {Q_MAX}]",
    2: "PlainGateConcat, [1]",
}
titles = {1: "Exp-1 q-rung gate", 2: "Exp-2 plain gate"}
for r, path in YAML_FILES.items():
    path.write_text(YAML_BODY.format(title=titles[r], merge=merge_rows[r]), encoding="utf-8")
print("YAML ready:", YAML_FILES[RUN].name)

# -------------------- 5) DATA: unzip + COCO -> YOLO --------------------

def find_zips():
    """Find annotations / train2017 / val2017 zip under dataset or /kaggle/input."""
    need = {"annotations", "train2017", "val2017"}
    found = {}
    roots = [DATASET, Path("/kaggle/input"), RAW]
    for root in roots:
        if not root.exists():
            continue
        for z in root.rglob("*.zip"):
            stem = z.stem
            if stem in need and stem not in found:
                found[stem] = z
    missing = need - set(found)
    assert not missing, f"Missing zips: {sorted(missing)}"
    return found


def unzip_all(zips):
    for name, zpath in zips.items():
        out = EXTRACT / name
        if out.exists() and any(out.iterdir()):
            continue
        print("Extracting", zpath)
        out.mkdir(parents=True, exist_ok=True)
        try:
            with pyzipper.AESZipFile(zpath) as zf:
                zf.setpassword(PASSWORD.encode())
                zf.extractall(out)
        except Exception:
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(out)


def coco_to_records():
    """Read COCO jsons -> list of dict(img, boxes, is_test). Official val = TEST."""
    # index images by filename
    by_name = defaultdict(list)
    for p in EXTRACT.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            by_name[p.name].append(p)
    # also search input if images already unpacked
    for p in Path("/kaggle/input").rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            by_name[p.name].append(p)

    records, class_names, name_to_idx = [], None, None
    for jpath in sorted(EXTRACT.rglob("*.json")):
        data = json.loads(jpath.read_text(encoding="utf-8"))
        if not {"images", "annotations", "categories"} <= set(data):
            continue
        if class_names is None:
            cats = sorted(data["categories"], key=lambda c: c["id"])
            class_names = [c["name"] for c in cats]
            name_to_idx = {c["name"]: i for i, c in enumerate(cats)}
        id_to_name = {c["id"]: c["name"] for c in data["categories"]}
        anns = defaultdict(list)
        for a in data["annotations"]:
            anns[a["image_id"]].append(a)

        tag = jpath.stem.lower().replace("instances_", "")
        is_test = ("val" in tag) or ("test" in tag)

        for im in data["images"]:
            cands = by_name.get(Path(im["file_name"]).name, [])
            if len(cands) > 1:
                cands = [c for c in cands if tag in [x.lower() for x in c.parts]] or cands
            if not cands:
                continue
            W, H = im["width"], im["height"]
            boxes = []
            for a in anns[im["id"]]:
                cname = id_to_name.get(a["category_id"])
                if cname not in name_to_idx:  # skip unknown ids (e.g. category_id 0)
                    continue
                x, y, w, h = a["bbox"]
                x0, y0 = max(0, x), max(0, y)
                x1, y1 = min(W, x + w), min(H, y + h)
                if x1 - x0 < 1 or y1 - y0 < 1:
                    continue
                boxes.append(
                    (
                        name_to_idx[cname],
                        (x0 + x1) / 2 / W,
                        (y0 + y1) / 2 / H,
                        (x1 - x0) / W,
                        (y1 - y0) / H,
                    )
                )
            records.append(dict(img=cands[0], boxes=boxes, is_test=is_test))

    assert records and class_names, "No COCO records found after extract."
    assert len(class_names) == 9, f"Expected 9 classes, got {class_names}"
    return records, class_names


def write_yolo_dataset(records, class_names):
    """Train = 90% of train2017, Val = 10% of train2017, Test = official val2017."""
    rng = random.Random(42)
    trainval = [r for r in records if not r["is_test"]]
    test = [r for r in records if r["is_test"]]
    rng.shuffle(trainval)
    n_val = max(1, int(0.10 * len(trainval)))
    splits = {"val": trainval[:n_val], "train": trainval[n_val:], "test": test}

    if YOLO_DATA.exists():
        shutil.rmtree(YOLO_DATA)
    for split, recs in splits.items():
        (YOLO_DATA / "images" / split).mkdir(parents=True)
        (YOLO_DATA / "labels" / split).mkdir(parents=True)
        for i, r in enumerate(recs):
            stem = f"{i:05d}_{r['img'].stem}".replace(" ", "_")
            dst = YOLO_DATA / "images" / split / (stem + r["img"].suffix.lower())
            try:
                os.symlink(r["img"].resolve(), dst)
            except OSError:
                shutil.copy2(r["img"], dst)
            lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in r["boxes"]]
            (YOLO_DATA / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    data_yaml = YOLO_DATA / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(YOLO_DATA),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(class_names),
                "names": {i: n for i, n in enumerate(class_names)},
            },
            sort_keys=False,
        )
    )
    print({k: len(v) for k, v in splits.items()})
    counts = {s: Counter(b[0] for r in recs for b in r["boxes"]) for s, recs in splits.items()}
    print(pd.DataFrame(counts, index=range(len(class_names))).fillna(0).astype(int).assign(name=class_names))
    return data_yaml


print("Preparing data...")
unzip_all(find_zips())
records, CLASS_NAMES = coco_to_records()
DATA_YAML = write_yolo_dataset(records, CLASS_NAMES)

# -------------------- 6) TRAIN (from scratch) --------------------
name = RUN_NAMES[RUN]
print(f"\n===== TRAIN {name} | imgsz={IMGSZ} batch={BATCH} epochs={EPOCHS} scratch =====")

unregister_gate()
register_gate(RUN)
model = YOLO(str(YAML_FILES[RUN]))

# Log gamma / q each epoch for Exp-1
history = []


def on_epoch_end(trainer):
    m = getattr(trainer.model, "module", trainer.model)
    row = {"epoch": trainer.epoch + 1}
    for i, layer in enumerate(gate_layers(m)):
        row[f"gamma{i}"] = float(layer.gamma.detach())
        if isinstance(layer, GqROFConcat):
            row[f"q{i}"] = float(layer.q.detach())
    history.append(row)


def on_train_end(trainer):
    if not history:
        return
    df = pd.DataFrame(history)
    df.to_csv(RESULTS / f"{name}_history.csv", index=False)
    qcols = [c for c in df.columns if c.startswith("q")]
    if qcols:
        df.plot(x="epoch", y=qcols, figsize=(7, 3), title=f"learned q ({name})")
        plt.tight_layout()
        plt.savefig(RESULTS / f"{name}_q.png", dpi=150)
        plt.close()
        print("final q:", {c: round(float(df[c].iloc[-1]), 4) for c in qcols})
    print("final gamma:", {c: round(float(df[c].iloc[-1]), 4) for c in df.columns if c.startswith("gamma")})


model.add_callback("on_train_epoch_end", on_epoch_end)
model.add_callback("on_train_end", on_train_end)

model.train(
    data=str(DATA_YAML),
    name=name,
    project=str(RUNS),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    seed=SEED,
    optimizer="AdamW",
    lr0=0.001,
    patience=0,
    pretrained=False,
    deterministic=True,
    workers=2,
    exist_ok=True,
    plots=True,
)
unregister_gate()
print("TRAIN DONE:", RUNS / name)

# -------------------- 7) TEST EVAL (all finished runs) --------------------
rows, per_class = [], {}
for r, rname in RUN_NAMES.items():
    best = RUNS / rname / "weights" / "best.pt"
    if not best.exists():
        print("no weights yet:", rname)
        continue
    unregister_gate()
    register_gate(r)
    m = YOLO(str(best))
    res = m.val(data=str(DATA_YAML), split="test", imgsz=IMGSZ, batch=BATCH, plots=False, verbose=False)
    qs = [round(float(g.q.detach()), 4) for g in gate_layers(m.model) if isinstance(g, GqROFConcat)]
    rows.append(
        dict(
            model=rname,
            precision=float(res.box.mp),
            recall=float(res.box.mr),
            mAP50=float(res.box.map50),
            mAP50_95=float(res.box.map),
            learned_q=qs or None,
        )
    )
    per_class[rname] = pd.Series(res.box.maps, index=CLASS_NAMES)
unregister_gate()

if rows:
    summary = pd.DataFrame(rows).set_index("model")
    print(summary.round(4))
    summary.to_csv(RESULTS / "summary_test.csv")
    pc = pd.DataFrame(per_class).round(4)
    print(pc)
    pc.to_csv(RESULTS / "per_class_ap50_95.csv")
    pc.plot.bar(figsize=(10, 4), title="Per-class AP50-95 (test)")
    plt.tight_layout()
    plt.savefig(RESULTS / "per_class_ap.png", dpi=150)
    plt.show()

print("\nNext: set RUN = 2 and run this cell again." if RUN == 1 else "\nBoth runs ready if Exp-1 weights exist.")
