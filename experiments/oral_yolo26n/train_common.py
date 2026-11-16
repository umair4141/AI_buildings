from __future__ import annotations

import sys
from pathlib import Path


def setup_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def resolve_data_yaml() -> str:
    kaggle = Path("/kaggle/working/oral_yolo/oral.yaml")
    local = Path("data/oral_yolo/oral.yaml")
    if kaggle.exists():
        return str(kaggle)
    if local.exists():
        return str(local)
    raise FileNotFoundError("Run prepare_kaggle_data.py first.")


def write_yolo26_yaml(nc: int, out: Path) -> Path:
    text = f"""
nc: {nc}
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
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return out


def replace_concats(model, mode: int) -> int:
    from ultralytics.nn.modules import Concat

    from experiments.oral_yolo26n.modules.fusion_blocks import (
        GqROFConcat,
        PlainGateConcat,
        copy_ultralytics_meta,
    )

    seq = model.model.model
    n = 0
    for i, layer in enumerate(seq):
        if isinstance(layer, Concat):
            dim = getattr(layer, "d", 1)
            new = GqROFConcat(dimension=dim) if mode == 1 else PlainGateConcat(dimension=dim)
            seq[i] = copy_ultralytics_meta(new, layer)
            n += 1
    return n


def train(run: int, run_name: str, nc: int = 9) -> None:
    """run: 0 baseline, 1 q-rung, 2 plain gate."""
    setup_path()
    from ultralytics import YOLO

    data_yaml = resolve_data_yaml()
    cfg = Path("/kaggle/working/cfgs") if Path("/kaggle/working").exists() else Path("configs")
    yaml_path = write_yolo26_yaml(nc, cfg / "yolo26n_oral.yaml")

    model = YOLO(str(yaml_path))  # from scratch
    if run in (1, 2):
        n = replace_concats(model, run)
        print(f"Replaced {n} Concat modules for run={run}")

    model.train(
        data=data_yaml,
        imgsz=1280,
        epochs=100,
        batch=4,  # Kaggle 16GB: batch 16 @ 1280 OOMs in fp32
        seed=0,
        workers=2,
        project="/kaggle/working/runs" if Path("/kaggle/working").exists() else "runs/oral",
        name=run_name,
        optimizer="AdamW",
        lr0=0.001,
        patience=0,
        pretrained=False,
        amp=True,
        cache=False,
        save=True,
        plots=True,
    )
