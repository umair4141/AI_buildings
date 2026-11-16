from __future__ import annotations

import sys
from pathlib import Path


def setup_path() -> Path:
    """Allow `experiments.oral_yolo26n...` imports on Kaggle and local."""
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
    raise FileNotFoundError(
        "Dataset YAML not found. Run prepare_kaggle_data.py first "
        "(creates /kaggle/working/oral_yolo/oral.yaml)."
    )


def train(model_yaml: Path, run_name: str) -> None:
    setup_path()
    from experiments.oral_yolo26n.modules.register_ultralytics import register

    register()

    from ultralytics import YOLO

    data_yaml = resolve_data_yaml()
    model = YOLO(str(model_yaml))
    # Load overlapping YOLO26n COCO weights into matching layers
    try:
        model.load("yolo26n.pt")
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not load yolo26n.pt ({exc}); training from YAML init.")

    model.train(
        data=data_yaml,
        imgsz=1280,
        epochs=100,
        batch=16,
        seed=0,
        workers=4,
        project="/kaggle/working/runs" if Path("/kaggle/working").exists() else "runs/oral",
        name=run_name,
        optimizer="auto",
        patience=0,
        save=True,
        plots=True,
    )
