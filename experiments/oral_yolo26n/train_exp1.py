#!/usr/bin/env python3
"""Exp-1: YOLO26n + GqROFConcat (learnable q)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.oral_yolo26n.train_common import train

if __name__ == "__main__":
    cfg = ROOT / "experiments/oral_yolo26n/configs/yolo26n_exp1_qrof.yaml"
    train(cfg, run_name="exp1_qrof_gate")
