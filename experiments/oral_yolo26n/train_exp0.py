#!/usr/bin/env python3
"""Exp-0: plain YOLO26n baseline (official Concat)."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.oral_yolo26n.train_common import train

if __name__ == "__main__":
    train(run=0, run_name="exp0_yolo26n_baseline")
