#!/usr/bin/env bash
# Run on a Kaggle GPU notebook / script after copying this repo into /kaggle/working/repo
set -euo pipefail

REPO="${REPO:-/kaggle/working/repo}"
cd "$REPO"

pip -q install ultralytics

python experiments/oral_yolo26n/prepare_kaggle_data.py

echo "===== EXP-1: q-rung proposed (learnable q) ====="
python experiments/oral_yolo26n/train_exp1.py

echo "===== EXP-2: plain feature gate (no q) ====="
python experiments/oral_yolo26n/train_exp2.py

echo "Done. Compare:"
echo "  /kaggle/working/runs/exp1_qrof_gate"
echo "  /kaggle/working/runs/exp2_plain_gate"
