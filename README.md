# Oral YOLO26n — first two experiments

Kaggle dataset:
`/kaggle/input/datasets/xxccfffffffffff/oral-dataset/oral dataset/`

## Two experiments only

| Exp | Name | YAML | Module at 4 neck merges |
|---|---|---|---|
| **1** | Proposed q-rung | `experiments/oral_yolo26n/configs/yolo26n_exp1_qrof.yaml` | `GqROFConcat` + **learnable q** |
| **2** | Plain gate control | `experiments/oral_yolo26n/configs/yolo26n_exp2_plain_gate.yaml` | `PlainGateConcat` (**no q**) |

Base architecture = official **YOLO26n** (`end2end: True`, `reg_max: 1`).  
Train both at **`imgsz=1280`**, `seed=0`.

## Kaggle commands

```bash
# 1) put this repo in /kaggle/working/repo
cd /kaggle/working/repo
pip install ultralytics

# 2) prepare COCO zips -> YOLO labels
python experiments/oral_yolo26n/prepare_kaggle_data.py

# 3) run the two experiments
python experiments/oral_yolo26n/train_exp1.py
python experiments/oral_yolo26n/train_exp2.py
```

Or: `bash experiments/oral_yolo26n/kaggle_run.sh`

Details: `experiments/oral_yolo26n/README.md`
