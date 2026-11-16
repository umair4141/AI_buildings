# Two YOLO26n experiments (OralXrays-9 style)

**Dataset (Kaggle):**
```text
/kaggle/input/datasets/xxccfffffffffff/oral-dataset/oral dataset/
  README (3).md
  annotations.zip
  train2017.zip
  val2017.zip
```

**Model base:** official Ultralytics `yolo26n.yaml` (P3/P4/P5 Detect, `end2end: True`, `reg_max: 1`).

**Image size:** `imgsz=1280` (closest square setting to OralXrays-9 paper size **1333×800**).

---

## Exp-1 — Proposed (q-rung gated fusion)

| Item | Value |
|---|---|
| Name | `exp1_qrof_gate` |
| YAML | `configs/yolo26n_exp1_qrof.yaml` |
| What | YOLO26n neck Concat → **GqROFConcat** (Gaussian q-rung gate, **learnable q** per merge) |
| Goal | Measure q-rung fuzzy gate |

```bash
python experiments/oral_yolo26n/train_exp1.py
```

---

## Exp-2 — Plain feature gate (control A)

| Item | Value |
|---|---|
| Name | `exp2_plain_gate` |
| YAML | `configs/yolo26n_exp2_plain_gate.yaml` |
| What | Same merge points → **PlainGateConcat** (gate from projected features only; **no μ/ν/π, no q**) |
| Goal | Isolate whether gain is from **gate** or from **q-rung** |

```bash
python experiments/oral_yolo26n/train_exp2.py
```

---

## Keep identical across both runs

- YOLO26n scale  
- `imgsz=1280`  
- same prepared YOLO labels  
- same epochs / batch / seed (`seed=0`)  
- same 4 merge locations in the neck  

**Only difference:** q-rung fuzzy gate vs plain feature gate.

---

## How to read results

| Outcome | Meaning |
|---|---|
| Exp-1 ≫ Exp-2 | q-rung helps |
| Exp-1 ≈ Exp-2 | only the gate helps |
| Exp-1 < Exp-2 | q-rung unstable / not helping |

Report: mAP50, mAP50-95, AP for **Decay** and **Apical Periodontitis**.

---

## Prepare data once (Kaggle)

```bash
python experiments/oral_yolo26n/prepare_kaggle_data.py
```

Writes under `/kaggle/working/oral_yolo/`:
- `images/train`, `images/val`
- `labels/train`, `labels/val`
- `oral.yaml`
