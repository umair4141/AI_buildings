# YOLO26n oral experiments (Exp-1 vs Exp-2 only)

No Exp-0 baseline.

| RUN | Name | What |
|---|---|---|
| **1** | `exp1_qrof_gate` | Gaussian q-rung gate (learnable \(q\)) |
| **2** | `exp2_plain_gate` | Plain feature gate (no \(q\)) |

## Kaggle one cell

Paste `KAGGLE_ONE_CELL.py`, then:

```python
RUN = 1   # Exp-1
# then
RUN = 2   # Exp-2
```

Settings: from scratch, `imgsz=1280`, `batch=16`, `patience=0`.
