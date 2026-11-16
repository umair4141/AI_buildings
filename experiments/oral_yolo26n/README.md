# YOLO26n oral experiments (fixed)

## Runs (one by one)

| RUN | Name | What |
|---|---|---|
| **0** | `exp0_yolo26n_baseline` | Official YOLO26n `Concat` |
| **1** | `exp1_qrof_gate` | Cross-stream **Gaussian q-rung** gate (learnable \(q\)) |
| **2** | `exp2_plain_gate` | Matched **plain** cross-stream gate (`2*sigmoid`, no \(q\)) |

## Kaggle one cell

Paste `KAGGLE_ONE_CELL.py`, then:

```python
RUN = 0   # then 1, then 2
```

Settings: `imgsz=1280`, `batch=16`, `optimizer="auto"`, `patience=0`, **from scratch** (no `yolo26n.pt`).

## Fixes included

- Skip unknown `category_id` (no `KeyError: 0`)
- Build official YAML with `Concat`, then **replace** Concat modules (no broken `parse_model` / `base_modules` hack)
- Unequal merge channels via lazy init; output is `cat` like Concat
- Cross-stream aggregation (not per-stream-only gate)
- Matched gate range `[0, 2]` for Exp-1 and Exp-2
- Val→test holdout for final score

## Dataset path

Default in one-cell:

`/kaggle/input/datasets/xxccfffffffffff/oral-dataset/oral dataset`

If your slug is different, change `KAGGLE_ROOT`.
