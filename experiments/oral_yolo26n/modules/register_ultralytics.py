"""Register custom fusion blocks into Ultralytics parse_model."""

from __future__ import annotations

import copy

from experiments.oral_yolo26n.modules.fusion_blocks import GqROFConcat, PlainGateConcat


def register() -> None:
    """Public entry used by train scripts."""
    import ultralytics.nn.tasks as tasks
    from ultralytics.utils.torch_utils import make_divisible

    tasks.GqROFConcat = GqROFConcat
    tasks.PlainGateConcat = PlainGateConcat

    if getattr(tasks, "_gqrof_parse_wrapped", False):
        return

    orig = tasks.parse_model

    def wrapped(d, ch, verbose=True):  # type: ignore[no-untyped-def]
        dd = copy.deepcopy(d)

        scale = dd.get("scale")
        depth, width, max_channels = 0.50, 0.25, 1024
        if dd.get("scales") and scale in dd["scales"]:
            depth, width, max_channels = dd["scales"][scale]

        ch0 = ch if isinstance(ch, list) else [ch]
        ch_track = [ch0[-1]]
        bb = dd["backbone"]
        hd = dd["head"]
        all_layers = bb + hd

        for i, (f, n, m, args) in enumerate(all_layers):
            m_name = m if isinstance(m, str) else getattr(m, "__name__", str(m))
            if m_name in {"GqROFConcat", "PlainGateConcat"}:
                c_in = ch_track[f[0]] if isinstance(f, list) else ch_track[f]
                all_layers[i] = [f, n, m_name, [c_in]]
                c2 = c_in * 2
            elif m_name == "Concat":
                c2 = sum(ch_track[x] for x in f)
            elif m_name in {"Detect"}:
                c2 = ch_track[-1]
            elif m_name.endswith("Upsample") or m_name == "nn.Upsample":
                c2 = ch_track[-1]
            elif args and isinstance(args[0], int):
                c2 = make_divisible(min(args[0], max_channels) * width, 8)
            else:
                c2 = ch_track[-1]
            ch_track.append(c2)

        dd["backbone"] = all_layers[: len(bb)]
        dd["head"] = all_layers[len(bb) :]

        g = orig.__globals__
        g["GqROFConcat"] = GqROFConcat
        g["PlainGateConcat"] = PlainGateConcat
        if "base_modules" in g:
            g["base_modules"] = frozenset(set(g["base_modules"]) | {GqROFConcat, PlainGateConcat})
        return orig(dd, ch, verbose)

    tasks.parse_model = wrapped  # type: ignore[assignment]
    tasks._gqrof_parse_wrapped = True
