"""Regenerate evaluation figures from cached plot data.

The figure legends were hardcoded to "v4 generated", so every v5 and
Phase C figure carried the wrong model name. Fixing the label does not
require recomputing any metric: each `figures_*` function reads a
`plotdata_*.npz` that already holds the numbers. This re-renders them
with the correct name at a cost of seconds rather than a GPU hour.

Usage:
    python -m model.src.refigure model/checkpoints/phase_b_v5/eval ...
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import evaluate as E


def regenerate(eval_dir: Path) -> None:
    cache = eval_dir / "eval_samples.npz"
    if cache.exists():
        E.load_cache(cache)          # sets MODEL_LABEL from the checkpoint path
    print(f"[refigure] {eval_dir}  ->  label '{E.MODEL_LABEL}'")

    jobs = [("plotdata_pillarA.npz",  E.figures_pillarA),
            ("plotdata_pillarBC.npz", E.figures_pillarBC),
            ("plotdata_pillarEF.npz", E.figures_pillarEF),
            ("plotdata_pillarD.npz",  E.figures_pillarD),
            ("plotdata_cfgsweep.npz", E.figures_cfgsweep),
            ("plotdata_ablation.npz", E.figures_ablation)]
    for name, fn in jobs:
        p = eval_dir / name
        if not p.exists():
            continue
        try:
            fn(p, eval_dir)
        except Exception as exc:                     # keep going on the rest
            print(f"[refigure]   {name}: FAILED ({exc})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for d in sys.argv[1:]:
        regenerate(Path(d))
