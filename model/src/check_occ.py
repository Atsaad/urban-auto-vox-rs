"""Quick occupancy check for progression samples.

Usage:
    python -m model.src.check_occ                       # latest step of phase_b_v3
    python -m model.src.check_occ 5000                  # specific step
    python -m model.src.check_occ 5000 phase_b_v2       # specific run
    python -m model.src.check_occ latest phase_b_v3     # latest step of a run
    python -m model.src.check_occ all phase_b_v3        # all steps, table

Real-building reference (mean of 7 sampled training buildings): 0.33 %.
Uniform noise reference: 83.3 %.
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


CH_NAMES = ["empty", "wall", "roof", "ground", "outer_ceiling", "closure"]


def _steps(run_dir: Path) -> list[int]:
    return sorted(
        int(Path(p).name[5:])
        for p in glob.glob(str(run_dir / "samples" / "step_*"))
        if Path(p).name.startswith("step_") and Path(p).name[5:].isdigit()
    )


def _report_one(run_dir: Path, step: int) -> None:
    npz_path = run_dir / "samples" / f"step_{step:06d}" / "samples.npz"
    npz = np.load(npz_path, allow_pickle=True)
    tensors = npz["tensors"]                              # (K*M, 6, D, D, D)
    meta = json.loads(str(npz["meta"]))
    lab = tensors.argmax(axis=1)                          # (N, D, D, D)

    by_cond = defaultdict(list)
    for i, m in enumerate(meta):
        occ = 100 * (lab[i] != 0).mean()
        by_cond[m["name"]].append((occ, lab[i]))

    print(f"\n=== {run_dir.name} · step {step:>6,} ===")
    for cond, entries in by_cond.items():
        occs = [e[0] for e in entries]
        # Per-class breakdown, averaged across seeds for this condition
        cls_pct = np.mean(
            [[100 * (e[1] == c).mean() for c in range(6)] for e in entries],
            axis=0,
        )
        print(
            f"  {cond[:34]:34s}  occ={np.mean(occs):5.2f}%  "
            f"(seeds min={min(occs):5.2f} max={max(occs):5.2f})  "
            f"|  wall={cls_pct[1]:4.2f}%  roof={cls_pct[2]:4.2f}%  "
            f"ground={cls_pct[3]:4.2f}%"
        )


def _report_all_table(run_dir: Path) -> None:
    steps = _steps(run_dir)
    if not steps:
        print(f"No step_* directories under {run_dir/'samples'}")
        return
    # Header row uses first step's conditions
    first_meta = json.loads(str(np.load(
        run_dir / "samples" / f"step_{steps[0]:06d}" / "samples.npz",
        allow_pickle=True)["meta"]))
    cond_order = []
    for m in first_meta:
        if m["name"] not in cond_order:
            cond_order.append(m["name"])

    header = f"{'step':>7} | " + " | ".join(f"{c[:22]:>22s}" for c in cond_order)
    print("\n" + header)
    print("-" * len(header))

    for step in steps:
        npz = np.load(run_dir / "samples" / f"step_{step:06d}" / "samples.npz",
                      allow_pickle=True)
        tensors = npz["tensors"]
        meta = json.loads(str(npz["meta"]))
        lab = tensors.argmax(axis=1)
        by_cond = defaultdict(list)
        for i, m in enumerate(meta):
            by_cond[m["name"]].append(100 * (lab[i] != 0).mean())
        row = [f"{step:>7,}"] + [
            f"{np.mean(by_cond[c]):>21.2f}%" for c in cond_order
        ]
        print(" | ".join(row))

    print("\nReal-building reference: 0.33 %.   Uniform-noise reference: 83.3 %.")


def main() -> None:
    step_arg = sys.argv[1] if len(sys.argv) > 1 else "latest"
    run = sys.argv[2] if len(sys.argv) > 2 else "phase_b_v3"
    run_dir = Path("model/checkpoints") / run

    if not run_dir.exists():
        print(f"Run dir does not exist: {run_dir}")
        sys.exit(1)

    if step_arg == "all":
        _report_all_table(run_dir)
        return

    if step_arg == "latest":
        steps = _steps(run_dir)
        if not steps:
            print(f"No sampled steps yet under {run_dir/'samples'}")
            sys.exit(1)
        step = steps[-1]
    else:
        step = int(step_arg)

    _report_one(run_dir, step)


if __name__ == "__main__":
    main()
