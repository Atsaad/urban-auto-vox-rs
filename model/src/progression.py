"""
progression.py -- assemble the final thesis "training progression" figure
by stitching per-step grids into one master image.

Layout of the input directory (produced by train.py's _sample_hook):
    <out_dir>/samples/
        step_005000/  samples.npz  grid.png
        step_010000/  samples.npz  grid.png
        ...

This script does one thing: for each fixed CONDITION (row), it builds a row
of buildings across TRAINING STEPS (columns), so the reader sees the same
sample evolving as training progresses.

Layout of the output PNG:
    rows    = N conditions (from DEFAULT_CONDITIONS)
    columns = M chosen training steps (default: log-spaced subset of what exists)
    cells   = one 3D voxel building (rendered isometric)

Usage:
    python -m model.src.progression \
        --samples-dir model/checkpoints/phase_b/samples \
        --seed 42 \
        --out  model/checkpoints/phase_b/progression.png
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .render import render_building


def _discover_steps(samples_dir: Path) -> list[int]:
    """List all step_NNNNNN directories in ascending order."""
    dirs = sorted(glob.glob(str(samples_dir / "step_*")))
    steps = []
    for d in dirs:
        name = os.path.basename(d)
        if name.startswith("step_"):
            try:
                steps.append(int(name[5:]))
            except ValueError:
                continue
    return sorted(steps)


def _load_step(samples_dir: Path, step: int) -> tuple[np.ndarray, list[dict]]:
    """Load (tensors, meta) for one step_NNNNNN directory."""
    npz = np.load(samples_dir / f"step_{step:06d}" / "samples.npz",
                  allow_pickle=True)
    tensors = npz["tensors"]                              # (B, C, D, D, D) uint8
    meta = json.loads(str(npz["meta"]))
    return tensors, meta


def _select_step_subset(all_steps: list[int], n_cols: int) -> list[int]:
    """Pick approximately log-spaced steps for the columns.

    We want the reader to see the early phases (blob), middle (walls) and
    late (sharp building) all represented, not just a linear slice.
    """
    if len(all_steps) <= n_cols:
        return all_steps
    # Log-spaced sampling in step index, so early progress is over-sampled.
    idx = np.geomspace(1, len(all_steps), n_cols).round().astype(int) - 1
    idx = np.unique(np.clip(idx, 0, len(all_steps) - 1))
    return [all_steps[i] for i in idx]


def build_progression(
    samples_dir: str | os.PathLike,
    out_path: str | os.PathLike,
    seed: int = 42,
    n_cols: int = 6,
    cell_size: float = 2.0,
    condition_filter: list[str] | None = None,
    suptitle: str | None = None,
) -> None:
    """Assemble the progression grid PNG."""
    samples_dir = Path(samples_dir)
    steps = _discover_steps(samples_dir)
    if not steps:
        raise FileNotFoundError(f"No step_* directories under {samples_dir}")

    chosen_steps = _select_step_subset(steps, n_cols)
    print(f"[progression] found {len(steps)} steps, using columns {chosen_steps}")

    # Load all chosen steps + isolate the requested seed for each condition.
    all_tensors: dict[str, list[np.ndarray]] = {}     # cond_name -> list of tensors, one per column
    cond_order: list[str] = []
    for step in chosen_steps:
        tensors, meta = _load_step(samples_dir, step)
        # meta is a flat list in the order: K conditions x M seeds
        # We want, for each condition, the entry with `seed == seed_arg`.
        for cond_name in {m["name"] for m in meta}:
            if condition_filter and cond_name not in condition_filter:
                continue
            match_idx = next(
                (i for i, m in enumerate(meta)
                 if m["name"] == cond_name and int(m["seed"]) == int(seed)),
                None,
            )
            if match_idx is None:
                print(f"[progression] WARN: seed {seed} not found for condition "
                      f"{cond_name} at step {step}; skipping cell")
                continue
            all_tensors.setdefault(cond_name, []).append(tensors[match_idx])
            if cond_name not in cond_order:
                cond_order.append(cond_name)

    if not all_tensors:
        raise RuntimeError("No matching (condition, seed) cells found.")

    # Each cell is (top / front / side) — 3 sub-columns per training-step column.
    rows, cols = len(cond_order), len(chosen_steps)
    fig, ax_arr = plt.subplots(
        rows, cols * 3,
        figsize=(cell_size * cols * 3 + 1.2,
                 cell_size * rows + (0.5 if suptitle else 0.2) + 0.5),
        dpi=140,
    )
    ax_arr = np.atleast_2d(ax_arr)

    for r, cond_name in enumerate(cond_order):
        for c, step in enumerate(chosen_steps):
            cell_axes = [ax_arr[r, c * 3 + k] for k in range(3)]
            if c < len(all_tensors[cond_name]):
                render_building(all_tensors[cond_name][c], axes=cell_axes)
            else:
                for a in cell_axes:
                    a.axis("off")
            if r == 0:
                # Put the step label above the FRONT view (centre of the triple).
                cell_axes[1].set_title(f"step {step:,}", fontsize=9, pad=6)
            if c == 0:
                # Row label at the far left: rotated condition name.
                cell_axes[0].set_ylabel(cond_name, fontsize=8, fontweight="bold",
                                        rotation=90, labelpad=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[progression] wrote {out_path}  "
          f"({rows} conditions x {cols} steps, seed={seed})")


# ---- CLI ----------------------------------------------------------------
def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", required=True,
                    help="e.g. model/checkpoints/phase_b/samples")
    ap.add_argument("--out", required=True, help="Output PNG path")
    ap.add_argument("--seed", type=int, default=42,
                    help="Which of the sample seeds to display (default 42)")
    ap.add_argument("--n-cols", type=int, default=6,
                    help="Number of training-step columns (log-spaced)")
    ap.add_argument("--suptitle", default="Training progression -- fixed seed",
                    help="Figure title")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="Optional: subset of condition names to show")
    args = ap.parse_args()

    build_progression(
        samples_dir=args.samples_dir,
        out_path=args.out,
        seed=args.seed,
        n_cols=args.n_cols,
        condition_filter=args.conditions,
        suptitle=args.suptitle,
    )


if __name__ == "__main__":
    _cli()
