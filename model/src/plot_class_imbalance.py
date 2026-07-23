"""
Figure 7: class-imbalance illustration for the intermediate presentation.

Shows two things side by side:
  (A) The real training-data class distribution -- 99.4 % empty, everything
      else in the low-hundredths-of-a-percent range.
  (B) What each of v2 / v3 / v4 actually generates -- do the produced samples
      match the real distribution?

Reading the resulting figure:
  - Real       : 99.4 % empty  → the target class distribution.
  - v2 / v3    : ~15 % per class  → uniform noise, model never learned the
                                    sparsity prior.
  - v4         : ~99 % empty  → matches training marginal.

The point of the slide: unweighted MSE (v1/v2) and fg-weighted MSE (v3) both
optimise on continuous ε; neither imposes the categorical "one class per
voxel" constraint. Cross-entropy on argmax (v4) does, and softmax's
normalisation gives the model a free bias toward the training marginal.
See understanding_guide §11.11 for the derivation.

Usage:
    python -m model.src.plot_class_imbalance --out figures/class_imbalance.png
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .dataset import Building3DDataset


CLASS_NAMES = ["empty", "wall", "roof", "ground", "outer_ceiling", "closure"]
CLASS_COLORS = [
    (0.85, 0.85, 0.85),   # empty light grey (visible on white bg)
    (0.28, 0.51, 0.71),   # wall blue
    (0.86, 0.30, 0.23),   # roof red
    (0.36, 0.58, 0.35),   # ground green
    (0.60, 0.40, 0.80),   # outer_ceiling purple
    (0.80, 0.70, 0.20),   # closure ochre
]


def _real_class_frequencies(
    cfg_path: str = "model/configs/phase_b.yaml",
    n_samples: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Sample n_samples training buildings; return normalised class fractions."""
    cfg = yaml.safe_load(open(cfg_path))
    ds = Building3DDataset(
        shards_dir=cfg["shards_dir"],
        manifest_path=cfg["manifest_path"],
    )
    rng = random.Random(seed)
    idxs = rng.sample(range(len(ds)), min(n_samples, len(ds)))
    totals = np.zeros(6, dtype=np.float64)
    for i in idxs:
        x = ds[i]["tensor"].numpy()
        lab = x.argmax(axis=0)
        for c in range(6):
            totals[c] += (lab == c).sum()
    return totals / totals.sum()


def _sample_class_frequencies(samples_dir: Path) -> np.ndarray | None:
    """Return normalised class fractions from the latest step_*/samples.npz."""
    steps = sorted(int(p.name[5:]) for p in samples_dir.glob("step_*")
                   if p.is_dir() and p.name[5:].isdigit())
    if not steps:
        return None
    latest = samples_dir / f"step_{steps[-1]:06d}" / "samples.npz"
    if not latest.exists():
        return None
    npz = np.load(latest, allow_pickle=True)
    tensors = npz["tensors"]                                  # (N, C, D, D, D)
    labs = tensors.argmax(axis=1)                             # (N, D, D, D)
    totals = np.zeros(6, dtype=np.float64)
    for c in range(6):
        totals[c] = (labs == c).sum()
    return totals / totals.sum()


def _draw_bars(ax, freqs: np.ndarray, title: str, subtitle: str) -> None:
    y = np.arange(len(CLASS_NAMES))
    ax.barh(y, freqs, color=CLASS_COLORS, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25 %", "50 %", "75 %", "100 %"], fontsize=8)
    ax.set_title(title, fontsize=9.5, loc="left", pad=6)
    ax.text(0.99, -0.08, subtitle, transform=ax.transAxes, ha="right",
            va="top", fontsize=7.5, color="grey", style="italic")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    for yi, v in enumerate(freqs):
        label = f"{100*v:.2f} %" if v < 0.01 else f"{100*v:.1f} %"
        ax.text(min(v + 0.015, 0.94), yi, label, va="center", fontsize=7)


def build_figure(out_path: str,
                 cfg_path: str = "model/configs/phase_b.yaml",
                 n_samples: int = 200) -> None:
    print(f"[plot_class_imbalance] sampling {n_samples} real buildings "
          "for class frequencies...")
    real_freq = _real_class_frequencies(cfg_path, n_samples)
    print("  REAL training data:")
    for n, f in zip(CLASS_NAMES, real_freq):
        print(f"    {n:16s}  {100*f:6.3f} %")

    # Load each Phase-B version's latest sample distribution (if available).
    ckpt_root = Path("model/checkpoints")
    versions = [
        ("v2  ε-MSE, unweighted",              "phase_b_v2"),
        ("v3  ε-MSE, fg-weighted (W = 100)",   "phase_b_v3"),
        ("v4  x0 + cross-entropy  (working)",  "phase_b_v4"),
    ]
    version_data = []
    for label, name in versions:
        freq = _sample_class_frequencies(ckpt_root / name / "samples")
        version_data.append((label, freq))
        print(f"  {label}:")
        if freq is None:
            print("    (no samples on disk)")
        else:
            for n, f in zip(CLASS_NAMES, freq):
                print(f"    {n:16s}  {100*f:6.3f} %")

    # 2x2 grid.  Real (top-left), v2 (top-right), v3 (bottom-left), v4 (bottom-right).
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), dpi=140,
                             sharex=True)
    _draw_bars(
        axes[0, 0], real_freq,
        "Real training-data distribution",
        f"n = {n_samples} random training buildings  ·  D = 64",
    )
    for (label, freq), ax in zip(version_data, [axes[0, 1], axes[1, 0], axes[1, 1]]):
        if freq is None:
            ax.text(0.5, 0.5, "(no samples on disk)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="grey")
            ax.set_title(label, fontsize=9.5, loc="left")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            subtitle = ("→ matches training marginal" if "v4" in label
                        else "→ ~uniform: model didn't learn sparsity prior")
            _draw_bars(ax, freq, label, subtitle)

    fig.suptitle(
        "Class distribution — real data vs. what each Phase B version generates",
        fontsize=12, y=0.995)
    fig.text(0.5, 0.005,
             "v2 / v3 samples are near-uniform (~16.7 % per class = "
             "83 % occupied noise). v4 matches the training marginal (~99 % empty).",
             ha="center", fontsize=8, color="grey", style="italic")

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[plot_class_imbalance] wrote {out}")


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/class_imbalance.png")
    ap.add_argument("--n-samples", type=int, default=200,
                    help="number of training buildings sampled for the real dist")
    ap.add_argument("--config", default="model/configs/phase_b.yaml")
    args = ap.parse_args()
    build_figure(args.out, cfg_path=args.config, n_samples=args.n_samples)


if __name__ == "__main__":
    _cli()
