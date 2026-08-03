"""Generate the per-version figures Chapter 6 references by name.

`plot_training.py` draws all versions on one axis, which is the right
figure for a comparison but not what the chapter cites: it refers to
v1_loss.png, v2_gradnorm.png and so on, one per run, each with its own
caption. This produces exactly those filenames.

Data comes from figures/data/*.csv, extracted from the training logs by
dump_figure_data.py — so v1--v3 are recoverable even though those runs
predate the losses.json checkpointing added later.

Colours follow the captions already written in Chapter 6: v1 red, v2
orange, v3 purple, v4 green.

Usage
-----
    python -m model.src.dump_figure_data          # once, writes the CSVs
    python -m model.src.thesis_figures --out-dir <thesis>/figures
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = Path("figures/data")
CKPT = Path("model/checkpoints")

COLOUR = {"v1": "#d03b3b", "v2": "#eb6834", "v3": "#7a5bd6", "v4": "#0ca30c"}
GRID = dict(alpha=.25, linewidth=.6)


def _read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _style(ax, xlabel, ylabel, title=None, log=True):
    if log:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(True, **GRID)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, out: Path):
    fig.tight_layout(pad=.4)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}  (+ .pdf)")


def loss_figure(v: str, out: Path, annotate=None) -> None:
    p = DATA / f"losses_{v}.csv"
    if not p.exists():
        print(f"  skip {v} loss: {p} missing")
        return
    rows = _read(p)
    it = [int(r["iter"]) for r in rows]
    ls = [float(r["loss"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(it, ls, color=COLOUR[v], lw=1.4)
    if annotate:
        for x, label in annotate:
            if x <= max(it):
                ax.axvline(x, color="grey", ls=":", lw=1)
                ax.annotate(label, xy=(x, max(ls)), fontsize=7.5, color="grey",
                            rotation=90, va="top", ha="right")
    _style(ax, "training iteration", "loss (log scale)")
    _save(fig, out)


def gradnorm_figure(v: str, out: Path) -> None:
    p = DATA / f"gradnorms_{v}.csv"
    if not p.exists():
        print(f"  skip {v} gradnorm: {p} missing "
              f"(v1 predates grad-norm logging)")
        return
    rows = _read(p)
    it = [int(r["iter"]) for r in rows]
    avg = [float(r["gn_avg_100"]) for r in rows]
    mx = [float(r["gn_max_100"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(it, mx, color=COLOUR[v], lw=.9, alpha=.45, label="max over 100 iters")
    ax.plot(it, avg, color=COLOUR[v], lw=1.5, label="mean over 100 iters")
    ax.legend(fontsize=8, frameon=False)
    _style(ax, "training iteration", "gradient norm (log scale)")
    _save(fig, out)


def phase_a_figure(out: Path) -> None:
    p = CKPT / "phase_a_smoke" / "losses.json"
    if not p.exists():
        print(f"  skip phase A: {p} missing")
        return
    h = json.loads(p.read_text())
    ls = [x[1] if isinstance(x, (list, tuple)) else x for x in h]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(range(1, len(ls) + 1), ls, color="#2a78d6", lw=1.5)
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.annotate("unit-variance baseline (zero-init head)", xy=(len(ls) * .45, 1.0),
                fontsize=7.5, color="grey", va="bottom")
    _style(ax, "training iteration", "$\\varepsilon$-MSE loss", log=False)
    _save(fig, out)


def class_distribution_figure(out: Path) -> None:
    p = DATA / "class_frequencies.csv"
    if not p.exists():
        print(f"  skip class distribution: {p} missing")
        return
    rows = _read(p)
    versions, names = [], []
    for r in rows:
        if r["version"] not in versions:
            versions.append(r["version"])
        if r["class_name"] not in names:
            names.append(r["class_name"])
    vals = {v: {r["class_name"]: float(r["frequency"])
                for r in rows if r["version"] == v} for v in versions}

    fig, axes = plt.subplots(1, len(versions), figsize=(2.6 * len(versions), 3.2),
                             sharey=True)
    if len(versions) == 1:
        axes = [axes]
    cols = {"real": "#2a78d6", "v2": "#eb6834", "v3": "#7a5bd6", "v4": "#0ca30c"}
    for ax, v in zip(axes, versions):
        y = [100 * vals[v].get(n, 0) for n in names]
        ax.bar(range(len(names)), y, color=cols.get(v, "#888"), width=.68)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
        ax.set_yscale("log")
        ax.set_title(v, fontsize=10)
        ax.grid(True, axis="y", **GRID)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("share of voxels (%, log)", fontsize=9)
    _save(fig, out)


def sample_grid(ckpt_dir: str, out: Path, isometric=False, step=None) -> None:
    """Render the newest progression sample of a run as a grid."""
    from .render import render_grid, render_isometric
    d = CKPT / ckpt_dir / "samples"
    steps = sorted(d.glob("step_*")) if d.exists() else []
    if not steps:
        print(f"  skip {out.name}: no samples under {d}")
        return
    chosen = steps[-1] if step is None else d / f"step_{step:06d}"
    npz = chosen / "samples.npz"
    if not npz.exists():
        print(f"  skip {out.name}: {npz} missing")
        return
    z = np.load(npz, allow_pickle=True)
    key = "samples" if "samples" in z.files else z.files[0]
    arr = z[key]
    meta = json.loads(str(z["meta"])) if "meta" in z.files else None
    titles = ([f"{m.get('name','')}\nseed={m.get('seed','')}" for m in meta]
              if meta else None)
    sup = f"{ckpt_dir} @ {chosen.name.replace('step_', 'step ')}"
    if isometric:
        # render_grid draws triple-projections; the isometric view has no
        # grid form of its own, so lay the per-building 3D axes out by hand
        # exactly as render.py's CLI does.
        n = len(arr)
        cols = 4
        rows = (n + cols - 1) // cols
        fig = plt.figure(figsize=(cols * 2.2, rows * 2.2), dpi=200)
        for i, t in enumerate(arr):
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
            render_isometric(t, ax=ax, title=titles[i] if titles else None)
        fig.suptitle(sup, fontsize=10)
        fig.tight_layout(pad=.3)
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
    else:
        render_grid(list(arr), titles=titles, cols=4, out_path=out,
                    suptitle=sup)
    print(f"  wrote {out.name}  (from {chosen.name})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("loss curves:")
    phase_a_figure(out / "phase_a_loss.png")
    # v1 diverged at ~9,500 and stayed pinned at 1.0 (claude.md, Ch 6 §exp-v1)
    loss_figure("v1", out / "v1_loss.png",
                annotate=[(9500, "diverged")])
    loss_figure("v2", out / "v2_loss.png")
    loss_figure("v3", out / "v3_loss.png")
    loss_figure("v4", out / "v4_loss.png")

    print("gradient norms:")
    gradnorm_figure("v2", out / "v2_gradnorm.png")
    gradnorm_figure("v3", out / "v3_gradnorm.png")
    gradnorm_figure("v4", out / "v4_gradnorm.png")

    print("class distribution:")
    class_distribution_figure(out / "class_distribution.png")

    print("sample grids:")
    sample_grid("phase_b",    out / "v1_samples.png")
    sample_grid("phase_b_v2", out / "v2_samples.png")
    sample_grid("phase_b_v3", out / "v3_samples.png")
    sample_grid("phase_b_v4", out / "v4_samples_projections.png")
    sample_grid("phase_b_v4", out / "v4_samples_isometric.png", isometric=True)


if __name__ == "__main__":
    main()
