"""
render.py -- turn a (C, D, D, D) one-hot voxel tensor into a PNG image.

DESIGN CHOICE (2026-07-02): triple-orthographic projection instead of
`matplotlib.voxels()` isometric rendering.

Why: matplotlib's `voxels()` API draws every face of every occupied cube
individually. At Phase A/B time, an untrained model is essentially random
noise -- ~83 % of the 262 144-cell grid comes out labelled non-empty, i.e.
~218 000 cubes to draw. Attempting this takes ~15 minutes per building on the
RTX 4000 Ada, dominating the training run. Even for a well-trained model with
~2 % occupancy, we still draw ~5 000 cubes, which is slow enough (~2 s) to be
irritating.

Instead we render three 2D orthographic projections:
    - top view    (X-Y plane; project along Z, take max class per column)
    - front view  (X-Z plane; project along Y)
    - side view   (Y-Z plane; project along X)

Each projection colours the pixel by the dominant surface class along the
projected axis. This gives a "footprint / silhouette / cross-section" triple
that's just as readable as a 3D render, ~50x faster (~30 ms per building),
and doesn't degrade as the noise fraction grows.

For polished thesis-quality 3D renders, run `render.py --isometric` after
training on the FINAL checkpoint's samples (a few dozen buildings, occupancy
~2 %, so isometric render is fast enough there).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---- channel -> RGB colour ------------------------------------------------
# Channels: 0=empty, 1=wall, 2=roof, 3=ground, 4=outer_ceiling, 5=closure
# RGB tuples (no alpha in the raster we build).
CHANNEL_RGB = {
    0: (1.00, 1.00, 1.00),   # empty          -> white
    1: (0.28, 0.51, 0.71),   # wall           -> steel blue
    2: (0.86, 0.30, 0.23),   # roof           -> tomato red
    3: (0.36, 0.58, 0.35),   # ground         -> olive green
    4: (0.60, 0.40, 0.80),   # outer_ceiling  -> purple
    5: (0.80, 0.70, 0.20),   # closure        -> ochre
}
CH_NAMES = ["empty", "wall", "roof", "ground", "outer_ceiling", "closure"]


# ---- projection helpers --------------------------------------------------
def _label_field(tensor: np.ndarray) -> np.ndarray:
    """(6, D, D, D) uint8 one-hot -> (D, D, D) int class labels 0..5."""
    if tensor.ndim != 4 or tensor.shape[0] != 6:
        raise ValueError(f"expected (6, D, D, D), got {tensor.shape}")
    return tensor.argmax(axis=0)                        # (D, D, D)


def _project_max_class(
    label: np.ndarray, axis: int, min_thickness: int = 2
) -> np.ndarray:
    """Project 3-D label field along `axis` using majority vote of non-empty voxels.

    For each 2-D pixel: count how many voxels of each non-empty class (1..5)
    lie along the projection column. If the column has at least
    ``min_thickness`` non-empty voxels, paint the pixel with the argmax class;
    otherwise paint empty (white).

    Why majority instead of priority:  the old priority scheme
    (roof > wall > ground > ...) painted a pixel red the moment ANY roof voxel
    appeared in the column. For a well-trained model with a thin roof surface
    that's fine, but for a divergent / noisy model where every column has
    ~10 voxels of each of the 5 non-empty classes, every pixel gets painted
    red -- garbage output looks like a solid red building. Majority voting
    over ALL non-empty classes makes noise render as noisy 2D pixels (obviously
    broken) while a properly reconstructed thin roof still wins its columns.
    """
    counts = np.stack(
        [(label == cls).sum(axis=axis) for cls in range(6)],
        axis=0,
    ).astype(np.int32)                                  # (6, H, W)
    nonempty_total = counts[1:].sum(axis=0)             # (H, W)
    argmax_nonempty = counts[1:].argmax(axis=0) + 1     # (H, W), class in 1..5
    return np.where(
        nonempty_total >= min_thickness, argmax_nonempty, 0
    ).astype(np.uint8)


def _labels_to_rgb(label_2d: np.ndarray) -> np.ndarray:
    """(H, W) int labels -> (H, W, 3) float RGB."""
    rgb = np.ones((*label_2d.shape, 3), dtype=np.float32)
    for cls, colour in CHANNEL_RGB.items():
        rgb[label_2d == cls] = colour
    return rgb


# ---- single-building renderer -------------------------------------------
def render_building(
    tensor: np.ndarray,
    out_path: str | os.PathLike | None = None,
    title: str | None = None,
    fig_size_inches: float = 3.0,
    axes: Sequence[plt.Axes] | None = None,
) -> plt.Figure | None:
    """Draw the three orthographic projections of a voxel building.

    Layout: 1 row, 3 columns -- [top view] [front view] [side view].
    If `axes` is a sequence of 3 axes, draws into them and returns None
    (used by render_grid). Otherwise creates a new figure.
    """
    if not isinstance(tensor, np.ndarray):
        tensor = np.asarray(tensor)
    if tensor.dtype != np.uint8:
        tensor = tensor.astype(np.uint8)

    label = _label_field(tensor)
    # Axis convention (build_tensors.py §18.2): dim0 = X (width),
    # dim1 = Y (depth), dim2 = Z (height, up).
    top_view   = _project_max_class(label, axis=2)      # look down Z
    front_view = _project_max_class(label, axis=1)      # look along Y (from front)
    side_view  = _project_max_class(label, axis=0)      # look along X (from side)

    # Flip Z-vertical projections so that "up" is up on the page.
    front_view = np.flipud(front_view.T)
    side_view  = np.flipud(side_view.T)

    standalone = axes is None
    if standalone:
        fig, ax_arr = plt.subplots(1, 3, figsize=(fig_size_inches * 3, fig_size_inches),
                                   dpi=100)
        axes = ax_arr

    for ax, view, name in zip(
        axes,
        (top_view, front_view, side_view),
        ("top (X-Y)", "front (X-Z)", "side (Y-Z)"),
    ):
        ax.imshow(_labels_to_rgb(view), interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=7, pad=2)

    if title and standalone:
        axes[1].set_xlabel(title, fontsize=8)          # centre label

    if standalone:
        if title:
            fig.suptitle(title, fontsize=8, y=1.02)
        fig.tight_layout(pad=0.4)
        if out_path:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
            return None
        return fig
    return None


# ---- grid renderer ------------------------------------------------------
def render_grid(
    tensors: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    out_path: str | os.PathLike | None = None,
    cols: int = 4,
    cell_size: float = 1.6,
    suptitle: str | None = None,
) -> plt.Figure:
    """Render N buildings into a grid of triple-projections.

    Each building occupies one row of 3 columns (top / front / side).
    `cols` sets how many *buildings* per grid row (not sub-columns).
    """
    n = len(tensors)
    rows = math.ceil(n / max(1, cols))
    # We show (top/front/side) for each cell, so total ncols = 3 * cols.
    fig, ax_arr = plt.subplots(
        rows, cols * 3,
        figsize=(cell_size * cols * 3 + 0.5,
                 cell_size * rows + (0.4 if suptitle else 0) + 0.3),
        dpi=120,
    )
    ax_arr = np.atleast_2d(ax_arr)

    for i, t in enumerate(tensors):
        r, c = divmod(i, cols)
        cell_axes = [ax_arr[r, c * 3 + k] for k in range(3)]
        render_building(t, axes=cell_axes)
        # Put the building title above the FRONT view (middle projection).
        if titles:
            cell_axes[1].set_xlabel(titles[i], fontsize=7)

    # Blank out unused cells.
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        for k in range(3):
            ax_arr[r, c * 3 + k].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(pad=0.3)
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    return fig


# ---- optional: polished isometric renderer for thesis figures -----------
def render_isometric(
    tensor: np.ndarray,
    out_path: str | os.PathLike | None = None,
    title: str | None = None,
    fig_size_inches: float = 3.0,
    elev: float = 20.0,
    azim: float = -60.0,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Slow but pretty 3D voxel render. Only use for a handful of FINAL samples.

    Aborts with a warning if occupancy > 10 % (would take >minute per building).
    """
    if not isinstance(tensor, np.ndarray):
        tensor = np.asarray(tensor)
    label = _label_field(tensor)
    filled = label > 0
    occ_frac = filled.mean()
    if occ_frac > 0.10:
        print(f"[render_isometric] WARN: occupancy {100*occ_frac:.0f}% > 10 %; "
              "would be very slow. Falling back to triple-projection.")
        return render_building(tensor, out_path=out_path, title=title)

    facecolors = np.zeros((*label.shape, 4), dtype=np.float32)
    for cls, rgb in CHANNEL_RGB.items():
        if cls == 0:
            continue
        facecolors[label == cls] = (*rgb, 1.0)

    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=(fig_size_inches, fig_size_inches), dpi=140)
        ax = fig.add_subplot(111, projection="3d")

    ax.voxels(filled, facecolors=facecolors, edgecolor=(0, 0, 0, 0.15), linewidth=0.2)
    D = tensor.shape[1]
    ax.set_xlim(0, D); ax.set_ylim(0, D); ax.set_zlim(0, D)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=8, pad=2)

    if standalone:
        fig.tight_layout(pad=0.2)
        if out_path:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            return None
        return fig
    return None


# ---- CLI ----------------------------------------------------------------
def _cli() -> None:
    import argparse
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True,
                    help=".pt file (from sample.py CLI) OR .npz file "
                         "(from train.py progression hook)")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--cols", type=int, default=4,
                    help="buildings per grid row")
    ap.add_argument("--suptitle", default=None)
    ap.add_argument("--isometric", action="store_true",
                    help="Use slow polished 3D render (only for well-trained "
                         "final samples with low occupancy)")
    args = ap.parse_args()

    # Accept both .pt (sample.py CLI) and .npz (train.py progression hook)
    if args.samples.endswith(".npz"):
        import json as _json
        blob = np.load(args.samples, allow_pickle=True)
        tensors = np.asarray(blob["tensors"])
        meta = _json.loads(str(blob["meta"]))
    else:
        blob = torch.load(args.samples, map_location="cpu", weights_only=False)
        tensors = blob["tensors"].numpy()
        meta = blob["meta"]
    titles = [f"{m.get('name','?')}\nseed={m.get('seed','?')}" for m in meta]

    if args.isometric:
        # Legacy path: one isometric plot per building, concatenated as a grid.
        n = len(tensors)
        cols = args.cols
        rows = math.ceil(n / cols)
        fig = plt.figure(figsize=(cols * 2.4, rows * 2.4), dpi=140)
        for i, t in enumerate(tensors):
            ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
            render_isometric(t, ax=ax, title=titles[i])
        if args.suptitle:
            fig.suptitle(args.suptitle, fontsize=10)
        fig.tight_layout(pad=0.4)
        fig.savefig(args.out, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    else:
        render_grid(list(tensors), titles=titles, out_path=args.out,
                    cols=args.cols, suptitle=args.suptitle)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    _cli()
