"""Render each stimulus as a rotating sprite sheet.

A single fixed viewpoint can hide the very thing the task asks about: a
monopitch seen end-on looks flat, and a hipped roof seen from the front
looks gabled. That turns an honest "can you read this roof?" into a
partly arbitrary one, and adds noise that falls on real and generated
buildings alike.

So every stimulus is rendered at N azimuths and tiled into one sprite
sheet. The page animates it by moving the background position, which
costs one HTTP request per building rather than N, and needs no
JavaScript animation library.

Framing differs from make_stimuli.py deliberately. There, a fixed window
preserved relative size, because size is a legitimate realism cue when
asking "is this real?". This study asks about roof *shape*, where size is
not the question and small buildings only waste the frame. Each building
is therefore fitted to its own bounding box with a margin, so all of them
fill the view.

Usage
-----
    PYTHONPATH=. python study/make_spins.py --angles 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model.src.render import CHANNEL_RGB, _label_field

DOCS = Path("study/docs")
KEY = Path("study/answer_key.json")


def fit_cubic(t: np.ndarray, margin: float = 0.16) -> np.ndarray:
    """Crop to the building's own extent, cubic, with a margin.

    Cubic so the aspect ratio is untouched -- an elongated shed must stay
    elongated; only the empty space around it goes.
    """
    fg = t[1:].sum(0) > 0
    if not fg.any():
        return t
    idx = np.argwhere(fg)
    lo, hi = idx.min(0), idx.max(0)
    centre = (lo + hi) / 2.0
    side = int(np.ceil(max(hi - lo + 1) * (1 + 2 * margin)))
    side = max(side, 8)
    out = np.zeros((t.shape[0], side, side, side), dtype=t.dtype)
    src_lo = np.maximum(0, np.round(centre - side / 2).astype(int))
    src_hi = np.minimum(np.array(t.shape[1:]), src_lo + side)
    src_lo = src_hi - (src_hi - src_lo)
    dst_lo = ((side - (src_hi - src_lo)) / 2).astype(int)
    sl_s = tuple(slice(a, b) for a, b in zip(src_lo, src_hi))
    sl_d = tuple(slice(a, a + (b - c)) for a, b, c in
                 zip(dst_lo, src_hi, src_lo))
    out[(slice(None),) + sl_d] = t[(slice(None),) + sl_s]
    return out


def spin_sheet(t: np.ndarray, out_path: Path, angles: int, px: int,
               cols: int, elev: float = 22.0) -> None:
    """Render `angles` azimuths of one building into a tiled sheet."""
    t = fit_cubic(t)
    lab = _label_field(t)                 # shell only; interior stays hidden
    filled = lab > 0
    face = np.zeros((*lab.shape, 4), dtype=np.float32)
    for cls, rgb in CHANNEL_RGB.items():
        if cls:
            face[lab == cls] = (*rgb, 1.0)

    rows = (angles + cols - 1) // cols
    dpi = 100
    fig = plt.figure(figsize=(cols * px / dpi, rows * px / dpi), dpi=dpi)
    fig.patch.set_alpha(0.0)
    D = t.shape[1]
    for i in range(angles):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.voxels(filled, facecolors=face, edgecolor=(0, 0, 0, .12),
                  linewidth=.15)
        ax.set_xlim(0, D); ax.set_ylim(0, D); ax.set_zlim(0, D)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=-60 + i * (360.0 / angles))
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
    fig.subplots_adjust(0, 0, 1, 1, 0, 0)
    fig.savefig(out_path, transparent=True, dpi=dpi,
                pad_inches=0, bbox_inches=None)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=int, default=12)
    ap.add_argument("--px", type=int, default=260, help="pixels per frame")
    ap.add_argument("--cols", type=int, default=4, help="frames per sheet row")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    a = ap.parse_args()

    key = json.loads(KEY.read_text())
    # every stimulus, resolved back to the array it was rendered from
    # Arm label (as written into the answer key) -> the npz it came from.
    # v7 is the configuration the thesis prefers; the study was rebuilt on
    # it so the human evaluation speaks to the model actually defended,
    # not to superseded ones (claude.md §77).
    src = {"v7":    "model/checkpoints/phase_b_v7/eval/eval_samples.npz",
           "v4_w0": "model/checkpoints/phase_b_v4/eval_w0/eval_samples.npz",
           "v5":    "model/checkpoints/phase_b_v5/eval/eval_samples.npz"}
    cache = {k: np.load(v, allow_pickle=True) for k, v in src.items()
             if Path(v).exists()}

    out = DOCS / "spin"
    out.mkdir(parents=True, exist_ok=True)
    items = sorted(key.items())
    if a.limit:
        items = items[:a.limit]

    print(f"{len(items)} stimuli x {a.angles} angles -> {out}")
    for n, (fname, meta) in enumerate(items, 1):
        dst = out / fname
        if dst.exists():
            continue
        z = cache.get(meta["model"])
        if z is None:
            print(f"  skip {fname}: no source for model {meta['model']}")
            continue
        arr = z["gen"] if meta["kind"] == "generated" else z["real"]
        spin_sheet(arr[meta["index"]], dst, a.angles, a.px, a.cols)
        if n % 10 == 0 or n == len(items):
            print(f"  {n}/{len(items)}", flush=True)

    (DOCS / "spin_meta.json").write_text(json.dumps(
        {"angles": a.angles, "cols": a.cols, "px": a.px}))
    print(f"wrote spin_meta.json  (angles={a.angles}, cols={a.cols}, px={a.px})")


if __name__ == "__main__":
    main()
