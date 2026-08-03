"""Re-frame the rendered sprite sheets so the building fills the frame.

`make_spins.py` renders each azimuth into a matplotlib 3D axes. Those
axes reserve generous margins, and an isometric projection of a cube
inside a square viewport wastes the corners on top of that. Measured on
the delivered sheets, the building occupies only about 44% x 41% of its
frame. With the page capping the sprite at 400 px inside a ~720 px card,
the building ended up at roughly a fifth of the width of the box it sits
in -- far short of the "fills most of the box" the study needs, and small
enough that roof form is genuinely harder to read than it should be.

This crops that dead space out of the existing sheets. Re-rendering would
cost about two hours and change nothing else.

The one subtlety is that the crop must be computed **once per sheet, over
the union of all its frames** -- never per frame. Each azimuth has a
different silhouette, so per-frame cropping would make the building pulse
in size as it rotates, which is both ugly and a false cue: apparent size
would then encode viewing angle rather than the building.

Scaling is uniform for the same reason `fit_cubic` is cubic: an elongated
shed must stay elongated.

Run once, on freshly rendered sheets. Running it twice is close to a
no-op -- the second pass finds a bbox that already fills the frame -- but
costs a second resampling, so it refuses unless --force is given.

Usage
-----
    PYTHONPATH=. python study/tighten_spins.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

DOCS = Path("study/docs")
SPIN = DOCS / "spin"
STAMP = SPIN / ".tightened"

MARGIN = 0.03          # keep a hair of space so edges do not touch the box
ALPHA = 8              # alpha above this counts as building


def frames(sheet: np.ndarray, cols: int, rows: int):
    """Yield (row, col, slice) for each tile of the sheet."""
    fh, fw = sheet.shape[0] // rows, sheet.shape[1] // cols
    for r in range(rows):
        for c in range(cols):
            yield r, c, (slice(r * fh, (r + 1) * fh),
                         slice(c * fw, (c + 1) * fw))


def tighten(path: Path, cols: int, rows: int) -> tuple[float, float]:
    im = np.array(Image.open(path).convert("RGBA"))
    fh, fw = im.shape[0] // rows, im.shape[1] // cols

    # union bounding box over every frame, in frame-local coordinates
    x0, y0, x1, y1 = fw, fh, 0, 0
    for _, _, sl in frames(im, cols, rows):
        a = im[sl][..., 3] > ALPHA
        if not a.any():
            continue
        ys, xs = np.where(a)
        x0, x1 = min(x0, xs.min()), max(x1, xs.max())
        y0, y1 = min(y0, ys.min()), max(y1, ys.max())
    if x1 <= x0 or y1 <= y0:
        return 0.0, 0.0

    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    s = min(fw * (1 - 2 * MARGIN) / bw, fh * (1 - 2 * MARGIN) / bh)
    nw, nh = max(1, round(bw * s)), max(1, round(bh * s))

    out = np.zeros_like(im)
    for r, c, sl in frames(im, cols, rows):
        crop = im[sl][y0:y1 + 1, x0:x1 + 1]
        # premultiply before resizing: resizing straight RGBA drags the
        # colour of fully transparent pixels into the edges
        f = crop.astype(np.float32)
        al = f[..., 3:4] / 255.0
        pre = np.dstack([f[..., :3] * al, f[..., 3:4]]).astype(np.uint8)
        rs = np.array(Image.fromarray(pre).resize((nw, nh), Image.LANCZOS)
                      ).astype(np.float32)
        a2 = np.clip(rs[..., 3:4], 0, 255) / 255.0
        rgb = np.where(a2 > 0.004, rs[..., :3] / np.maximum(a2, 1e-6), 0.0)
        tile = np.dstack([np.clip(rgb, 0, 255),
                          np.clip(rs[..., 3:4], 0, 255)]).astype(np.uint8)

        oy, ox = r * fh + (fh - nh) // 2, c * fw + (fw - nw) // 2
        out[oy:oy + nh, ox:ox + nw] = tile

    Image.fromarray(out).save(path)
    return bw / fw, bh / fh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-tighten sheets that were already tightened")
    a = ap.parse_args()

    if STAMP.exists() and not a.force:
        raise SystemExit(
            f"[spins] already tightened ({STAMP}).\n"
            f"        Re-running costs a second resampling for no gain.\n"
            f"        Re-render with make_spins.py first, or pass --force.")

    meta = json.loads((DOCS / "spin_meta.json").read_text())
    cols = meta["cols"]
    rows = -(-meta["angles"] // cols)

    sheets = sorted(SPIN.glob("*.png"))
    print(f"{len(sheets)} sheets, {cols}x{rows} frames each")
    before = []
    for n, p in enumerate(sheets, 1):
        w, h = tighten(p, cols, rows)
        if w:
            before.append((w, h))
        if n % 32 == 0 or n == len(sheets):
            print(f"  {n}/{len(sheets)}", flush=True)

    b = np.array(before)
    STAMP.write_text("tightened by study/tighten_spins.py\n")
    print(f"building filled {b[:,0].mean():.0%} x {b[:,1].mean():.0%} "
          f"of the frame; now {1-2*MARGIN:.0%} on its larger axis")


if __name__ == "__main__":
    main()
