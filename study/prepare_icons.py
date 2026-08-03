"""Normalise the hand-made roof icons into a uniform set for the study.

The source renders under `local comments/study icons/` are the right
drawings -- they are voxel buildings in the same colours as the stimuli,
one per AdV Dachform -- but they are not yet a usable icon set:

  * they are opaque white, and the page has a dark mode;
  * they range from 332x299 to 353x968 px;
  * the building sits at a different scale in each frame, so side by side
    in the option grid they would read as arbitrarily different sizes.

This script fixes those three things and nothing else. The drawings are
untouched.

  1. White -> transparent. A threshold mask, not a per-pixel unmatting:
     the content is saturated red/blue/green and never near-white, so a
     threshold separates it cleanly. Anti-aliased edges are recovered in
     step 3 rather than guessed at here.
  2. Crop to the drawing's own bounding box, so framing differences in
     the source stop mattering.
  3. Resize premultiplied, then unpremultiply. Resizing straight RGBA
     bleeds the white of the transparent pixels into every edge and
     leaves a pale halo, which is very visible against the dark theme.
     Downscaling the binary mask is also what turns it back into a
     smooth alpha edge.
  4. Centre on one canvas of fixed size, fitting to the box. Fit rather
     than fill: an elongated tower must stay elongated -- that is the
     feature that names it -- so aspect is preserved and the leftover
     space is transparent.

Usage
-----
    PYTHONPATH=. python study/prepare_icons.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path("local comments/study icons")
OUT = Path("study/docs/roof")

# Canvas, in px. Rendered at 88 px wide in the option grid, so this is
# ~3.5x for retina headroom. 4:3.4 suits the widest drawings while still
# giving the tower somewhere to be tall.
CW, CH = 320, 272
MARGIN = 6

# Source file -> the option value the page stores in the response row.
# The three that are ever *correct* -- flat, gabled, monopitch -- keep the
# names the answer key already uses; changing those would silently break
# scoring. The rest are distractors, so their values only have to be
# distinct and readable in the exported data.
NAMES = {
    "Flat roof":                  "flat",
    "Gable roof":                 "gabled",
    "Hipped roof":                "hipped",
    "Jerkinhead roof":            "half_hipped",
    "Monopitch roof":             "monopitch",
    "Staggered mono-pitch roof":  "staggered_monopitch",
    "Shed roof":                  "sawtooth",
    "Mansard roof":               "mansard",
    "Arched roof":                "curved",
    "Tent roof":                  "tented",
    "Turmdach":                   "tower",
    "Conical roof":               "conical",
    "Domed roof":                 "domed",
}

# min(r,g,b) at or above this counts as background. The renders are
# anti-aliased against white, so a little headroom below 255 keeps the
# faint outer ring of a stroke from being cut into the mask.
WHITE = 243


def to_rgba(path: Path) -> Image.Image:
    """Load, key out the white background, crop to the drawing."""
    a = np.array(Image.open(path).convert("RGB")).astype(np.uint8)
    solid = a.min(axis=2) < WHITE
    if not solid.any():
        raise ValueError(f"{path.name}: no content found")

    out = np.dstack([a, np.where(solid, 255, 0).astype(np.uint8)])
    ys, xs = np.where(solid)
    return Image.fromarray(out[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def fit(im: Image.Image, box_w: int, box_h: int) -> Image.Image:
    """Resize to fit inside the box, preserving aspect, without halos."""
    s = min(box_w / im.width, box_h / im.height)
    w, h = max(1, round(im.width * s)), max(1, round(im.height * s))

    a = np.array(im).astype(np.float32)
    al = a[..., 3:4] / 255.0
    pre = np.dstack([a[..., :3] * al, a[..., 3:4]]).astype(np.uint8)
    r = np.array(Image.fromarray(pre).resize((w, h), Image.LANCZOS)
                 ).astype(np.float32)
    al2 = np.clip(r[..., 3:4], 0, 255) / 255.0
    rgb = np.where(al2 > 0.004, r[..., :3] / np.maximum(al2, 1e-6), 255.0)
    return Image.fromarray(
        np.dstack([np.clip(rgb, 0, 255), np.clip(r[..., 3:4], 0, 255)]
                  ).astype(np.uint8))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    missing = [k for k in NAMES if not (SRC / f"{k}.png").exists()]
    if missing:
        raise SystemExit(f"missing source icons: {missing}")

    for stem, value in NAMES.items():
        im = fit(to_rgba(SRC / f"{stem}.png"), CW - 2 * MARGIN, CH - 2 * MARGIN)
        canvas = Image.new("RGBA", (CW, CH), (0, 0, 0, 0))
        canvas.paste(im, ((CW - im.width) // 2, (CH - im.height) // 2), im)
        canvas.save(OUT / f"{value}.png")
        print(f"  {stem:<28} -> {value}.png   "
              f"drawn {im.width}x{im.height} in {CW}x{CH}")

    print(f"wrote {len(NAMES)} icons -> {OUT}")


if __name__ == "__main__":
    main()
