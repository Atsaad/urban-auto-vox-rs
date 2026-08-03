"""Render the roof-type options as voxel buildings.

The first attempt drew them as hand-authored isometric line art, and it
did not work: at icon size, gabled, hipped, half-hipped, mansard,
gambrel and curved all collapse into the same small blob, which is
exactly the discrimination the task depends on.

These are generated instead with the same renderer, projection and
colours as the stimuli, so an option looks like the thing it names. Blue
walls, red roof, green ground -- the participant is matching like to
like rather than translating a diagram.

Each is a small idealised building: a rectangular footprint with walls,
carrying one canonical roof.

Usage
-----
    PYTHONPATH=. python study/make_roof_icons.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from model.src.render import render_isometric

OUT = Path("study/docs/roof")
# Geometry and proportions follow the AdV roof-form codelist (Dachform
# 1000-4000), the same reference the source data is coded against, so an
# option means what the standard says it means. Proportions matter as
# much as shape there: the codelist draws a LOW wall band under a
# prominent roof, which is what makes Walmdach read differently from
# Zeltdach at a glance. Tall walls flatten that distinction.
#
# The grid is larger than the shapes need because the roof is a height
# map quantised to voxels, and coarse steps blur exactly the
# discriminations the options exist to convey.
D = 46
X0, X1 = 5, 41              # wide, shallow footprint as in the codelist
Y0, Y1 = 13, 33
SQ = 11, 35                  # square footprint, for pyramid and tented:
                            # min(dx,dy) on a rectangle gives a ridge, i.e.
                            # a hipped roof, not a point
ZW = 6, 12                  # low walls: roof dominates, per the codelist                  # wall band
WALL, ROOF, GROUND = 1, 2, 3


def _footprint(kind: str) -> tuple[int, int, int, int]:
    if kind in ("pyramid", "tented"):
        return SQ[0], SQ[1], SQ[0], SQ[1]
    return X0, X1, Y0, Y1


def _height_map(kind: str) -> np.ndarray:
    """Roof height above the wall top, per (x, y) cell of the footprint."""
    x0, x1, y0, y1 = _footprint(kind)
    nx, ny = x1 - x0, y1 - y0
    xs = np.arange(nx)[:, None].repeat(ny, 1)
    ys = np.arange(ny)[None, :].repeat(nx, 0)
    cx, cy = (nx - 1) / 2, (ny - 1) / 2
    # distance to the nearer edge, per axis, normalised
    dx = 1 - np.abs(xs - cx) / cx
    dy = 1 - np.abs(ys - cy) / cy

    if kind == "flat":        return np.zeros((nx, ny))
    if kind == "gabled":      return 13 * dy
    if kind == "hipped":      return 13 * np.minimum(dx * 2.2, dy)
    if kind == "half_hipped":
        h = 13 * dy
        clip = (xs < nx * .12) | (xs > nx * .88)
        return np.where(clip, np.minimum(h, 13 * dx * 2.4), h)
    if kind == "monopitch":   return 11 * (ys / (ny - 1))
    if kind == "shed":        return 16 * (ys / (ny - 1))
    if kind == "mansard":
        # steep lower slope then shallow upper, on all four sides
        m = np.minimum(dx, dy)
        return np.where(m < .45, 22 * m, 10.0 + 4.5 * (m - .45))
    if kind == "gambrel":
        # the same profile but only across the ridge axis: a barn
        return np.where(dy < .45, 22 * dy, 10.0 + 4.5 * (dy - .45))
    if kind == "pyramid":     return 16 * np.minimum(dx, dy)
    if kind == "tented":      return 26 * np.minimum(dx, dy)
    if kind == "curved":      return 13 * np.sin(np.clip(dy, 0, 1) * np.pi / 2)
    if kind == "butterfly":   return 10 * (1 - dy)
    raise ValueError(kind)


def build(kind: str) -> np.ndarray:
    t = np.zeros((7, D, D, D), dtype=np.uint8)
    z0, z1 = ZW
    x0, x1, y0, y1 = _footprint(kind)

    t[GROUND, x0:x1, y0:y1, z0 - 1] = 1
    t[WALL, x0:x1, y0:y1, z0:z1] = 1                       # shell, not solid
    t[WALL, x0 + 1:x1 - 1, y0 + 1:y1 - 1, z0:z1] = 0

    hm = _height_map(kind)
    nx, ny = x1 - x0, y1 - y0
    zz = np.minimum(z1 + np.rint(hm).astype(int), D - 1)

    for i in range(nx):
        for j in range(ny):
            top = int(zz[i, j])
            t[ROOF, x0 + i, y0 + j, top] = 1
            # A steep slope leaves a vertical gap between a cell and its
            # lower neighbour, which reads as a hole in the roof. Close it
            # down to the HIGHEST neighbour only -- filling the whole column
            # to the wall top turns the roof into a solid block, which is
            # what the first attempt did.
            lo = top
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                nb = (int(zz[ii, jj]) if 0 <= ii < nx and 0 <= jj < ny
                      else z1 - 1)
                lo = min(lo, nb + 1)
            for k in range(max(lo, z1), top):
                t[ROOF, x0 + i, y0 + j, k] = 1

    t[0] = (t[1:].sum(0) == 0).astype(np.uint8)
    return t


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    kinds = ["flat", "gabled", "hipped", "half_hipped", "monopitch", "shed",
             "mansard", "gambrel", "pyramid", "tented", "curved", "butterfly"]
    for k in kinds:
        render_isometric(build(k), out_path=OUT / f"{k}.png",
                         fig_size_inches=1.8, elev=26, azim=-62)
        print(f"  {k}")
    print(f"wrote {len(kinds)} icons -> {OUT}")


if __name__ == "__main__":
    main()
