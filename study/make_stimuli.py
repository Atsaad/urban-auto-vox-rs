"""Render paired stimuli for the Pillar G perceptual study.

Design notes
------------
The study is a two-alternative forced choice: a participant sees one real
and one generated building and picks the one they believe is real. A
model whose outputs are indistinguishable drives accuracy towards 50 %.

The study is BLIND BY CONSTRUCTION. Image files are named by hash, and
the manifest that ships to the browser (`stimuli.json`) lists only which
two files form a pair -- it never says which is real. The mapping from
file to {real, generated, model, condition} is written to
`study/answer_key.json`, deliberately OUTSIDE the deployed `docs/`
directory so it cannot be served even by accident.
Responses are therefore scored offline, and there is nothing in the page
a participant could read to find the answer.

It records no identifiers, no free text, and no account -- only which
image was clicked, per trial.

Arms are additive: rerun with a different --gen to append a new model
(v5, Phase C, ...) into the same key under a different label.

Usage
-----
    python study/make_stimuli.py \
        --gen model/checkpoints/phase_b_v4/eval_w0/eval_samples.npz \
        --label v4_w0 --pairs 30
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from model.src.render import render_isometric

OUT = Path("study/docs")          # GitHub Pages serves from /docs
IMG = OUT / "img"


def _hashed_name(label: str, kind: str, idx: int, salt: str) -> str:
    """Opaque filename: nothing about the source is recoverable from it."""
    h = hashlib.sha256(f"{salt}|{label}|{kind}|{idx}".encode()).hexdigest()
    return f"{h[:16]}.png"


def crop_window(t: np.ndarray, win: int = 44) -> np.ndarray:
    """Cut a FIXED-size cubic window around the building.

    A 64^3 grid leaves most of the frame empty, which wastes display area
    and makes surface detail — the thing the study is actually about —
    hard to see on a phone.

    The window is a fixed size rather than the building's own bounding
    box, deliberately: cropping each building to its own extent would
    normalise size and destroy a legitimate realism cue. A fixed window
    removes empty margin while leaving big buildings looking big.

    Centred on the footprint in x/y, and taken from the ground up in z,
    since buildings rest on the ground plane rather than sitting mid-grid.
    """
    fg = t[1:].sum(0) > 0
    if not fg.any():
        return t
    D = t.shape[1]
    win = min(win, D)
    xs, ys, zs = (np.where(fg.any(axis=tuple(j for j in range(3) if j != i)))[0]
                  for i in range(3))
    out = [None, None, None]
    for i, v in enumerate((xs, ys)):
        c = (int(v.min()) + int(v.max())) // 2
        lo = max(0, min(c - win // 2, D - win))
        out[i] = slice(lo, lo + win)
    lo_z = max(0, min(int(zs.min()) - 1, D - win))
    out[2] = slice(lo_z, lo_z + win)
    return t[:, out[0], out[1], out[2]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True, help="eval_samples.npz with gen+real")
    ap.add_argument("--label", required=True, help="model arm name, e.g. v4_w0")
    ap.add_argument("--pairs", type=int, default=30)
    ap.add_argument("--salt", default="uav-pillar-g")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--window", type=int, default=44,
                    help="cubic crop window in voxels")
    a = ap.parse_args()

    IMG.mkdir(parents=True, exist_ok=True)
    z = np.load(a.gen, allow_pickle=True)
    gen, real = z["gen"], z["real"]
    gmeta = json.loads(str(z["gen_meta"]))
    rmeta = json.loads(str(z["real_meta"]))
    rng = np.random.default_rng(a.seed)

    # Pair within condition, so the choice is about realism and not about
    # having been shown two obviously different kinds of building.
    conds = sorted({m["name"] for m in gmeta})
    per = max(1, a.pairs // len(conds))
    # condition name -> requested roof type, for the answer key
    from model.src.sample import DEFAULT_CONDITIONS
    conds_by_name = {c["name"]: c.get("roof_type_label", "unknown")
                     for c in DEFAULT_CONDITIONS}

    key_path = OUT.parent / "answer_key.json"   # deliberately OUTSIDE docs/
    key = json.loads(key_path.read_text()) if key_path.exists() else {}
    stim_path = OUT / "stimuli.json"
    stim = json.loads(stim_path.read_text()) if stim_path.exists() else []

    pair_no = len(stim)
    for cond in conds:
        gi = [i for i, m in enumerate(gmeta) if m["name"] == cond]
        ri = [i for i, m in enumerate(rmeta) if m["name"] == cond]
        if not gi or not ri:
            continue
        for k in range(min(per, len(gi), len(ri))):
            g_idx = int(rng.choice(gi)); r_idx = int(rng.choice(ri))
            gname = _hashed_name(a.label, "gen", pair_no, a.salt)
            rname = _hashed_name(a.label, "real", pair_no, a.salt)
            render_isometric(crop_window(gen[g_idx], a.window),
                             out_path=IMG / gname, fig_size_inches=4.0)
            render_isometric(crop_window(real[r_idx], a.window),
                             out_path=IMG / rname, fig_size_inches=4.0)
            # Ground truth for the roof-identification task. For a
            # generated building it is the roof type that was REQUESTED;
            # for a real one it is the roof type recorded in CityGML.
            key[gname] = {"kind": "generated", "model": a.label,
                          "condition": cond, "index": g_idx,
                          "roof_type": conds_by_name[cond]}
            key[rname] = {"kind": "real", "model": a.label,
                          "condition": cond, "index": r_idx,
                          "roof_type": (rmeta[r_idx].get("roof_type_label")
                                        or "unknown")}
            # Trials are single images, not pairs: the task is to name the
            # roof type, which needs no comparison partner.
            stim.append({"id": pair_no * 2,     "img": gname})
            stim.append({"id": pair_no * 2 + 1, "img": rname})
            pair_no += 1
            print(f"  pair {pair_no:3d}  {cond}", flush=True)

    stim_path.write_text(json.dumps(stim, indent=1))
    key_path.write_text(json.dumps(key, indent=1))
    print(f"\nwrote {len(stim)} trials -> {stim_path}")
    print(f"answer key -> {key_path}  (outside docs/, never deployed)")


if __name__ == "__main__":
    main()
