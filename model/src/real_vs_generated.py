"""
Real vs generated side-by-side comparison for the intermediate presentation.

For each of the 4 DEFAULT_CONDITIONS from sample.py, this script:
  1. Finds the closest matching REAL building in the training dataset
     (must-match on function_label + roof_type_label; nearest on
     measured_height + length_to_width_ratio + storeys).
  2. Generates a fresh sample from a v4 checkpoint with the same condition
     (single seed, DDIM 50 steps, CFG 1.5).
  3. Renders the real vs generated pair side-by-side.

Output:  a 4 x 2 grid PNG (rows = conditions, columns = real | generated),
each cell is a triple-orthographic projection (top / front / side).

Usage:
    python -m model.src.real_vs_generated \
        --ckpt model/checkpoints/phase_b_v4/ckpt_100000.pt \
        --out  figures/real_vs_v4.png

Optional flags:
    --seed N          seed for the generated sample (default 42)
    --isometric       polished 3D isometric render instead of triple-projection
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import Building3DDataset
from .render import _label_field, _labels_to_rgb, _project_max_class, render_isometric
from .sample import DEFAULT_CONDITIONS, load_checkpoint, sample as ddim_sample, to_hard_onehot


# ---- matching real buildings ---------------------------------------------
def _condition_match_score(row: dict, cond: dict) -> float:
    """Lower is better. inf = disqualifying mismatch.

    Categoricals must match exactly (function_label + roof_type_label).
    Continuous fields contribute normalised absolute-difference terms.
    """
    for k in ("function_label", "roof_type_label"):
        if str(row.get(k, "")).strip() != str(cond.get(k, "")).strip():
            return math.inf

    score = 0.0
    for col, weight in (
        ("measured_height", 1.0),
        ("length_to_width_ratio", 0.5),
        ("storeys_above_ground", 0.5),
    ):
        want = cond.get(col)
        got = row.get(col)
        if want is None or got in (None, "", "nan"):
            continue
        try:
            got_f = float(got)
            want_f = float(want)
            if not (math.isfinite(got_f) and math.isfinite(want_f)):
                continue
            score += weight * abs(got_f - want_f) / max(1.0, abs(want_f))
        except (TypeError, ValueError):
            continue
    return score


def _find_best_real(ds: Building3DDataset, cond: dict) -> tuple[int, dict, float]:
    """Return (dataset_index, row_dict, score) for the best-matching real bldg."""
    best_idx, best_row, best_score = None, None, math.inf
    for i, rec in enumerate(ds.records):
        s = _condition_match_score(rec["row"], cond)
        if s < best_score:
            best_score, best_idx, best_row = s, i, rec["row"]
    if best_idx is None:
        raise RuntimeError(f"No real building matches condition {cond['name']}")
    return best_idx, best_row, best_score


# ---- rendering helpers ---------------------------------------------------
def _draw_triple(ax_top, ax_front, ax_side, tensor_uint8: np.ndarray) -> None:
    label = _label_field(tensor_uint8)
    top   = _project_max_class(label, axis=2)
    front = np.flipud(_project_max_class(label, axis=1).T)
    side  = np.flipud(_project_max_class(label, axis=0).T)
    for ax, view, name in (
        (ax_top,   top,   "top (X-Y)"),
        (ax_front, front, "front (X-Z)"),
        (ax_side,  side,  "side (Y-Z)"),
    ):
        ax.imshow(_labels_to_rgb(view), interpolation="nearest", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(name, fontsize=7, pad=2)


def _tensor_to_hard(tensor_float: np.ndarray) -> np.ndarray:
    """(C, D, D, D) float -> (C, D, D, D) uint8 one-hot via argmax."""
    idx = tensor_float.argmax(axis=0)                            # (D, D, D)
    hard = np.zeros_like(tensor_float, dtype=np.uint8)
    for c in range(tensor_float.shape[0]):
        hard[c] = (idx == c).astype(np.uint8)
    return hard


# ---- main --------------------------------------------------------------
def build_figure(
    ckpt_path: str,
    out_path: str,
    seed: int = 42,
    isometric: bool = False,
    conditions: list[dict] | None = None,
) -> None:
    conds = list(conditions or DEFAULT_CONDITIONS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[real_vs_generated] loading {ckpt_path} on {device}")
    ckpt = load_checkpoint(ckpt_path, device)

    cfg = ckpt["cfg"]
    print(f"[real_vs_generated] parameterization="
          f"{cfg.get('parameterization', 'eps')}, "
          f"batch_size={cfg.get('batch_size')}, iters={cfg.get('iters')}")

    print(f"[real_vs_generated] indexing training set for closest matches")
    ds = Building3DDataset(
        shards_dir=cfg["shards_dir"],
        manifest_path=cfg["manifest_path"],
    )

    # For each condition: pick best real + generate one sample from v4.
    real_tensors, gen_tensors, real_meta = [], [], []
    for cond in conds:
        idx, row, score = _find_best_real(ds, cond)
        item = ds[idx]
        real_tensors.append(_tensor_to_hard(item["tensor"].numpy()))
        real_meta.append({"gmlid": row.get("gmlid", "?"),
                          "score": round(score, 3),
                          "cond_name": cond["name"]})
        print(f"[real_vs_generated] {cond['name']:36s}  "
              f"matched {row.get('gmlid')} (score {score:.2f})")

    print(f"[real_vs_generated] generating {len(conds)} v4 samples "
          f"(seed={seed}, DDIM 50 steps, CFG 1.5)")
    x, meta = ddim_sample(
        ckpt["unet"], ckpt["cond_enc"], ckpt["schedule"],
        conditions=conds,
        vocabs=ckpt["vocabs"],
        continuous_stats=ckpt["continuous_stats"],
        seeds=[seed],
        n_steps=50, guidance_scale=1.5, device=device,
        parameterization=str(cfg.get("parameterization", "eps")),
    )
    hard = to_hard_onehot(x).cpu().numpy()
    for i in range(len(conds)):
        gen_tensors.append(hard[i])

    # Layout: rows = conditions, columns split into (real triple | generated triple).
    rows = len(conds)
    if isometric:
        fig = plt.figure(figsize=(6.5, 2.6 * rows), dpi=140)
        for r, cond in enumerate(conds):
            ax_l = fig.add_subplot(rows, 2, r * 2 + 1, projection="3d")
            ax_r = fig.add_subplot(rows, 2, r * 2 + 2, projection="3d")
            render_isometric(real_tensors[r], ax=ax_l,
                             title=f"real  ·  {real_meta[r]['gmlid']}")
            render_isometric(gen_tensors[r], ax=ax_r,
                             title=f"generated  ·  seed={seed}")
            # Row label using a text next to the leftmost cell.
            ax_l.text2D(-0.15, 0.5, cond["name"], transform=ax_l.transAxes,
                        rotation=90, va="center", ha="right",
                        fontsize=8, fontweight="bold")
        fig.suptitle("Real training buildings vs. Phase B v4 samples "
                     "(same condition)", fontsize=11)
        fig.tight_layout(rect=(0.05, 0, 1, 0.96))
    else:
        fig, ax_arr = plt.subplots(
            rows, 6,
            figsize=(2.0 * 6 + 1.0, 1.7 * rows + 0.8),
            dpi=140,
        )
        ax_arr = np.atleast_2d(ax_arr)
        for r, cond in enumerate(conds):
            _draw_triple(ax_arr[r, 0], ax_arr[r, 1], ax_arr[r, 2],
                         real_tensors[r])
            _draw_triple(ax_arr[r, 3], ax_arr[r, 4], ax_arr[r, 5],
                         gen_tensors[r])
            ax_arr[r, 1].set_xlabel(f"real  ·  {real_meta[r]['gmlid'][:22]}",
                                    fontsize=7)
            ax_arr[r, 4].set_xlabel(f"generated (v4)  ·  seed={seed}", fontsize=7)
            ax_arr[r, 0].set_ylabel(cond["name"], fontsize=8,
                                    fontweight="bold", rotation=90, labelpad=8)
        fig.suptitle("Real training buildings vs. Phase B v4 samples "
                     "(same condition)", fontsize=11)
        fig.tight_layout(pad=0.4)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[real_vs_generated] wrote {out}")


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="v4 checkpoint, e.g. model/checkpoints/phase_b_v4/ckpt_100000.pt")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--isometric", action="store_true",
                    help="use 3D isometric render (slower, prettier)")
    args = ap.parse_args()
    build_figure(args.ckpt, args.out, seed=args.seed, isometric=args.isometric)


if __name__ == "__main__":
    _cli()
