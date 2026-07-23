"""
Plot training loss and grad-norm curves for Phase B v1-v4 on one figure each.

Sources of truth (in preference order):
  1. losses.json / grad_norms.json in each phase_b*/ output dir
  2. train log file at model/checkpoints/phase_b*_train.log (line-parsed)

v1 predates the grad-norm logging feature added in v2 (see claude.md
§26.11), so its grad-norm trace is intentionally absent -- this is
itself a talking point for the presentation ("we added grad-norm
logging because v1 collapsed and we couldn't see why").

Usage:
    python -m model.src.plot_training                 # both figures
    python -m model.src.plot_training --metric loss   # loss only
    python -m model.src.plot_training --metric gn     # grad-norm only
    python -m model.src.plot_training --out-dir figures
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Order + colors for the version legend (colour-blind-friendly).
VERSIONS = [
    # (name, ckpt_dir, log_path, colour, iters_run)
    ("v1  (52M, no warmup)",       "phase_b",    "phase_b_train.log",    "#d62728",  20000),
    ("v2  (13.7M, warmup+clip)",   "phase_b_v2", "phase_b_v2_train.log", "#ff7f0e",  17800),
    ("v3  (fg-weighted MSE)",      "phase_b_v3", "phase_b_v3_train.log", "#9467bd",  20200),
    ("v4  (x0 + cross-entropy)",   "phase_b_v4", "phase_b_v4_train.log", "#2ca02c", 100000),
]

CKPT_ROOT = Path("model/checkpoints")

LOG_LINE_RE = re.compile(
    r"^\[train\] it=\s*(\d+)/\d+\s+loss=([\d.eE+-]+)"
    r"(?:\s+recent_min=[\d.eE+-]+)?"
    r"(?:\s+gn=([\d.eE+-]+)\s+\(max=([\d.eE+-]+)\))?"
)


def _parse_log(log_path: Path) -> tuple[list[int], list[float], list[float] | None]:
    """Return (iters, losses, gn_avg_or_None). Skips any header lines."""
    iters, losses, gns = [], [], []
    has_gn = False
    for line in log_path.read_text().splitlines():
        m = LOG_LINE_RE.match(line)
        if not m:
            continue
        it = int(m.group(1))
        loss = float(m.group(2))
        gn = float(m.group(3)) if m.group(3) else None
        iters.append(it)
        losses.append(loss)
        if gn is not None:
            gns.append(gn)
            has_gn = True
    return iters, losses, (gns if has_gn else None)


def _load_json_if_exists(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    try:
        return list(json.loads(path.read_text()))
    except (json.JSONDecodeError, OSError):
        return None


def _collect_series(kind: str) -> list[dict]:
    """kind is 'loss' or 'gn'. Returns per-version dicts of iters/values."""
    out = []
    for name, ckpt_dir, log_name, color, iters_run in VERSIONS:
        # First try the JSON dump (only survives on completed runs -- v4).
        json_path = CKPT_ROOT / ckpt_dir / (
            "losses.json" if kind == "loss" else "grad_norms.json"
        )
        json_series = _load_json_if_exists(json_path)
        if json_series is not None and len(json_series) > 0:
            # These are per-iteration values; downsample to every 100 to
            # match the log resolution and keep the figure light.
            values = np.asarray(json_series, dtype=np.float64)
            step = max(1, len(values) // 2000)
            iters = np.arange(len(values), dtype=np.int64)[::step] + 1
            values = values[::step]
            out.append({
                "name": name, "color": color, "iters": iters,
                "values": values, "source": "json",
            })
            continue

        # Fall back to log parsing.
        log_path = CKPT_ROOT / log_name
        if not log_path.exists():
            print(f"[plot_training] no data for {name}: neither {json_path} "
                  f"nor {log_path} exist")
            continue
        iters, losses, gns = _parse_log(log_path)
        if kind == "loss":
            values = losses
        else:
            values = gns
            if values is None:
                # e.g. v1: pre-gradnorm-logging. Skip cleanly with a note.
                out.append({
                    "name": name, "color": color, "iters": None,
                    "values": None, "note": "no grad-norm data (predates §26.11 logging)",
                })
                continue
        out.append({
            "name": name, "color": color,
            "iters": np.asarray(iters), "values": np.asarray(values),
            "source": "log",
        })
    return out


# ---- plotting -----------------------------------------------------------
def _plot_series(series: list[dict], kind: str, out_path: Path,
                 log_y: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=140)
    for s in series:
        if s.get("values") is None:
            continue
        ax.plot(s["iters"], s["values"], label=s["name"], color=s["color"],
                linewidth=1.4, alpha=0.85)

    ax.set_xlabel("training iteration")
    if kind == "loss":
        ax.set_ylabel("loss  (log scale)" if log_y else "loss")
        ax.set_title("Phase B — training loss across the four iterations "
                     "(v1 collapse → v4 working)")
    else:
        ax.set_ylabel("grad-norm avg over 100 iters  (log scale)"
                      if log_y else "grad-norm avg over 100 iters")
        ax.set_title("Phase B — gradient-norm behaviour across v2, v3, v4  "
                     "(v1 predates gradient-norm logging)")

    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(loc="best", fontsize=9)

    # Reference lines for context.
    if kind == "loss":
        ax.axhline(1.0, color="grey", linestyle=":", alpha=0.4,
                   label="_nolegend_")
        ax.text(1000, 1.0, "  MSE baseline (predict zero) = 1.0",
                fontsize=7, color="grey", va="bottom")
        ax.axhline(np.log(6), color="grey", linestyle=":", alpha=0.4,
                   label="_nolegend_")
        ax.text(1000, np.log(6),
                "  CE baseline (uniform softmax) = log(6) ≈ 1.79",
                fontsize=7, color="grey", va="bottom")
        ax.axhline(0.05, color="grey", linestyle=":", alpha=0.4,
                   label="_nolegend_")
        ax.text(1000, 0.05, "  claimed healthy MSE plateau ~ 0.05",
                fontsize=7, color="grey", va="bottom")
    else:
        ax.axhline(0.5, color="grey", linestyle=":", alpha=0.4,
                   label="_nolegend_")
        ax.text(1000, 0.5, "  grad_clip threshold (v2-v4) = 0.5",
                fontsize=7, color="grey", va="bottom")

    # Add absence-of-data notes to the legend area.
    notes = [s for s in series if s.get("values") is None]
    if notes:
        txt = "\n".join(f"• {n['name']}: {n.get('note', 'no data')}"
                       for n in notes)
        ax.text(0.98, 0.02, txt, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="grey",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="lightgrey", alpha=0.9))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[plot_training] wrote {out_path}")


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["loss", "gn", "both"], default="both")
    ap.add_argument("--out-dir", default="figures",
                    help="output directory for the PNGs")
    ap.add_argument("--linear-y", action="store_true",
                    help="use linear y-axis (default: log)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)

    if args.metric in ("loss", "both"):
        series = _collect_series("loss")
        _plot_series(series, "loss", out_dir / "loss_v1_v4.png",
                     log_y=not args.linear_y)

    if args.metric in ("gn", "both"):
        series = _collect_series("gn")
        _plot_series(series, "gn", out_dir / "gradnorm_v1_v4.png",
                     log_y=not args.linear_y)


if __name__ == "__main__":
    _cli()
