"""
Extract clean CSV / JSON copies of the raw data behind every presentation
figure, so the figures can be reproduced (or restyled in another tool)
without re-running the training logs / raw HDF5 shards.

Output layout under figures/data/:
  losses_v1.csv, losses_v2.csv, losses_v3.csv, losses_v4.csv
    columns: iter,loss,recent_min,lr,it_per_s,elapsed_s
  gradnorms_v2.csv, gradnorms_v3.csv, gradnorms_v4.csv
    columns: iter,gn_avg_100,gn_max_100
    (v1 has no grad-norm data -- feature added in v2, see claude.md §26.11)
  class_frequencies.csv
    columns: version,class_idx,class_name,frequency
    versions: real (empirical training-data marginal), v2, v3, v4
  occupancy_trend.csv
    columns: version,step,condition,occupancy_pct,wall_pct,roof_pct,ground_pct
  real_vs_v4_metadata.json
    which real training buildings were matched for each condition + match
    score, so a reader can look them up in the CityGML source.

Usage:
    python -m model.src.dump_figure_data --out-dir figures/data
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .dataset import Building3DDataset
from .real_vs_generated import _find_best_real
from .sample import DEFAULT_CONDITIONS


CKPT_ROOT = Path("model/checkpoints")

# One entry per Phase-B version.
VERSIONS = [
    # (name, ckpt_dir_relative, log_filename)
    ("v1", "phase_b",    "phase_b_train.log"),
    ("v2", "phase_b_v2", "phase_b_v2_train.log"),
    ("v3", "phase_b_v3", "phase_b_v3_train.log"),
    ("v4", "phase_b_v4", "phase_b_v4_train.log"),
]

CLASS_NAMES = ["empty", "wall", "roof", "ground", "outer_ceiling", "closure"]

LOG_LINE_RE = re.compile(
    r"^\[train\] it=\s*(\d+)/\d+\s+loss=([\d.eE+-]+)"
    r"(?:\s+recent_min=([\d.eE+-]+))?"
    r"(?:\s+gn=([\d.eE+-]+)\s+\(max=([\d.eE+-]+)\))?"
    r"(?:\s+lr=([\d.eE+-]+))?"
    r"(?:\s+it/s=([\d.eE+-]+))?"
    r"(?:\s+elapsed=(\d+)s)?"
)


def _parse_log(log_path: Path) -> list[dict]:
    """Return one dict per logged step."""
    rows = []
    for line in log_path.read_text().splitlines():
        m = LOG_LINE_RE.match(line)
        if not m:
            continue
        rows.append({
            "iter":       int(m.group(1)),
            "loss":       float(m.group(2)),
            "recent_min": float(m.group(3)) if m.group(3) else None,
            "gn_avg":     float(m.group(4)) if m.group(4) else None,
            "gn_max":     float(m.group(5)) if m.group(5) else None,
            "lr":         float(m.group(6)) if m.group(6) else None,
            "it_per_s":   float(m.group(7)) if m.group(7) else None,
            "elapsed_s":  int(m.group(8)) if m.group(8) else None,
        })
    return rows


def _write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[dump_figure_data]  wrote {path}  ({len(rows)} rows)")


# ---- loss & grad-norm dumps ---------------------------------------------
def _dump_loss_and_gradnorm(out_dir: Path) -> None:
    for v_name, ckpt_dir, log_name in VERSIONS:
        log_path = CKPT_ROOT / log_name
        if not log_path.exists():
            print(f"[dump_figure_data]  skip {v_name}: no log at {log_path}")
            continue
        rows = _parse_log(log_path)
        loss_rows = [
            {"iter": r["iter"], "loss": r["loss"],
             "recent_min": r["recent_min"], "lr": r["lr"],
             "it_per_s": r["it_per_s"], "elapsed_s": r["elapsed_s"]}
            for r in rows
        ]
        _write_csv(
            out_dir / f"losses_{v_name}.csv",
            ["iter", "loss", "recent_min", "lr", "it_per_s", "elapsed_s"],
            loss_rows,
        )
        gn_rows = [
            {"iter": r["iter"], "gn_avg_100": r["gn_avg"],
             "gn_max_100": r["gn_max"]}
            for r in rows if r["gn_avg"] is not None
        ]
        if gn_rows:
            _write_csv(
                out_dir / f"gradnorms_{v_name}.csv",
                ["iter", "gn_avg_100", "gn_max_100"],
                gn_rows,
            )
        else:
            print(f"[dump_figure_data]  skip gradnorms_{v_name}.csv:"
                  f" no gn data (predates §26.11 logging)")


# ---- class frequency dump ----------------------------------------------
def _class_freq_from_samples(samples_dir: Path) -> np.ndarray | None:
    """Read latest step_*/samples.npz and return normalised class fractions."""
    steps = sorted(int(p.name[5:]) for p in samples_dir.glob("step_*")
                   if p.is_dir() and p.name[5:].isdigit())
    if not steps:
        return None
    latest = samples_dir / f"step_{steps[-1]:06d}" / "samples.npz"
    if not latest.exists():
        return None
    npz = np.load(latest, allow_pickle=True)
    tensors = npz["tensors"]
    labs = tensors.argmax(axis=1)
    totals = np.zeros(6, dtype=np.float64)
    for c in range(6):
        totals[c] = (labs == c).sum()
    return totals / totals.sum()


def _dump_class_frequencies(out_dir: Path, cfg_path: str,
                            n_real_samples: int = 200) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    ds = Building3DDataset(
        shards_dir=cfg["shards_dir"], manifest_path=cfg["manifest_path"],
    )
    rng = random.Random(42)
    idxs = rng.sample(range(len(ds)), min(n_real_samples, len(ds)))
    real_totals = np.zeros(6, dtype=np.float64)
    for i in idxs:
        x = ds[i]["tensor"].numpy()
        lab = x.argmax(axis=0)
        for c in range(6):
            real_totals[c] += (lab == c).sum()
    real_freq = real_totals / real_totals.sum()

    rows = []
    for c in range(6):
        rows.append({"version": "real", "class_idx": c,
                     "class_name": CLASS_NAMES[c],
                     "frequency": float(real_freq[c])})
    for v_name in ("v2", "v3", "v4"):
        samples_dir = CKPT_ROOT / f"phase_b_{v_name}" / "samples"
        freq = _class_freq_from_samples(samples_dir)
        if freq is None:
            continue
        for c in range(6):
            rows.append({"version": v_name, "class_idx": c,
                         "class_name": CLASS_NAMES[c],
                         "frequency": float(freq[c])})
    _write_csv(
        out_dir / "class_frequencies.csv",
        ["version", "class_idx", "class_name", "frequency"],
        rows,
    )


# ---- occupancy trend dump ----------------------------------------------
def _dump_occupancy_trend(out_dir: Path) -> None:
    rows = []
    for v_name in ("v2", "v3", "v4"):
        samples_dir = CKPT_ROOT / f"phase_b_{v_name}" / "samples"
        if not samples_dir.exists():
            continue
        steps = sorted(int(p.name[5:]) for p in samples_dir.glob("step_*")
                       if p.is_dir() and p.name[5:].isdigit())
        for step in steps:
            npz_path = samples_dir / f"step_{step:06d}" / "samples.npz"
            if not npz_path.exists():
                continue
            npz = np.load(npz_path, allow_pickle=True)
            tensors = npz["tensors"]
            meta = json.loads(str(npz["meta"]))
            labs = tensors.argmax(axis=1)
            by_cond = defaultdict(list)
            for i, m in enumerate(meta):
                by_cond[m["name"]].append(labs[i])
            for cond, lab_list in by_cond.items():
                total_vox = 0
                per_class = np.zeros(6, dtype=np.float64)
                for lab in lab_list:
                    total_vox += lab.size
                    for c in range(6):
                        per_class[c] += (lab == c).sum()
                per_class /= total_vox
                rows.append({
                    "version": v_name, "step": step, "condition": cond,
                    "occupancy_pct": float(100 * (1 - per_class[0])),
                    "empty_pct": float(100 * per_class[0]),
                    "wall_pct":  float(100 * per_class[1]),
                    "roof_pct":  float(100 * per_class[2]),
                    "ground_pct": float(100 * per_class[3]),
                })
    _write_csv(
        out_dir / "occupancy_trend.csv",
        ["version", "step", "condition", "occupancy_pct",
         "empty_pct", "wall_pct", "roof_pct", "ground_pct"],
        rows,
    )


# ---- real-vs-v4 metadata dump ------------------------------------------
def _dump_real_vs_v4_metadata(out_dir: Path, cfg_path: str) -> None:
    cfg = yaml.safe_load(open(cfg_path))
    ds = Building3DDataset(
        shards_dir=cfg["shards_dir"], manifest_path=cfg["manifest_path"],
    )
    out = []
    for cond in DEFAULT_CONDITIONS:
        idx, row, score = _find_best_real(ds, cond)
        out.append({
            "condition_name": cond["name"],
            "condition_dict": cond,
            "matched_gmlid": row.get("gmlid"),
            "matched_shard": row.get("shard"),
            "match_score": round(score, 4),
            "matched_row": {k: str(v) for k, v in row.items()
                            if k in ("gmlid", "function_label",
                                     "roof_type_label", "measured_height",
                                     "length_to_width_ratio",
                                     "storeys_above_ground",
                                     "estimatedConstructionPeriod",
                                     "constructionPeriodReliability",
                                     "gemeindeschluessel", "kreis")},
        })
    path = out_dir / "real_vs_v4_metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[dump_figure_data]  wrote {path}  ({len(out)} conditions)")


# ---- CLI ----------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="figures/data")
    ap.add_argument("--config", default="model/configs/phase_b.yaml")
    ap.add_argument("--n-real", type=int, default=200,
                    help="training buildings sampled for real class dist")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["losses", "class_freq", "occupancy", "matches"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "losses" not in args.skip:
        print("[dump_figure_data] parsing training logs...")
        _dump_loss_and_gradnorm(out_dir)
    if "class_freq" not in args.skip:
        print("[dump_figure_data] computing class frequencies...")
        _dump_class_frequencies(out_dir, args.config, args.n_real)
    if "occupancy" not in args.skip:
        print("[dump_figure_data] computing occupancy trends...")
        _dump_occupancy_trend(out_dir)
    if "matches" not in args.skip:
        print("[dump_figure_data] resolving real-vs-v4 match metadata...")
        _dump_real_vs_v4_metadata(out_dir, args.config)

    print(f"[dump_figure_data] done -> {out_dir}")


if __name__ == "__main__":
    main()
