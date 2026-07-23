"""
Building3DDataset — PyTorch Dataset over the sharded HDF5 produced by
build_tensors.py (see claude.md §18, §24).

Reads:
  - tensorbuilding/shards/<tag>_NNNN.h5     (one tensor per group + attrs)
  - tensorbuilding/shards/manifest.csv      (one row per building)

Returns per __getitem__:
  {
    "tensor"       : (C, D, D, D) float32 one-hot   (C=6, D=64)
    "cont"         : (n_cont,) float32 normalised   (NaN -> 0)
    "cat"          : dict[str, int]                 (categorical IDs)
    "drop"         : bool                           (used by CFG; see collate)
    "gmlid"        : str
  }

Vocabularies are built from manifest.csv at construction time and stored on the
Dataset object (Dataset.vocabs and Dataset.continuous_stats). Empty strings
become a dedicated "(null)" token id so a building with no
`estimatedConstructionPeriod` is just another category — no special path is
needed for partial-missingness inside the conditional pipeline.

Worker safety: h5py file handles are opened lazily PER WORKER (via the
get_worker_info hook), so spawning many DataLoader workers is fine.
"""

from __future__ import annotations

import csv
import glob
import math
import os
from collections import OrderedDict
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# ---- categorical & continuous feature list ------------------------------
# These names must exist in the manifest CSV (and as attrs on HDF5 groups).
# Order is stable — used by ConditionEncoder.
CATEGORICAL_COLS = [
    "function_label", "roof_type_label", "height_cluster", "ratio_cluster",
    "estimatedConstructionPeriod", "constructionPeriodReliability",
]
CONTINUOUS_COLS = [
    "measured_height", "length_to_width_ratio", "constructionPeriodConfidence",
    # storeys_above_ground is currently missing on the 5cities_balanced shards
    # (column-drop upstream — see claude.md §23 C4 gap note). We add it here
    # so future shards with the column work automatically.
    "storeys_above_ground",
]
# Sentinel used both in vocabs and as "empty string -> null id".
NULL_TOK = "(null)"


# ---- vocabulary helpers --------------------------------------------------
def _build_vocabs(manifest_rows):
    """{col: OrderedDict(value -> id)} including NULL_TOK as id 0."""
    vocabs: dict[str, OrderedDict] = {c: OrderedDict([(NULL_TOK, 0)]) for c in CATEGORICAL_COLS}
    for row in manifest_rows:
        for c in CATEGORICAL_COLS:
            v = (row.get(c) or "").strip()
            if v == "":
                continue
            if v not in vocabs[c]:
                vocabs[c][v] = len(vocabs[c])
    return vocabs


def _compute_cont_stats(manifest_rows):
    """{col: (mean, std)} from non-empty FINITE values. Robust to NaNs / missing.

    Important: the CSV stringifies NumPy NaNs as the literal text "nan", and
    `float("nan")` returns a NaN without raising. Without an explicit
    `isfinite` check those NaNs would poison the mean/std and propagate into
    every normalised sample.
    """
    stats: dict[str, tuple[float, float]] = {}
    for c in CONTINUOUS_COLS:
        xs = []
        for r in manifest_rows:
            v = r.get(c)
            if v is None or v == "":
                continue
            try:
                fv = float(v)
            except ValueError:
                continue
            if not math.isfinite(fv):
                continue
            xs.append(fv)
        if not xs:
            stats[c] = (0.0, 1.0)
            continue
        a = np.asarray(xs, dtype=np.float64)
        m = float(a.mean())
        s = float(a.std()) or 1.0
        stats[c] = (m, s)
    return stats


# ---- Dataset -------------------------------------------------------------
class Building3DDataset(Dataset):
    """Sharded-HDF5 dataset for the building voxel tensors."""

    def __init__(
        self,
        shards_dir: str,
        tag_glob: str = "*",
        manifest_path: Optional[str] = None,
        filter_fn: Optional[Callable[[dict], bool]] = None,
    ):
        super().__init__()
        self.shards_dir = shards_dir

        # 1. Manifest: source of conditions + the (gmlid -> shard) routing.
        if manifest_path is None:
            manifest_path = os.path.join(shards_dir, "manifest.csv")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        with open(manifest_path, newline="") as f:
            rows = list(csv.DictReader(f))

        # 2. Optional filter (used for Pattern E train/val/test splits).
        if filter_fn is not None:
            rows = [r for r in rows if filter_fn(r)]
        if not rows:
            raise ValueError("No samples after filtering — check filter_fn / manifest.")

        # 3. Vocabs + continuous stats (built once on the kept rows).
        self.vocabs = _build_vocabs(rows)
        self.continuous_stats = _compute_cont_stats(rows)

        # 4. Per-sample index.
        self.records = []
        for r in rows:
            shard = r.get("shard") or ""
            if not shard:
                continue
            self.records.append({
                "gmlid": r["gmlid"],
                "shard_path": os.path.join(shards_dir, shard),
                "row": r,
            })

        # 5. Per-worker shard file handle cache (lazy).
        self._handles: dict[str, h5py.File] = {}

        # 6. Sanity check tensor metadata against the first shard.
        first = self.records[0]
        with h5py.File(first["shard_path"], "r") as f:
            self.voxel_size = float(f.attrs.get("voxel_size", 0.5))
            self.target = int(f.attrs.get("target", 64))
            self.channel_names = str(f.attrs.get("channel_names", ""))
            t = f[f"buildings/{first['gmlid']}/tensor"]
            self.tensor_shape = tuple(t.shape)
            assert t.dtype == np.uint8, f"expected uint8 tensors, got {t.dtype}"

    # -- vocabulary sizes for the condition encoder ----------------------
    @property
    def vocab_sizes(self) -> dict[str, int]:
        return {c: len(self.vocabs[c]) for c in CATEGORICAL_COLS}

    # -- handle cache: opened per worker, never crosses process boundary --
    def _handle(self, path: str) -> h5py.File:
        h = self._handles.get(path)
        if h is None:
            # SWMR off (we only read), libver "latest" to match writer.
            h = h5py.File(path, "r", libver="latest", swmr=False)
            self._handles[path] = h
        return h

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        rec = self.records[i]
        grp = self._handle(rec["shard_path"])[f"buildings/{rec['gmlid']}"]

        # Tensor: stored uint8 one-hot, model wants float32.
        tensor = torch.from_numpy(grp["tensor"][...]).to(torch.float32)

        # Categorical IDs (empty -> NULL_TOK -> id 0).
        cat: dict[str, int] = {}
        for c in CATEGORICAL_COLS:
            raw = (str(grp.attrs.get(c, "")) or "").strip()
            cat[c] = self.vocabs[c].get(raw if raw else NULL_TOK, 0)

        # Continuous: standardise; NaN/missing -> 0 (the post-norm mean).
        cont = np.zeros(len(CONTINUOUS_COLS), dtype=np.float32)
        cont_cs = grp.get("conditions_continuous")
        cont_arr = cont_cs[...] if cont_cs is not None else None
        # Continuous fields the shard already exposes (in order):
        # measured_height, length_to_width_ratio, storeys, periodConfidence.
        # The manifest CSV columns are the source of truth for normalisation.
        for k, col in enumerate(CONTINUOUS_COLS):
            row_val = rec["row"].get(col)
            if row_val is None or row_val == "":
                continue
            try:
                v = float(row_val)
            except ValueError:
                continue
            if not math.isfinite(v):
                continue
            m, s = self.continuous_stats[col]
            cont[k] = (v - m) / s

        return {
            "tensor": tensor,
            "cont": torch.from_numpy(cont),
            "cat": cat,
            "gmlid": rec["gmlid"],
        }


# ---- DataLoader collate --------------------------------------------------
def collate_batch(samples: list[dict]) -> dict:
    """Stack a list of __getitem__ outputs into a model-ready batch."""
    batch = {
        "tensor": torch.stack([s["tensor"] for s in samples], dim=0),
        "cont":   torch.stack([s["cont"]   for s in samples], dim=0),
        "gmlid":  [s["gmlid"] for s in samples],
    }
    # Categoricals: one (B,) long tensor per feature, in fixed CATEGORICAL_COLS order.
    for c in CATEGORICAL_COLS:
        batch[f"cat_{c}"] = torch.tensor(
            [s["cat"][c] for s in samples], dtype=torch.long
        )
    return batch
