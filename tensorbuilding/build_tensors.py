#!/usr/bin/env python3
"""
build_tensors.py — assemble (C, T, T, T) semantic voxel tensors + condition
vectors into sharded HDF5, ready for the diffusion DataLoader.

This implements STEP 6 (tensor export) + STEP 7 (HDF5 assembler) of the thesis
plan in `local comments/projectoverview/claude.md`.

INPUTS (all produced by the chunk voxelization pipeline, read-only here):
  - tensorbuilding/building_metadata_clean.csv   9.99M rows, the CONDITIONS CSV
  - voxel_csvs/chunk_NNNNN_voxels.csv            (x,y,z,surface_class) per voxel
  - voxel_csvs/chunk_NNNNN_grid_sizes.csv        per-building bbox dims + fits_64

PIPELINE (one streaming pass over each input — never loads a whole voxel CSV):
  1. Scan all *_grid_sizes.csv of DONE chunks -> gmlid -> chunk_id, and the
     fits_64 set (buildings whose voxel bbox fits the target grid).
  2. Stream the big metadata CSV once, keep rows matching the selection filter
     (function / roof / height) AND present in the fits_64 set. Sample to --cap.
  3. Group chosen gmlids by chunk. For each needed chunk, stream its voxel CSV
     once (manual comma-split, no csv module), collecting voxels only for chosen
     buildings, build the tensor, and write it to the current HDF5 shard.
  4. Write a flat manifest.csv (one row per building) for fast DataLoader indexing.

TENSOR ENCODING (matches the pipeline's own grid_sizes binning exactly):
  local_index = int((coord - coord_min) / voxel_size)        # 0 .. cells-1
  per-axis cells == w/d/h in grid_sizes  ->  fits iff max(cells) <= target
  The building is centred in a target^3 grid; one-hot across C channels.
  Axis order: dim1 = X (width), dim2 = Y (depth), dim3 = Z (height, GIS up).

CHANNELS (surface_class -> channel; see claude.md section 6 & 10):
  0 = empty (derived: set where no surface voxel lands)
  1 = WallSurface       2 = RoofSurface        3 = GroundSurface
  4 = OuterCeilingSurface   5 = ClosureSurface
  surface_class 0 (Unknown) does not occur in Munich LOD2; if seen it is
  skipped and counted (reported at the end).

Safe to run while the voxelization pipeline is active: reads completed
voxel_csvs only, writes solely under tensorbuilding/shards/.
"""

import argparse
import csv
import glob
import os
import random
import sys
import time
from collections import Counter, defaultdict

import h5py
import numpy as np

# ---- fixed encoding -------------------------------------------------------
N_CHANNELS = 6
SURFACE_TO_CHANNEL = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}  # class 0 (Unknown) -> skip
CHANNEL_NAMES = ["empty", "wall", "roof", "ground", "outer_ceiling", "closure"]

# Metadata CSV columns I carry into the dataset as the condition vector.
# The construction-period columns (added 2026-05-25, see claude.md §20 / §23 B1)
# come from the Zensus-enriched CSV; on the unenriched CSV they resolve to ""
# / NaN, which the DataLoader treats as the CFG null token.
META_CONTINUOUS = [
    "measured_height", "length_to_width_ratio", "storeys_above_ground",
    "constructionPeriodConfidence",
]
META_CATEGORICAL = [
    "function_label", "roof_type_label", "height_cluster", "ratio_cluster",
    "gemeindeschluessel", "kreis", "gemeinde",
    "estimatedConstructionPeriod", "constructionPeriodReliability",
    "storeys_source",
]

# AGS prefix -> friendly name for the 5 thesis cities (§12 Pattern C).
# `--cities` accepts either the prefix or this name (case-insensitive).
CITY_PREFIXES = {
    "munich": "09162", "augsburg": "09761", "nuernberg": "09564",
    "nurnberg": "09564", "wuerzburg": "09663", "wurzburg": "09663",
    "regensburg": "09362",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def scan_grid_sizes(voxel_dir, target):
    """gmlid -> chunk_id for buildings that fit `target` (max_cells <= target)."""
    gmlid_chunk = {}
    files = sorted(glob.glob(os.path.join(voxel_dir, "chunk_*_grid_sizes.csv")))
    if not files:
        sys.exit(f"No *_grid_sizes.csv found in {voxel_dir} — nothing voxelized yet.")
    for gf in files:
        chunk_id = os.path.basename(gf).replace("_grid_sizes.csv", "")
        with open(gf, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                if int(row["max_cells"]) <= target:
                    gmlid_chunk[row["gmlid"]] = chunk_id
    log(f"grid_sizes: {len(files)} done chunks, "
        f"{len(gmlid_chunk):,} buildings fit {target}^3")
    return gmlid_chunk


def _city_prefix_for(ags, prefixes):
    """Return the matching prefix from `prefixes` for an AGS, or None."""
    for p in prefixes:
        if ags.startswith(p):
            return p
    return None


def select_buildings(meta_csv, fit_set, args, prefixes):
    """Stream the big conditions CSV once; keep matching rows; sample.

    If `prefixes` is non-empty, only buildings whose `gemeindeschluessel` starts
    with one of those prefixes are kept (city filter, §23). If `--per-city-cap`
    is set, sampling is balanced per matched prefix; otherwise the global `--cap`
    applies after city-filtering.
    """
    chosen = []
    matched = 0
    by_city = defaultdict(list)
    with open(meta_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            g = row["gmlid"]
            if g not in fit_set:
                continue
            ags = row.get("gemeindeschluessel", "") or ""
            city_p = _city_prefix_for(ags, prefixes) if prefixes else None
            if prefixes and city_p is None:
                continue
            if args.function and row["function_label"] != args.function:
                continue
            if args.roof and row["roof_type_label"] != args.roof:
                continue
            try:
                h = float(row["measured_height"])
            except (ValueError, KeyError):
                h = -1.0
            if args.max_height and not (0 < h < args.max_height):
                continue
            matched += 1
            chosen.append(row)
            if prefixes:
                by_city[city_p].append(row)
    log(f"metadata: {matched:,} buildings match the selection filter")

    if args.per_city_cap and prefixes:
        rng = random.Random(args.seed)
        sampled = []
        for p in prefixes:
            bucket = by_city.get(p, [])
            n = min(len(bucket), args.per_city_cap)
            sampled.extend(rng.sample(bucket, n))
            log(f"  city {p}: {len(bucket):,} matched -> sampled {n:,}")
        chosen = sampled
        log(f"per-city cap {args.per_city_cap:,} -> {len(chosen):,} total "
            f"(seed={args.seed})")
    elif args.cap and len(chosen) > args.cap:
        rng = random.Random(args.seed)
        chosen = rng.sample(chosen, args.cap)
        log(f"global cap {args.cap:,} (seed={args.seed})")
    return chosen


INTERIOR_CHANNEL = 6
INTERIOR_NAME = "interior"


def _dilate6(x):
    """One step of 6-CONNECTED dilation (face neighbours + self).

    NOT max_pool3d(x, 3): a 3x3x3 kernel is 26-connected and lets the
    flood leak diagonally through a shell that is watertight under
    6-connectivity. That mismatch silently reported 27 of 40 real
    buildings as open when it was tested against the reference DFS.
    Three axis-aligned pools give exactly the 6-neighbourhood, matching
    the definition used by `evaluate.py`.
    """
    import torch
    import torch.nn.functional as F
    return torch.maximum(
        torch.maximum(F.max_pool3d(x, (3, 1, 1), 1, (1, 0, 0)),
                      F.max_pool3d(x, (1, 3, 1), 1, (0, 1, 0))),
        F.max_pool3d(x, (1, 1, 3), 1, (0, 0, 1)))


def _exterior_reachable_batch(shells, device=None):
    """Empty voxels reachable from outside, for a BATCH of buildings.

    A flood fill from the grid boundary, expressed as repeated
    6-connected dilation intersected with free space. Batching is what
    makes this viable at dataset scale: one pooling call floods every
    building in the batch at once, whereas the reference DFS in
    evaluate.py walks voxels in Python and costs ~0.24 s per building.

    Pads by one so the flood has somewhere to start even when a building
    touches the grid boundary.
    """
    import torch
    import torch.nn.functional as F

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    shells = np.ascontiguousarray(shells)
    free = torch.from_numpy(~shells).to(device).float()[:, None]
    free = F.pad(free, (1, 1, 1, 1, 1, 1), value=1.0)
    reach = torch.zeros_like(free)
    reach[:, :, 0, 0, 0] = 1.0                    # padded corner is outside
    for _ in range(3 * shells.shape[-1] + 8):     # cap; normally breaks early
        nxt = _dilate6(reach) * free
        if torch.equal(nxt, reach):
            break
        reach = nxt
    return reach[:, 0, 1:-1, 1:-1, 1:-1].cpu().numpy() > 0.5


def _exterior_reachable(shell, device=None):
    """Single-building convenience wrapper around the batch version."""
    return _exterior_reachable_batch(shell[None], device)[0]


def add_interior_channel(grid, close_first=True):
    """Append an `interior` channel: the volume the shell encloses.

    Rationale (claude.md §47): the v4 objective is a per-voxel
    cross-entropy with no term rewarding closure, so an open shell costs
    almost nothing. Labelling the enclosed volume converts closure from a
    topological property into a dense per-voxel target the existing loss
    can learn directly, and cuts the class imbalance as a side effect.

    Returns (grid7, status, n_interior) where status is one of:
      "filled"  -- enclosed directly
      "closed"  -- only after a 1-voxel morphological closing
      "open"    -- could not be enclosed; interior left empty

    About 8 % of real buildings are not watertight at 0.5 m. Those are
    labelled "open" and kept shell-only rather than silently dropped, so
    the count is auditable from the manifest afterwards.
    """
    import torch
    import torch.nn.functional as F

    T = grid.shape[1]
    shell = grid[1:].sum(axis=0) > 0
    interior = ~shell & ~_exterior_reachable(shell)
    status = "filled"

    if interior.sum() == 0 and close_first:
        # Seal one-voxel gaps, fill against the sealed shell, then keep
        # only voxels that are empty in the ORIGINAL grid -- the closing
        # is a probe for what the shell was trying to enclose, it must
        # not add material to the output.
        x = torch.from_numpy(shell).float()[None, None]
        x = F.max_pool3d(x, 3, stride=1, padding=1)
        x = -F.max_pool3d(-x, 3, stride=1, padding=1)
        closed = x[0, 0].numpy() > 0.5
        cand = ~closed & ~_exterior_reachable(closed)
        interior = cand & ~shell
        status = "closed" if interior.sum() else "open"

    out = np.zeros((INTERIOR_CHANNEL + 1, T, T, T), dtype=np.uint8)
    out[:INTERIOR_CHANNEL] = grid[:INTERIOR_CHANNEL]
    out[INTERIOR_CHANNEL] = interior.astype(np.uint8)
    out[0] = (out[1:].sum(axis=0) == 0).astype(np.uint8)   # recompute `empty`
    return out, status, int(interior.sum())


def build_tensor(coords, classes, target, vs):
    """coords (N,3) float UTM, classes (N,) int -> (C,T,T,T) uint8 ONE-HOT.

    Fine surface points collapse into 0.5 m cells; a boundary cell (e.g. a
    wall-meets-roof corner) can receive points of >1 class (~8% of occupied
    cells). We resolve each cell to a SINGLE class by majority vote -- the
    class with the most points in that cell wins; ties go to the lower class
    id -- so the output is strictly one-hot, as the diffusion model expects.
    """
    mn = coords.min(axis=0)
    local = np.floor((coords - mn) / vs).astype(np.int64)   # 0 .. cells-1
    extent = local.max(axis=0) + 1
    if int(extent.max()) > target:
        return None, extent                                 # safety (shouldn't hit)
    offset = (target - extent) // 2
    idx = local + offset
    ch = np.array([SURFACE_TO_CHANNEL.get(int(c), 0) for c in classes])
    keep = ch > 0                                           # drop Unknown(0)
    idx, ch = idx[keep], ch[keep]

    grid = np.zeros((N_CHANNELS, target, target, target), dtype=np.uint8)
    if len(ch):
        # majority vote per cell: count (cell, class) pairs, keep argmax class
        flat = (idx[:, 0] * target + idx[:, 1]) * target + idx[:, 2]
        key = flat * N_CHANNELS + ch                        # encode cell + class
        uk, cnt = np.unique(key, return_counts=True)        # uk sorted asc
        ucell, ucls = uk // N_CHANNELS, uk % N_CHANNELS
        order = np.argsort(-cnt, kind="stable")             # high count first;
        ucell_s, ucls_s = ucell[order], ucls[order]         # ties keep low class
        _, first = np.unique(ucell_s, return_index=True)    # winner per cell
        wcell, wcls = ucell_s[first], ucls_s[first]
        wz = wcell % target
        wy = (wcell // target) % target
        wx = wcell // (target * target)
        grid[wcls, wx, wy, wz] = 1
    grid[0] = (grid[1:].sum(axis=0) == 0).astype(np.uint8)  # empty channel
    return grid, extent


def collect_chunk_voxels(voxel_csv, want):
    """Stream one voxel CSV once; return {gmlid: (coords list, class list)}.

    Manual split is safe: all 8 columns are comma-free
    (voxel_position,building_gmlid,surface_gmlid,surface_class,x,y,z,vox_geom).
    """
    xs = defaultdict(list)
    cs = defaultdict(list)
    with open(voxel_csv) as f:
        f.readline()  # header
        for line in f:
            p = line.split(",", 7)
            g = p[1]
            if g in want:
                xs[g].append((float(p[4]), float(p[5]), float(p[6])))
                cs[g].append(int(p[3]))
    return xs, cs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Default points at the Zensus-enriched CSV so the construction-period
    # condition is wired in by default (§23 B1, §20.4). Pass the older
    # `tensorbuilding/building_metadata_clean.csv` to get unenriched shards.
    ap.add_argument("--meta", default=os.path.expanduser(
        "~/Documents/github/GebauJahr/prepared_outputs/"
        "building_metadata_clean_with_estimated_baujahr_reliability.csv"))
    ap.add_argument("--voxel-dir", default="voxel_csvs")
    ap.add_argument("--out-dir", default="tensorbuilding/shards")
    ap.add_argument("--target", type=int, default=64, help="grid edge length")
    ap.add_argument("--voxel-size", type=float, default=0.5)
    ap.add_argument("--shard-size", type=int, default=1000, help="buildings/shard")
    ap.add_argument("--cap", type=int, default=0,
                    help="global cap; 0 = no cap (use --per-city-cap or take all)")
    ap.add_argument("--per-city-cap", type=int, default=0,
                    help="per-city cap for balanced sampling (requires --cities)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--function", default="", help="'' = no filter (default)")
    ap.add_argument("--roof", default="", help="'' = no filter (default)")
    ap.add_argument("--max-height", type=float, default=16.0, help="0 to disable")
    ap.add_argument("--cities", default="",
                    help="comma-separated AGS prefixes or city names "
                         "(e.g. 'munich,augsburg,nuernberg,wuerzburg,regensburg' "
                         "or '09162,09761,09564,09663,09362'). Empty = no city filter.")
    ap.add_argument("--tag", default="dataset",
                    help="shard filename prefix (e.g. 5cities_balanced)")
    ap.add_argument("--interior", action="store_true",
                    help="add a 7th `interior` channel by flood-filling the "
                         "enclosed volume (v5 target; see claude.md §47)")
    args = ap.parse_args()

    # Resolve --cities to an ordered list of AGS prefixes (preserves user order).
    prefixes = []
    for tok in (args.cities or "").split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        prefixes.append(CITY_PREFIXES.get(tok, tok))
    if prefixes:
        log(f"city filter: {len(prefixes)} prefix(es) -> {prefixes}")
    if args.per_city_cap and not prefixes:
        sys.exit("--per-city-cap requires --cities")

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    fit_chunk = scan_grid_sizes(args.voxel_dir, args.target)
    chosen = select_buildings(args.meta, set(fit_chunk), args, prefixes)
    if not chosen:
        sys.exit("No buildings selected — relax the filters.")

    # group chosen gmlids by chunk so we stream each voxel CSV at most once
    by_chunk = defaultdict(dict)        # chunk_id -> {gmlid: meta_row}
    for row in chosen:
        by_chunk[fit_chunk[row["gmlid"]]][row["gmlid"]] = row
    log(f"chosen buildings span {len(by_chunk)} chunks")
    if args.interior:
        log("interior channel ENABLED -> 7-channel v5 targets")

    manifest = []
    interior_stats = Counter()
    skipped_unknown = 0
    written = 0
    shard_idx = 0
    shard = None
    shard_count = 0

    def open_shard():
        nonlocal shard, shard_idx, shard_count
        path = os.path.join(args.out_dir, f"{args.tag}_{shard_idx:04d}.h5")
        shard = h5py.File(path, "w")
        shard.attrs["voxel_size"] = args.voxel_size
        shard.attrs["target"] = args.target
        shard.attrs["srid"] = 25832
        shard.attrs["channel_names"] = ",".join(
            CHANNEL_NAMES + ([INTERIOR_NAME] if args.interior else []))
        shard_count = 0
        return path

    cur_path = open_shard()
    log(f"shard -> {cur_path}")

    for chunk_id in sorted(by_chunk):
        want = by_chunk[chunk_id]
        vcsv = os.path.join(args.voxel_dir, f"{chunk_id}_voxels.csv")
        if not os.path.exists(vcsv):
            log(f"  WARN {vcsv} missing — skipping {len(want)} buildings")
            continue
        ts = time.time()
        xs, cs = collect_chunk_voxels(vcsv, set(want))
        for g, meta in want.items():
            if g not in xs:
                continue
            coords = np.asarray(xs[g], dtype=np.float64)
            classes = np.asarray(cs[g], dtype=np.int64)
            su = int((classes == 0).sum())
            if su:
                skipped_unknown += su
            grid, extent = build_tensor(coords, classes, args.target, args.voxel_size)
            if grid is None:
                log(f"  WARN {g} extent {extent.tolist()} > {args.target}, skipped")
                continue

            int_status, int_vox = "", 0
            if args.interior:
                grid, int_status, int_vox = add_interior_channel(grid)
                interior_stats[int_status] += 1

            if shard_count >= args.shard_size:
                shard.close()
                shard_idx += 1
                cur_path = open_shard()
                log(f"shard -> {cur_path}")

            grp = shard.create_group(f"buildings/{g}")
            grp.create_dataset("tensor", data=grid, dtype="uint8",
                               compression="gzip", compression_opts=4,
                               chunks=(grid.shape[0], 16, 16, 16))
            cont = np.array([_f(meta.get(k)) for k in META_CONTINUOUS], dtype=np.float32)
            grp.create_dataset("conditions_continuous", data=cont)
            for k in META_CATEGORICAL:
                grp.attrs[k] = meta.get(k, "") or ""
            grp.attrs["gmlid"] = g
            grp.attrs["chunk_id"] = chunk_id
            grp.attrs["centrepoint"] = meta.get("centrepoint", "") or ""
            grp.attrs["n_voxels"] = int(coords.shape[0])
            grp.attrs["extent"] = extent.astype(np.int32)
            occ = [int(grid[c].sum()) for c in range(grid.shape[0])]
            grp.attrs["occupied_per_channel"] = np.array(occ, dtype=np.int32)

            # Manifest row. Column ORDER here defines the CSV column order
            # (DictWriter takes fieldnames from the first row's keys), and is
            # kept identical to the manifest that trained Phase B v4 so the
            # two datasets stay directly comparable.
            #
            # IMPORTANT (fixed 2026-07-27): this dict previously omitted
            # length_to_width_ratio, storeys_above_ground, ratio_cluster,
            # kreis and gemeinde. Because model/src/dataset.py reads its
            # conditioning fields from the MANIFEST (not from the HDF5
            # attrs), that silently zeroed three of the ten conditioning
            # fields — ratio_cluster collapsed to the null token alone, and
            # both continuous geometry fields became 0. The v4 dataset only
            # escaped this because a one-off post-hoc patch added the
            # columns afterwards (see manifest.csv.bak_pre_p1). Every field
            # the model consumes is now written here directly, so the
            # pipeline is correct by construction and needs no patch step.
            manifest.append({
                "gmlid": g, "chunk_id": chunk_id, "shard": os.path.basename(cur_path),
                "interior_status": int_status, "interior_voxels": int_vox,
                "function_label": meta.get("function_label", ""),
                "roof_type_label": meta.get("roof_type_label", ""),
                # -- continuous conditioning fields (dataset.py CONTINUOUS_COLS)
                "measured_height": meta.get("measured_height", ""),
                "length_to_width_ratio": meta.get("length_to_width_ratio", ""),
                "storeys_above_ground": meta.get("storeys_above_ground", ""),
                "constructionPeriodConfidence": meta.get("constructionPeriodConfidence", ""),
                # -- categorical conditioning fields (dataset.py CATEGORICAL_COLS)
                "height_cluster": meta.get("height_cluster", ""),
                "ratio_cluster": meta.get("ratio_cluster", ""),
                # -- administrative provenance (not model inputs, but needed
                #    for per-city stratification and hold-out experiments)
                "gemeindeschluessel": meta.get("gemeindeschluessel", ""),
                "kreis": meta.get("kreis", ""),
                "gemeinde": meta.get("gemeinde", ""),
                # -- enrichment columns (empty if --meta is the unenriched CSV)
                "estimatedConstructionPeriod": meta.get("estimatedConstructionPeriod", ""),
                "constructionPeriodReliability": meta.get("constructionPeriodReliability", ""),
                "storeys_source": meta.get("storeys_source", ""),
                "w": int(extent[0]), "d": int(extent[1]), "h": int(extent[2]),
                "n_voxels": int(coords.shape[0]),
                "occ_wall": occ[1], "occ_roof": occ[2], "occ_ground": occ[3],
            })
            written += 1
            shard_count += 1
        log(f"  {chunk_id}: {len(xs)}/{len(want)} buildings, "
            f"{time.time()-ts:.1f}s  (total written {written})")

    shard.close()

    mpath = os.path.join(args.out_dir, "manifest.csv")
    with open(mpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)

    # ---- manifest contract check ------------------------------------
    # model/src/dataset.py reads its conditioning fields from THIS file,
    # not from the HDF5 attrs. A field that is missing, or present but
    # universally empty, silently degrades to the null token (categorical)
    # or 0.0 (continuous) and the model trains on a crippled condition
    # vector while every log line still looks healthy. That is exactly how
    # ratio_cluster / length_to_width_ratio / storeys_above_ground were
    # lost before 2026-07-27, so the failure mode is checked here rather
    # than left to be discovered during training.
    _REQUIRED_CATEGORICAL = [
        "function_label", "roof_type_label", "height_cluster", "ratio_cluster",
        "estimatedConstructionPeriod", "constructionPeriodReliability",
    ]
    _REQUIRED_CONTINUOUS = [
        "measured_height", "length_to_width_ratio",
        "constructionPeriodConfidence", "storeys_above_ground",
    ]
    cols = set(manifest[0].keys())
    problems = []
    for k in _REQUIRED_CATEGORICAL + _REQUIRED_CONTINUOUS:
        if k not in cols:
            problems.append(f"MISSING column '{k}'")
            continue
        n_set = sum(1 for r in manifest if str(r.get(k, "")).strip() not in ("", "nan"))
        if n_set == 0:
            problems.append(f"column '{k}' present but EMPTY in all {len(manifest):,} rows")
        elif n_set < len(manifest) * 0.5:
            log(f"  NOTE: '{k}' populated in only {n_set:,}/{len(manifest):,} rows "
                f"({100*n_set/len(manifest):.1f}%)")

    if problems:
        log("")
        log("!! MANIFEST CONTRACT VIOLATION — the model would train on a")
        log("!! degraded conditioning vector. Fix before training:")
        for p in problems:
            log(f"!!   - {p}")
        log("")
        raise SystemExit(2)

    log(f"manifest contract OK: {len(_REQUIRED_CATEGORICAL)} categorical + "
        f"{len(_REQUIRED_CONTINUOUS)} continuous fields populated")

    log(f"DONE: {written:,} tensors in {shard_idx + 1} shard(s) -> {args.out_dir}")
    log(f"manifest -> {mpath}")
    if skipped_unknown:
        log(f"NOTE: skipped {skipped_unknown:,} Unknown(class 0) voxels")

    if args.interior:
        tot = sum(interior_stats.values()) or 1
        log("interior fill outcome:")
        for k in ("filled", "closed", "open"):
            n = interior_stats.get(k, 0)
            log(f"    {k:8s} {n:6,d}  ({100*n/tot:5.1f}%)")
        # An all-open run means the fill silently failed and every v5
        # target is a shell with no interior -- louder than a warning,
        # because training on it would waste ~17 GPU hours.
        if interior_stats.get("open", 0) > 0.5 * tot:
            log("ERROR: >50% of buildings could not be enclosed. The fill "
                "is probably broken (check 6- vs 26-connectivity). "
                "Refusing to leave this unflagged.")
            sys.exit(3)
    log(f"elapsed {time.time()-t0:.1f}s")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


if __name__ == "__main__":
    main()
