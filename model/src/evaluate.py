"""
Evaluation harness for the conditional voxel diffusion model.

Full specification and rationale:
    local comments/projectoverview/evaluation_plan.md
Decisions, supervisor constraints and provenance:
    claude.md §34 (plan), §35 (supervisor annotations), §36 (Phase C backlog)

This module is organised in stages. Stage 1 builds a cached sample set;
every later stage reads that cache, so metrics can be recomputed and
refined without touching the GPU again.

    stage 1   build-samples   -> eval_samples.npz (+ optional renders)
    stage 2   pillar A        -> distributional realism / diversity
    stage 3   pillars B, C    -> geometry, semantics
    stage 4   pillars E, F    -> topology, watertightness
    stage 5   pillar D        -> conditional validity + ablation

Usage:
    python -m model.src.evaluate build-samples \
        --ckpt model/checkpoints/phase_b_v4/ckpt_100000.pt \
        --out  model/checkpoints/phase_b_v4/eval \
        --n-per-condition 125

Design notes worth knowing before changing anything here:

*   **Real buildings are sampled too, and they are the benchmark.**
    Voxelising a thin shell at 0.5 m introduces artefacts (pin-holes at
    oblique junctions, fragmentation of thin features), so real
    buildings do NOT score perfectly on connectivity or watertightness.
    Every structural metric is therefore computed on both sets and
    reported side by side. Comparing generated output against a
    theoretical ideal instead of the measured real distribution would
    manufacture failures that are really voxelisation artefacts.

*   **Render parity is a correctness requirement.** For the human
    discrimination study, real and generated images must come from the
    same code path with identical camera, palette and resolution. If the
    two sets are rendered even slightly differently, participants learn
    to spot the renderer rather than the building — the study then
    measures the wrong thing while appearing to work.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .dataset import Building3DDataset
from .sample import (
    DEFAULT_CONDITIONS,
    load_checkpoint,
    sample as ddim_sample,
    to_hard_onehot,
)
from .real_vs_generated import _condition_match_score


# ---------------------------------------------------------------------
# stage 1 — sample set
# ---------------------------------------------------------------------
def _rank_real_by_condition(ds: Building3DDataset, cond: dict, k: int
                            ) -> list[tuple[int, float]]:
    """The k best-matching real buildings for one condition.

    Reuses `_condition_match_score` from real_vs_generated.py so that the
    matching rule is defined in exactly one place: a hard match on
    function_label + roof_type_label, then nearest on height / ratio /
    storeys. Returns [(dataset_index, score)] ascending by score.
    """
    scored = []
    for i, rec in enumerate(ds.records):
        s = _condition_match_score(rec["row"], cond)
        if math.isfinite(s):
            scored.append((i, s))
    scored.sort(key=lambda t: t[1])
    return scored[:k]


def build_sample_set(ckpt_path: str, out_dir: Path, n_per_cond: int,
                     n_steps: int, guidance: float, seed0: int,
                     shards_dir: str | None, manifest_path: str | None) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_checkpoint(ckpt_path, device)
    cfg = bundle["cfg"]
    param = str(cfg.get("parameterization", "eps"))
    print(f"[eval] checkpoint: {ckpt_path}")
    print(f"[eval] parameterization={param}  cfg_guidance={guidance}  ddim_steps={n_steps}")

    conds = list(DEFAULT_CONDITIONS)
    print(f"[eval] {len(conds)} conditions x {n_per_cond} samples "
          f"= {len(conds)*n_per_cond} generated")

    # -- generated ----------------------------------------------------
    # One distinct seed per sample, deterministic and reproducible. Seeds
    # are offset per condition so no two conditions share a noise draw —
    # otherwise apparent cross-condition similarity could be an artefact
    # of shared noise rather than of the model.
    gen_t, gen_meta = [], []
    t0 = time.time()
    for ci, cond in enumerate(conds):
        seeds = [seed0 + ci * 100_000 + j for j in range(n_per_cond)]
        # chunk so GPU memory stays bounded regardless of n_per_cond
        for s in range(0, len(seeds), 16):
            chunk = seeds[s:s + 16]
            x, meta = ddim_sample(
                bundle["unet"], bundle["cond_enc"], bundle["schedule"],
                conditions=[cond], vocabs=bundle["vocabs"],
                continuous_stats=bundle["continuous_stats"],
                seeds=chunk, n_steps=n_steps, guidance_scale=guidance,
                device=device, parameterization=param,
            )
            gen_t.append(to_hard_onehot(x).cpu().numpy().astype(np.uint8))
            gen_meta.extend(meta)
        print(f"[eval]   generated {cond['name']:36s} "
              f"{n_per_cond:4d}  ({time.time()-t0:.0f}s elapsed)", flush=True)
    gen = np.concatenate(gen_t, axis=0)
    print(f"[eval] generated array: {gen.shape}  {gen.nbytes/1e6:.0f} MB")

    # -- real ---------------------------------------------------------
    ds = Building3DDataset(
        shards_dir=shards_dir or cfg["shards_dir"],
        manifest_path=manifest_path or cfg.get("manifest_path"),
    )
    print(f"[eval] real dataset: {len(ds):,} buildings")

    real_t, real_meta = [], []
    used: set[int] = set()
    for cond in conds:
        ranked = _rank_real_by_condition(ds, cond, k=n_per_cond * 4)
        picked = 0
        for idx, score in ranked:
            if picked >= n_per_cond:
                break
            if idx in used:          # never reuse a real building across
                continue             # conditions — it would correlate the sets
            used.add(idx)
            rec = ds.records[idx]
            item = ds[idx]
            t = item["tensor"]
            t = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
            real_t.append(t.astype(np.uint8))
            real_meta.append({
                "name": cond["name"], "gmlid": rec["row"].get("gmlid", ""),
                "match_score": float(score),
                "function_label": rec["row"].get("function_label", ""),
                "roof_type_label": rec["row"].get("roof_type_label", ""),
                "measured_height": rec["row"].get("measured_height", ""),
                "storeys_above_ground": rec["row"].get("storeys_above_ground", ""),
                "kreis": rec["row"].get("kreis", ""),
            })
            picked += 1
        print(f"[eval]   matched   {cond['name']:36s} {picked:4d} real "
              f"(best score {ranked[0][1]:.3f})" if ranked else "", flush=True)
    real = np.stack(real_t, axis=0)
    print(f"[eval] real array:      {real.shape}  {real.nbytes/1e6:.0f} MB")

    npz = out_dir / "eval_samples.npz"
    np.savez_compressed(
        npz,
        gen=gen, real=real,
        gen_meta=json.dumps(gen_meta, default=str),
        real_meta=json.dumps(real_meta, default=str),
        provenance=json.dumps({
            "ckpt": str(ckpt_path),
            "parameterization": param,
            "ddim_steps": n_steps,
            "guidance": guidance,
            "n_per_condition": n_per_cond,
            "conditions": [c["name"] for c in conds],
            "seed0": seed0,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, default=str),
    )
    print(f"[eval] wrote {npz}  ({npz.stat().st_size/1e6:.1f} MB)")

    # Sanity: occupancy of both sets, as an immediate smoke test that the
    # cache is usable. A generated occupancy near 75 % would mean a
    # noise-producing checkpoint (the v2/v3 failure signature).
    for nm, arr in (("generated", gen), ("real", real)):
        occ = 100.0 * (arr[:, 1:6].sum(axis=(1, 2, 3, 4)) / (64 ** 3))
        print(f"[eval] {nm:9s} shell occupancy: mean={occ.mean():.3f}%  "
              f"median={np.median(occ):.3f}%  [{occ.min():.3f}, {occ.max():.3f}]")
    return npz


# ---------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------
CLASS_NAMES = ["empty", "wall", "roof", "ground", "outer_ceiling", "closure"]


def load_cache(npz_path: Path) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    try:
        _set_model_label(json.loads(str(z["provenance"]))["ckpt"])
    except Exception:
        pass
    return {
        "gen": z["gen"], "real": z["real"],
        "gen_meta": json.loads(str(z["gen_meta"])),
        "real_meta": json.loads(str(z["real_meta"])),
        "provenance": json.loads(str(z["provenance"])),
    }


N_SHELL_CHANNELS = 6      # empty + the five CityGML boundary-surface classes


def _labels(arr: np.ndarray) -> np.ndarray:
    """(N,C,D,D,D) one-hot -> (N,D,D,D) int8 class index over the SHELL.

    Only the first six channels are used. v4 targets have exactly those
    six; v5 adds a seventh `interior` channel (claude.md §47), and every
    metric in this module is defined on the boundary surfaces.

    Two reasons this must drop the interior rather than include it:

    1. **Comparability.** A seven-class marginal is not comparable with a
       six-class one, and v5 would appear to win on occupancy purely by
       having an extra class to fill.
    2. **Correctness.** Topology treats `label > 0` as solid. Counting
       interior voxels as solid would leave no enclosed empty space at
       all, so watertightness -- the metric the whole v5 intervention
       exists to improve -- would read as 0 % for every sample.

    Because the input is hard one-hot, an interior voxel is zero across
    all six shell channels, so `argmax` resolves it to 0 (`empty`) and
    the shell is recovered exactly as v4 would have produced it.
    """
    return arr[:, :N_SHELL_CHANNELS].argmax(axis=1).astype(np.int8)


def interior_metrics(arr: np.ndarray) -> dict:
    """v5-only: is the predicted `interior` class placed consistently?

    The model is asked to emit a 7th class marking the volume its shell
    encloses. That gives a consistency test unavailable for v4: flood
    the shell independently, and compare the region the shell ACTUALLY
    encloses against the region the model CLAIMED to enclose.

    Returns per-sample arrays:
      claimed   -- voxels the model labelled `interior`
      actual    -- voxels genuinely enclosed by the emitted shell
      iou       -- agreement between the two
      leaked    -- claimed interior that is not in fact enclosed
    """
    if arr.shape[1] <= N_SHELL_CHANNELS:
        return {}
    lab = _labels(arr)
    claimed_m = arr[:, N_SHELL_CHANNELS] > 0
    out = {k: [] for k in ("claimed", "actual", "iou", "leaked")}
    for i in range(arr.shape[0]):
        shell = lab[i] > 0
        actual_m = ~shell & ~_exterior_mask(shell)
        c, a = claimed_m[i], actual_m
        inter = int((c & a).sum())
        union = int((c | a).sum())
        out["claimed"].append(int(c.sum()))
        out["actual"].append(int(a.sum()))
        out["iou"].append(inter / union if union else np.nan)
        out["leaked"].append(int((c & ~a).sum()))
    return {k: np.array(v, float) for k, v in out.items()}


def _class_fractions(lab: np.ndarray) -> np.ndarray:
    """Aggregate voxel share per class over a whole set. Returns (6,)."""
    tot = lab.size
    return np.array([(lab == c).sum() / tot for c in range(6)], dtype=np.float64)


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits. 0 = identical distributions.

    Symmetric and bounded (unlike KL), which is why it is preferred here:
    an unbounded score would be dominated by whichever class one side
    happens to lack entirely — and two of our six classes are always
    empty, so that case is guaranteed to arise.
    """
    eps = 1e-12
    p = np.clip(p, eps, None); p = p / p.sum()
    q = np.clip(q, eps, None); q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float((a * np.log2(a / b)).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _occupancy(arr: np.ndarray) -> np.ndarray:
    """Per-building SHELL foreground fraction (%).

    Channels 1..5 only. A v5 tensor carries a 6th index holding the
    `interior` class; including it would report roughly three times the
    occupancy and make v5 incomparable with v4 -- the model would appear
    to have inflated its output when it merely gained a class to fill.
    Slicing to `N_SHELL_CHANNELS` is a no-op on 6-channel v4 tensors.
    """
    return (100.0 * arr[:, 1:N_SHELL_CHANNELS].sum(axis=(1, 2, 3, 4))
            / (64 ** 3))


def _wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """1-D Wasserstein (earth-mover) distance between two samples.

    Implemented directly rather than via scipy so the module keeps its
    dependency surface to numpy/torch/h5py (see DDML-Docker requirements).
    For 1-D it is the mean absolute difference of the sorted quantile
    functions, which is exact.
    """
    qs = np.linspace(0, 1, 1001)
    return float(np.abs(np.quantile(a, qs) - np.quantile(b, qs)).mean())


def _hist_overlap(a: np.ndarray, b: np.ndarray, bins: int = 50) -> float:
    """Histogram intersection in [0,1]; 1 = identical, 0 = disjoint."""
    lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    ha = ha / max(ha.sum(), 1); hb = hb / max(hb.sum(), 1)
    return float(np.minimum(ha, hb).sum())


# ---------------------------------------------------------------------
# semantic IoU distance  (the §5 decision: raw space, no learned embedding)
# ---------------------------------------------------------------------
def semantic_iou_matrix(A: np.ndarray, B: np.ndarray, device: torch.device,
                        classes=(1, 2, 3), chunk: int = 32) -> np.ndarray:
    """Pairwise distance matrix 1 - mean per-class IoU, shape (len(A), len(B)).

    This operationalises Heeramaglore & Kolbe (2022), who propose comparing
    two voxelised models through "set intersection ... based on their
    geometry and semantics". Per-class IoU is exactly that intersection,
    normalised.

    Only the populated foreground classes are used (wall/roof/ground);
    `empty` is excluded because at 99.35 % background every pair would
    score ~1.0 on it and the metric would be swamped, and channels 4/5
    are identically zero in this dataset so their IoU is undefined (0/0).

    Computed on GPU in chunks: 500x494 pairs x 3 classes x 262144 voxels
    is ~194 G voxel-comparisons, which is impractical on CPU.
    """
    def packed(X):
        # (N, K, D^3) bool on device, one plane per evaluated class
        t = torch.from_numpy(np.stack([(X.argmax(axis=1) == c) for c in classes], axis=1))
        return t.reshape(t.shape[0], len(classes), -1).to(device)

    Ab, Bb = packed(A), packed(B)
    out = np.zeros((Ab.shape[0], Bb.shape[0]), dtype=np.float32)

    def _pass(c: int) -> None:
        for i in range(0, Ab.shape[0], c):
            a = Ab[i:i + c].unsqueeze(1).float()          # (ca,1,K,V)
            for j in range(0, Bb.shape[0], c):
                b = Bb[j:j + c].unsqueeze(0).float()      # (1,cb,K,V)
                inter = (a * b).sum(-1)
                union = (a + b - a * b).sum(-1)
                # A class absent from BOTH shapes is treated as perfect
                # agreement (IoU 1) rather than 0/0 -> nan.
                iou = torch.where(union > 0, inter / union.clamp(min=1e-9),
                                  torch.ones_like(union))
                out[i:i + c, j:j + c] = (1.0 - iou.mean(-1)).cpu().numpy()

    # The GPU is often shared with a training run, so a chunk size that
    # worked yesterday can OOM today. Halve and retry rather than failing
    # an hour of analysis on a transient memory condition.
    while True:
        try:
            _pass(chunk)
            return out
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if chunk <= 1:
                raise
            chunk = max(1, chunk // 2)
            print(f"[eval]   (CUDA OOM -- retrying at chunk={chunk})", flush=True)


def cov_mmd_1nna(d_rg: np.ndarray, d_rr: np.ndarray, d_gg: np.ndarray) -> dict:
    """Achlioptas-style set metrics from precomputed distance matrices.

    d_rg: real x generated,  d_rr: real x real,  d_gg: generated x generated
    (the two square matrices must have +inf on the diagonal so a sample is
    never its own nearest neighbour).

    MMD    - for each REAL, distance to nearest GENERATED, averaged. Low
             is good: measures fidelity. Blind to mode collapse.
    COV    - fraction of REALs that are the nearest real to >=1 generated.
             High is good: measures diversity. Catches mode collapse.
    1-NNA  - pool both sets; for each sample ask whether its nearest
             neighbour came from its own set. **50 % is the ideal**: the
             two sets are then indistinguishable to a 1-NN classifier.
             ->100 % under both poor fidelity AND mode collapse, which is
             why it is the headline number.
    """
    mmd = float(d_rg.min(axis=1).mean())
    cov = float(len(np.unique(d_rg.argmin(axis=0))) / d_rg.shape[0])

    nr, ng = d_rr.shape[0], d_gg.shape[0]
    # nearest neighbour for each real: own set (d_rr) vs other set (d_rg)
    real_same = d_rr.min(axis=1) < d_rg.min(axis=1)
    # nearest neighbour for each generated: own set (d_gg) vs other (d_rg.T)
    gen_same = d_gg.min(axis=1) < d_rg.T.min(axis=1)
    nna = float((real_same.sum() + gen_same.sum()) / (nr + ng))
    return {"MMD": mmd, "COV": cov, "1-NNA": nna,
            "1-NNA_ideal": 0.5, "n_real": nr, "n_gen": ng}


# ---------------------------------------------------------------------
# per-building structural metrics  (Pillars B and C)
# ---------------------------------------------------------------------
# Implemented in pure numpy. scipy/skimage are deliberately not used: the
# DDML-Docker image ships numpy/h5py/PyYAML/matplotlib only, and adding a
# dependency here would mean the evaluation could not run in the same
# container as training.

def _same_class_neighbours(mask: np.ndarray) -> np.ndarray:
    """Count of 6-connected neighbours that share the mask, per voxel."""
    n = np.zeros(mask.shape, dtype=np.int8)
    n[1:, :, :] += mask[:-1, :, :]; n[:-1, :, :] += mask[1:, :, :]
    n[:, 1:, :] += mask[:, :-1, :]; n[:, :-1, :] += mask[:, 1:, :]
    n[:, :, 1:] += mask[:, :, :-1]; n[:, :, :-1] += mask[:, :, 1:]
    return n


def _axis_neighbours(mask: np.ndarray) -> tuple[int, int, int]:
    """Total same-class adjacencies along x, y, z separately."""
    ax = int((mask[1:, :, :] & mask[:-1, :, :]).sum())
    ay = int((mask[:, 1:, :] & mask[:, :-1, :]).sum())
    az = int((mask[:, :, 1:] & mask[:, :, :-1]).sum())
    return ax, ay, az


def structural_metrics(lab: np.ndarray) -> dict:
    """Pillar B + C metrics for one building. `lab` is (64,64,64) int class ids.

    Axis convention (build_tensors.py): axis 0 = easting, 1 = northing,
    2 = elevation. So "vertical" means axis 2.
    """
    fg = lab > 0
    out: dict = {"n_fg": int(fg.sum())}
    if out["n_fg"] < 8:
        return {**out, "degenerate": True}
    out["degenerate"] = False

    wall = lab == 1
    roof = lab == 2
    grnd = lab == 3
    zi = np.arange(lab.shape[2])

    # ---- B2 wall verticality -------------------------------------
    # Fraction of wall-to-wall adjacencies that run vertically. A planar
    # vertical wall contributes ~0.5 (equal z and in-plane horizontal
    # adjacency); a horizontal slab contributes 0. Higher = more upright.
    if wall.sum() > 1:
        ax, ay, az = _axis_neighbours(wall)
        tot = ax + ay + az
        out["B2_wall_verticality"] = float(az / tot) if tot else np.nan
    else:
        out["B2_wall_verticality"] = np.nan

    # ---- B3 ground horizontality ---------------------------------
    # Share of ground voxels sitting within +/-1 voxel of the modal ground
    # layer. A flat slab -> ~1.0; ground smeared up the elevation axis -> low.
    if grnd.sum() > 0:
        gz = zi[np.where(grnd.any(axis=(0, 1)))[0]] if grnd.any() else np.array([])
        zs = np.repeat(zi, grnd.sum(axis=(0, 1)))
        modal = int(np.bincount(zs, minlength=lab.shape[2]).argmax())
        out["B3_ground_horizontality"] = float(np.abs(zs - modal).__le__(1).mean())
        out["B3_ground_z_std"] = float(zs.std())
    else:
        out["B3_ground_horizontality"] = np.nan
        out["B3_ground_z_std"] = np.nan

    # ---- B4 surface smoothness (planarity proxy) -----------------
    # Mean same-class 6-neighbour count over foreground voxels. A voxel in
    # the interior of a plane has 4; an edge 3; an isolated speck 0. So
    # higher = smoother, more plane-like surfaces; low = noisy//fragmented.
    # This is a proxy for planarity that avoids segmenting surfaces and
    # fitting planes, which would need scipy.
    nb = _same_class_neighbours(fg)
    out["B4_mean_neighbours"] = float(nb[fg].mean())
    out["B4_isolated_frac"] = float((nb[fg] <= 1).mean())

    # ---- C1 vertical class ordering ------------------------------
    def _mean_z(m):
        c = m.sum(axis=(0, 1))
        return float((zi * c).sum() / c.sum()) if c.sum() else np.nan
    zg, zw, zr = _mean_z(grnd), _mean_z(wall), _mean_z(roof)
    out["C1_mean_z_ground"], out["C1_mean_z_wall"], out["C1_mean_z_roof"] = zg, zw, zr
    out["C1_order_ok"] = bool(
        np.isfinite([zg, zw, zr]).all() and zg < zw < zr)

    # ---- C2 roof exposed from above ------------------------------
    # For each roof voxel, is there any foreground strictly above it?
    # cumsum from the top down gives the count above each voxel in O(V).
    above = np.cumsum(fg[:, :, ::-1], axis=2)[:, :, ::-1] - fg
    out["C2_roof_exposed"] = float((roof & (above == 0)).sum() / roof.sum()) \
        if roof.sum() else np.nan

    # ---- C3 ground at the base -----------------------------------
    zmin = int(np.where(fg.any(axis=(0, 1)))[0].min())
    if grnd.sum() > 0:
        zs = np.repeat(zi, grnd.sum(axis=(0, 1)))
        out["C3_ground_at_base"] = float((zs <= zmin + 2).mean())
    else:
        out["C3_ground_at_base"] = np.nan

    # ---- B5 fractal dimension (box counting) ---------------------
    # Cropped to the bounding box and with box sizes capped below the
    # object's smallest dimension. Both are essential: on the padded 64^3
    # grid the count saturates once boxes exceed the building and the
    # fitted slope collapses to ~1.3 instead of ~2.14 (claude.md, B5 note).
    idx = np.argwhere(fg)
    idx = idx - idx.min(0)
    ext = idx.max(0) + 1
    smax = max(2, int(min(ext) // 2))
    sizes = [s for s in (1, 2, 3, 4, 6, 8, 12, 16) if s <= smax]
    if len(sizes) >= 4:
        N = np.array([len(np.unique(idx // s, axis=0)) for s in sizes], float)
        x, y = np.log(1.0 / np.array(sizes, float)), np.log(N)
        slope, icpt = np.polyfit(x, y, 1)
        ssr = ((y - (slope * x + icpt)) ** 2).sum()
        sst = ((y - y.mean()) ** 2).sum()
        out["B5_fractal_dim"] = float(slope)
        out["B5_fractal_r2"] = float(1 - ssr / sst) if sst > 0 else np.nan
        out["B5_n_scales"] = len(sizes)
    else:
        # Too small to fit: ~56 % of real buildings fall here. Recorded as
        # NaN and counted, never silently dropped.
        out["B5_fractal_dim"] = np.nan
        out["B5_fractal_r2"] = np.nan
        out["B5_n_scales"] = len(sizes)
    return out


# ---------------------------------------------------------------------
# topology + watertightness  (Pillars E and F)
# ---------------------------------------------------------------------
# Connected-component labelling implemented here rather than imported from
# scipy.ndimage, to keep the evaluation runnable inside the DDML-Docker
# image. Iterative flood fill on a bool grid, using an explicit stack — a
# recursive version overflows Python's stack on a 64^3 volume.

def _flood_count(mask: np.ndarray, seeds: list[tuple[int, int, int]] | None = None
                 ) -> tuple[np.ndarray, int]:
    """6-connected component labelling. Returns (labels int32, n_components).

    labels are 1..n over True voxels; 0 elsewhere.
    """
    D = mask.shape
    lab = np.zeros(D, dtype=np.int32)
    cur = 0
    idxs = np.argwhere(mask)
    for p in idxs:
        p = tuple(int(v) for v in p)
        if lab[p]:
            continue
        cur += 1
        stack = [p]
        lab[p] = cur
        while stack:
            x, y, zz = stack.pop()
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                               (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                a, b, c = x + dx, y + dy, zz + dz
                if 0 <= a < D[0] and 0 <= b < D[1] and 0 <= c < D[2] \
                        and mask[a, b, c] and not lab[a, b, c]:
                    lab[a, b, c] = cur
                    stack.append((a, b, c))
    return lab, cur


def _exterior_mask(fg: np.ndarray) -> np.ndarray:
    """Empty voxels reachable from outside the grid, by flood fill.

    Pads by one voxel so the flood always has somewhere to start even when
    the building touches the grid boundary, then floods the empty space.
    Anything empty and NOT reached is enclosed interior.
    """
    p = np.pad(fg, 1, constant_values=False)
    free = ~p
    reach = np.zeros_like(free)
    stack = [(0, 0, 0)]
    reach[0, 0, 0] = True
    D = free.shape
    while stack:
        x, y, z = stack.pop()
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            a, b, c = x + dx, y + dy, z + dz
            if 0 <= a < D[0] and 0 <= b < D[1] and 0 <= c < D[2] \
                    and free[a, b, c] and not reach[a, b, c]:
                reach[a, b, c] = True
                stack.append((a, b, c))
    return reach[1:-1, 1:-1, 1:-1]


def topology_metrics(lab: np.ndarray) -> dict:
    """Pillars E and F for one building."""
    fg = lab > 0
    out: dict = {}
    if fg.sum() < 8:
        return {"degenerate": True}
    out["degenerate"] = False

    # ---- E1  b0: connected components of the shell ---------------
    _, ncomp = _flood_count(fg)
    out["E1_components"] = int(ncomp)
    out["E1_single_component"] = bool(ncomp == 1)

    # ---- F1-F3  watertightness via flood fill --------------------
    # Empty voxels not reachable from outside are enclosed interior. A
    # sealed shell encloses volume; a leaking one encloses none because
    # the flood pours straight in.
    ext = _exterior_mask(fg)
    interior = (~fg) & (~ext)
    n_int = int(interior.sum())
    out["F3_enclosed_volume"] = n_int
    out["F1_watertight"] = bool(n_int > 0)

    # ---- E3  b2: number of distinct enclosed cavities ------------
    if n_int:
        _, ncav = _flood_count(interior)
        out["E3_cavities"] = int(ncav)
    else:
        out["E3_cavities"] = 0

    # ---- F4  solidity: filled volume vs its bounding box ---------
    # True convex hull needs scipy; the axis-aligned bounding box is a
    # cheap, monotone stand-in. Near 1 = block-like and box-filling;
    # lower = articulated form with recesses. Comparative only.
    solid = fg | interior
    idx = np.argwhere(solid)
    ext_bb = idx.max(0) - idx.min(0) + 1
    bb = int(np.prod(ext_bb))
    out["F4_solidity_bbox"] = float(solid.sum() / bb) if bb else np.nan
    out["F_filled_volume"] = int(solid.sum())
    return out


# ---------------------------------------------------------------------
# figure + table export
# ---------------------------------------------------------------------
# Two-stage by design: metrics persist the *raw arrays* they were computed
# from into plotdata.npz, and figures are rendered from that. So a slide
# variant (bigger fonts, fewer elements) can be produced later without
# re-running any GPU work — only the render step is repeated.
#
# Style follows model/src/plot_training.py so Chapters 6 and 7 look like
# one document: dpi=140, alpha-0.25 grids, matplotlib tab10, reference
# lines dotted grey with a small annotation.

# Consistent across every evaluation figure.
C_REAL = "#1f77b4"     # blue  — real / reference
C_GEN = "#2ca02c"      # green — generated (matches v4's colour in plot_training)

# Figure legends must name the model actually being plotted. These were
# hardcoded to "v4", so every v5 and Phase C figure was mislabelled --
# invisible in the numbers and obvious in the PDF. Set from the
# checkpoint path recorded in the sample cache.
MODEL_LABEL = "generated"


def _set_model_label(ckpt_path: str) -> None:
    global MODEL_LABEL
    p = str(ckpt_path)
    if "phase_c" in p:
        MODEL_LABEL = "Phase C"
    elif "phase_b_v5" in p:
        MODEL_LABEL = "v5"
    elif "phase_b_v4" in p:
        MODEL_LABEL = "v4"
    else:
        MODEL_LABEL = "generated"


def _glabel(extra: str = "") -> str:
    """Legend text for the generated series, e.g. 'v5 generated (n=500)'."""
    return f"{MODEL_LABEL} generated{extra}"
C_REF = "grey"         # dotted reference lines


def _fig_style(ax, xlabel: str, ylabel: str, title: str | None = None):
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=9)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    # PDF alongside PNG: vector for the thesis, raster for quick viewing.
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.15)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"[eval] figure -> {path}  (+ .pdf)")


def _tex_table(path: Path, caption: str, label: str, header: list[str],
               rows: list[list[str]], note: str | None = None):
    """Emit a booktabs table for \\input{} into the thesis.

    Written to disk rather than transcribed by hand: manual copying of
    numbers into LaTeX is exactly how the Chapter 4 errors of 2026-07-27
    happened.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    col = "l" + "r" * (len(header) - 1)
    lines = [
        "% AUTO-GENERATED by model/src/evaluate.py — do not edit by hand.",
        "% Re-run the evaluation to regenerate.",
        "\\begin{table}[htbp]",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\centering",
        "  \\small",
        f"  \\begin{{tabular}}{{{col}}}",
        "    \\toprule",
        "    " + " & ".join(header) + " \\\\",
        "    \\midrule",
    ]
    lines += ["    " + " & ".join(r) + " \\\\" for r in rows]
    lines += ["    \\bottomrule", "  \\end{tabular}"]
    if note:
        lines.append(f"  \\par\\smallskip\\footnotesize {note}")
    lines += ["\\end{table}", ""]
    path.write_text("\n".join(lines))
    print(f"[eval] table  -> {path}")


def figures_pillarA(pd_path: Path, out_dir: Path) -> None:
    """Render Pillar A figures from persisted plot data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(pd_path, allow_pickle=True)
    figs, tabs = out_dir / "figures", out_dir / "tables"

    # --- F1  occupancy distributions --------------------------------
    og, orl = z["occ_gen"], z["occ_real"]
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    bins = np.linspace(0, max(og.max(), orl.max()), 45)
    ax.hist(orl, bins=bins, alpha=0.55, color=C_REAL, label=f"real, condition-matched (n={len(orl)})")
    ax.hist(og, bins=bins, alpha=0.55, color=C_GEN, label=_glabel(f" (n={len(og)})"))
    ax.axvline(orl.mean(), color=C_REAL, ls="--", lw=1.2)
    ax.axvline(og.mean(), color=C_GEN, ls="--", lw=1.2)
    # NOTE: matplotlib text is plain + mathtext ($...$) only. LaTeX commands
    # such as \% or \emph{} are NOT interpreted here and would render
    # literally — they belong exclusively in _tex_table output.
    ax.annotate(f"real mean {orl.mean():.2f}%", (orl.mean(), ax.get_ylim()[1]*0.92),
                fontsize=8, color=C_REAL, ha="right", rotation=90, va="top")
    ax.annotate(f"gen mean {og.mean():.2f}%", (og.mean(), ax.get_ylim()[1]*0.92),
                fontsize=8, color=C_GEN, ha="right", rotation=90, va="top")
    _fig_style(ax, "foreground occupancy per building (%)", "buildings")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, figs / "eval_occupancy.png")

    # --- F2  foreground class composition ---------------------------
    # Foreground only: including `empty` at ~99 % renders every other bar
    # invisible and hides the entire result.
    names = ["wall", "roof", "ground"]
    fg, fr = z["fg_gen"], z["fg_real"]
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=140)
    ax.bar(x - w/2, 100*fr, w, color=C_REAL, label="real, condition-matched")
    ax.bar(x + w/2, 100*fg, w, color=C_GEN, label=_glabel())
    for i in range(3):
        ax.text(x[i]-w/2, 100*fr[i], f"{100*fr[i]:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i]+w/2, 100*fg[i], f"{100*fg[i]:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    _fig_style(ax, "", "share of foreground voxels (%)")
    ax.legend(fontsize=9)
    _save(fig, figs / "eval_class_composition.png")

    # --- F3  nearest-neighbour distances (the C4 baseline argument) --
    # The single most important figure of Pillar A: it turns the bare
    # number "MMD = 0.78" into a visible comparison against how far apart
    # two *real* buildings already are.
    nn_rr, nn_rg = z["nn_real_real"], z["nn_real_gen"]
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    bins = np.linspace(0, 1, 45)
    ax.hist(nn_rr, bins=bins, alpha=0.55, color=C_REAL,
            label=f"real $\\rightarrow$ nearest real (mean {nn_rr.mean():.3f})")
    ax.hist(nn_rg, bins=bins, alpha=0.55, color=C_GEN,
            label=f"real $\\rightarrow$ nearest generated (mean {nn_rg.mean():.3f})")
    ax.axvline(nn_rr.mean(), color=C_REAL, ls="--", lw=1.2)
    ax.axvline(nn_rg.mean(), color=C_GEN, ls="--", lw=1.2)
    # The right-hand mass is the substantive result: real buildings with no
    # generated counterpart at all (claude.md §38.7).
    ax.axvline(0.90, color=C_REF, ls=":", lw=1.0)
    ax.annotate("no close match\n(d > 0.90)", (0.905, ax.get_ylim()[1]*0.75),
                fontsize=8, color=C_REF, ha="left")
    _fig_style(ax, "distance  1 - mean per-class IoU   (0 = identical)", "buildings")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, figs / "eval_nn_distance.png")

    # --- F4  coverage per condition ---------------------------------
    # The headline figure of Pillar A: it shows that the aggregate metrics
    # average over a bimodal population (claude.md §38.7).
    m = json.loads((out_dir / "metrics_pillarA.json").read_text())
    pc = m["A3_coverage_per_condition"]["conditions"]
    thr = m["A3_coverage_per_condition"]["threshold"]
    names = sorted(pc, key=lambda k: pc[k]["uncovered_by_generated"])
    ur = np.array([100*pc[n]["uncovered_by_real"] for n in names])
    ug = np.array([100*pc[n]["uncovered_by_generated"] for n in names])
    y = np.arange(len(names)); h = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=140)
    ax.barh(y + h/2, ur, h, color=C_REAL, label="nearest real building")
    ax.barh(y - h/2, ug, h, color=C_GEN, label="nearest generated building")
    for i in range(len(names)):
        ax.text(ug[i] + 1.2, y[i] - h/2, f"{ug[i]:.1f}%", va="center", fontsize=8, color=C_GEN)
        ax.text(ur[i] + 1.2, y[i] + h/2, f"{ur[i]:.1f}%", va="center", fontsize=8, color=C_REAL)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}\n(n = {pc[n]['n_real']})" for n in names], fontsize=8)
    ax.set_xlim(0, 100)
    _fig_style(ax, f"real buildings with no match closer than d = {thr}  (%)", "")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, figs / "eval_coverage_per_condition.png")

    # --- tables ------------------------------------------------------
    a1, a2, a3 = m["A1_class_marginal"], m["A2_occupancy"], m["A3_set_metrics"]

    _tex_table(
        tabs / "eval_A3_coverage.tex",
        caption="Coverage of the real distribution, per condition. A real building is "
                "counted as uncovered when no sample of the comparison set lies within "
                f"$d = {thr}$ (equivalently, mean per-class IoU below ${1-thr:.2f}$). "
                "Every real building has a close \\emph{real} neighbour; nearly a third "
                "have no close \\emph{generated} one, and the shortfall is concentrated "
                "almost entirely in the two non-residential conditions.",
        label="tab:eval-coverage",
        header=["Condition", "$n$", "Uncov.\\ by real", "Uncov.\\ by generated"],
        rows=[[n.replace("_", "\\_"), str(pc[n]["n_real"]),
               f"{100*pc[n]['uncovered_by_real']:.1f}\\%",
               f"\\textbf{{{100*pc[n]['uncovered_by_generated']:.1f}\\%}}"
               if pc[n]["uncovered_by_generated"] > 0.5
               else f"{100*pc[n]['uncovered_by_generated']:.1f}\\%"]
              for n in names],
        note="Aggregate set metrics average over this strongly bimodal population and "
             "therefore understate the failure on the data-poor rural archetype.",
    )

    _tex_table(
        tabs / "eval_A1_class_marginal.tex",
        caption="Class marginal of generated versus condition-matched real buildings. "
                "The Jensen--Shannon divergence over all six classes is dominated by the "
                "\\texttt{empty} class and is therefore reported alongside the "
                "foreground-only value, which is the informative figure.",
        label="tab:eval-class-marginal",
        header=["Class", "Generated", "Real (matched)", "$\\Delta$"],
        rows=[[c.replace("_", "\\_"),
               f"{100*a1['generated'][c]:.4f}\\%",
               f"{100*a1['real_condition_matched'][c]:.4f}\\%",
               f"{100*(a1['generated'][c]-a1['real_condition_matched'][c]):+.4f}"]
              for c in CLASS_NAMES],
        note=f"JSD (6-class) = {a1['JSD_bits']:.5f} bits; "
             f"JSD (foreground only) = {a1['JSD_foreground_bits']:.5f} bits.",
    )

    _tex_table(
        tabs / "eval_A2_A3_summary.tex",
        caption="Distributional realism and diversity. 1-NNA is the headline metric: "
                "$50\\%$ indicates that real and generated samples are indistinguishable "
                "to a nearest-neighbour classifier, and the score rises towards $100\\%$ "
                "under both poor fidelity and mode collapse.",
        label="tab:eval-set-metrics",
        header=["Metric", "Value", "Reference"],
        rows=[
            ["Occupancy, generated (mean)", f"{a2['generated']['mean']:.3f}\\%", "---"],
            ["Occupancy, real matched (mean)", f"{a2['real_condition_matched']['mean']:.3f}\\%", "---"],
            ["Occupancy ratio", f"{a2['ratio_of_means']:.2f}$\\times$", "$1.0\\times$"],
            ["Wasserstein-1 (occupancy)", f"{a2['wasserstein_pct']:.4f}\\,pp", "$0$"],
            ["Histogram overlap", f"{a2['histogram_overlap']:.3f}", "$1.0$"],
            ["MMD", f"{a3['MMD']:.4f}", f"{m['C4_iou_baseline']['real_real_nearest_mean']:.4f} (real--real)"],
            ["COV", f"{100*a3['COV']:.1f}\\%", "higher is better"],
            ["\\textbf{1-NNA}", f"\\textbf{{{100*a3['1-NNA']:.1f}\\%}}", "\\textbf{$50\\%$}"],
        ],
        note=f"$n = {a3['n_gen']}$ generated, $n = {a3['n_real']}$ condition-matched real. "
             f"Distance is $1-$ mean per-class IoU over \\{{wall, roof, ground\\}}.",
    )


# ---------------------------------------------------------------------
# stage 2 — Pillar A
# ---------------------------------------------------------------------
def pillar_A(cache: dict, out_dir: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen, real = cache["gen"], cache["real"]
    lg, lr = _labels(gen), _labels(real)
    res: dict = {}

    # -- A1 class marginal -------------------------------------------
    fg_, fr_ = _class_fractions(lg), _class_fractions(lr)
    # Foreground-only composition. The 6-class JSD is dominated by the
    # ~99 % `empty` class and is nearly blind to foreground differences —
    # reporting it alone would understate the discrepancy ~3x.
    fgf = fg_[1:4] / fg_[1:4].sum()
    frf = fr_[1:4] / fr_[1:4].sum()
    res["A1_class_marginal"] = {
        "generated": {CLASS_NAMES[i]: float(fg_[i]) for i in range(6)},
        "real_condition_matched": {CLASS_NAMES[i]: float(fr_[i]) for i in range(6)},
        "JSD_bits": _jsd(fg_, fr_),
        "JSD_foreground_bits": _jsd(fgf, frf),
        "foreground_composition": {
            "generated": {n: float(v) for n, v in zip(CLASS_NAMES[1:4], fgf)},
            "real_condition_matched": {n: float(v) for n, v in zip(CLASS_NAMES[1:4], frf)},
        },
        "wall_roof_ratio": {"generated": float(fg_[1] / fg_[2]),
                            "real_condition_matched": float(fr_[1] / fr_[2])},
    }
    print("\n=== A1  class marginal ===")
    print(f"  {'class':16s} {'generated':>11s} {'real(cm)':>11s} {'delta':>9s}")
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:16s} {100*fg_[i]:10.4f}% {100*fr_[i]:10.4f}% {100*(fg_[i]-fr_[i]):+8.4f}%")
    print(f"  JSD (6-class, empty-dominated) = {res['A1_class_marginal']['JSD_bits']:.5f} bits")
    print(f"  JSD (foreground only)          = {res['A1_class_marginal']['JSD_foreground_bits']:.5f} bits  <- informative")
    print(f"  wall:roof ratio  generated={fg_[1]/fg_[2]:.2f}  real={fr_[1]/fr_[2]:.2f}")

    # -- A2 occupancy -------------------------------------------------
    og, orl = _occupancy(gen), _occupancy(real)
    res["A2_occupancy"] = {
        "generated": {"mean": float(og.mean()), "median": float(np.median(og)),
                      "std": float(og.std()), "min": float(og.min()), "max": float(og.max())},
        "real_condition_matched": {"mean": float(orl.mean()), "median": float(np.median(orl)),
                                   "std": float(orl.std()), "min": float(orl.min()), "max": float(orl.max())},
        "wasserstein_pct": _wasserstein1(og, orl),
        "histogram_overlap": _hist_overlap(og, orl),
        "ratio_of_means": float(og.mean() / orl.mean()),
    }
    a2 = res["A2_occupancy"]
    print("\n=== A2  occupancy (% foreground per building) ===")
    print(f"  generated  mean={a2['generated']['mean']:.3f}%  median={a2['generated']['median']:.3f}%")
    print(f"  real (cm)  mean={a2['real_condition_matched']['mean']:.3f}%  "
          f"median={a2['real_condition_matched']['median']:.3f}%")
    print(f"  Wasserstein-1 = {a2['wasserstein_pct']:.4f} pp   "
          f"histogram overlap = {a2['histogram_overlap']:.3f}   "
          f"mean ratio = {a2['ratio_of_means']:.2f}x")

    # -- A3 COV / MMD / 1-NNA ----------------------------------------
    print("\n=== A3  set metrics (semantic IoU distance, wall/roof/ground) ===")
    t0 = time.time()
    d_rg = semantic_iou_matrix(real, gen, device)
    d_rr = semantic_iou_matrix(real, real, device); np.fill_diagonal(d_rr, np.inf)
    d_gg = semantic_iou_matrix(gen, gen, device);   np.fill_diagonal(d_gg, np.inf)
    m = cov_mmd_1nna(d_rg, d_rr, d_gg)
    res["A3_set_metrics"] = m
    print(f"  MMD   = {m['MMD']:.4f}   (lower = better fidelity)")
    print(f"  COV   = {100*m['COV']:.1f}%  (higher = better diversity)")
    print(f"  1-NNA = {100*m['1-NNA']:.1f}%  (50% = indistinguishable  <- headline)")
    print(f"  [{time.time()-t0:.0f}s]")

    # Real-vs-real IoU is the §6 baseline: it answers "how similar are two
    # DIFFERENT real buildings sharing a condition?" Without it, a low
    # generated-vs-real IoU cannot be interpreted, because two distinct
    # real buildings also score low.
    nn_rr, nn_rg = d_rr.min(axis=1), d_rg.min(axis=1)
    res["C4_iou_baseline"] = {
        "real_real_nearest_mean": float(nn_rr.mean()),
        "gen_real_nearest_mean": float(nn_rg.mean()),
        "note": "distance = 1 - mean per-class IoU; equal values mean generated "
                "samples sit as close to real buildings as real buildings do to each other",
    }
    print(f"\n=== C4  paired-IoU baseline (§6 decision) ===")
    print(f"  nearest real->real      distance = {nn_rr.mean():.4f}")
    print(f"  nearest real->generated distance = {nn_rg.mean():.4f}")

    # -- coverage, per condition -------------------------------------
    # Aggregate set metrics average over what turned out to be a strongly
    # bimodal population: two conditions are covered essentially
    # perfectly, one fails almost completely. A global MMD reads as
    # "somewhat worse" and hides that entirely, so coverage is reported
    # per condition by mandate (claude.md §38.7).
    real_cond = np.array([m_["name"] for m_ in cache["real_meta"]])
    THRESH = 0.90
    cov_rows, per_cond = [], {}
    for name in sorted(set(real_cond.tolist())):
        sel = real_cond == name
        unc_g = float((nn_rg[sel] > THRESH).mean())
        unc_r = float((nn_rr[sel] > THRESH).mean())
        per_cond[name] = {
            "n_real": int(sel.sum()),
            "uncovered_by_generated": unc_g,
            "uncovered_by_real": unc_r,
            "mean_nn_gen": float(nn_rg[sel].mean()),
            "mean_nn_real": float(nn_rr[sel].mean()),
        }
        cov_rows.append((name, int(sel.sum()), unc_r, unc_g))
    res["A3_coverage_per_condition"] = {"threshold": THRESH, "conditions": per_cond}

    print(f"\n=== A3b coverage per condition (real buildings with no match, d > {THRESH}) ===")
    print(f"  {'condition':38s} {'n':>4s} {'->real':>8s} {'->gen':>8s}")
    for name, n, ur, ug in cov_rows:
        flag = "  <-- FAILS" if ug > 0.5 else ""
        print(f"  {name:38s} {n:4d} {100*ur:7.1f}% {100*ug:7.1f}%{flag}")
    print(f"  {'OVERALL':38s} {len(real_cond):4d} "
          f"{100*(nn_rr>THRESH).mean():7.1f}% {100*(nn_rg>THRESH).mean():7.1f}%")

    (out_dir / "metrics_pillarA.json").write_text(json.dumps(res, indent=2))
    print(f"\n[eval] wrote {out_dir/'metrics_pillarA.json'}")

    # Persist the arrays the figures are drawn from, NOT just the summary
    # statistics. This is what lets a presentation-styled variant (larger
    # fonts, fewer elements) be rendered later without repeating any GPU
    # work — only the render step is re-run.
    pd_path = out_dir / "plotdata_pillarA.npz"
    np.savez_compressed(
        pd_path,
        occ_gen=og, occ_real=orl,
        fg_gen=fgf, fg_real=frf,
        class_frac_gen=fg_, class_frac_real=fr_,
        nn_real_real=nn_rr, nn_real_gen=nn_rg,
        nn_gen_gen=d_gg.min(axis=1), nn_gen_real=d_rg.T.min(axis=1),
        gen_cond=np.array([m_["name"] for m_ in cache["gen_meta"]]),
        real_cond=real_cond,
        cov_threshold=THRESH,
    )
    print(f"[eval] plot data -> {pd_path}  ({pd_path.stat().st_size/1e6:.1f} MB)")

    figures_pillarA(pd_path, out_dir)
    return res


# ---------------------------------------------------------------------
# stage 3 — Pillars B and C
# ---------------------------------------------------------------------
# Metrics reported per condition as well as in aggregate, by mandate:
# Pillar A showed that global figures average over a bimodal population
# and concealed a near-total failure on one archetype (claude.md §38.7).

METRICS_BC = [
    ("B2_wall_verticality",   "wall verticality",        "higher = more upright"),
    ("B3_ground_horizontality", "ground horizontality",  "higher = flatter"),
    ("B4_mean_neighbours",    "surface smoothness",      "higher = more planar"),
    ("B4_isolated_frac",      "isolated voxel fraction", "lower = less speckle"),
    ("B5_fractal_dim",        "fractal dimension",       "comparative only"),
    ("C2_roof_exposed",       "roof exposed from above", "higher = better"),
    ("C3_ground_at_base",     "ground at base",          "higher = better"),
]


def _agg(rows: list[dict], key: str) -> dict:
    v = np.array([r[key] for r in rows if not r.get("degenerate")
                  and np.isfinite(r.get(key, np.nan))], dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None}
    return {"n": int(v.size), "mean": float(v.mean()), "std": float(v.std()),
            "median": float(np.median(v))}


def pillar_BC(cache: dict, out_dir: Path) -> dict:
    gen, real = cache["gen"], cache["real"]
    gcond = np.array([m["name"] for m in cache["gen_meta"]])
    rcond = np.array([m["name"] for m in cache["real_meta"]])

    print("\n[eval] computing structural metrics ...")
    t0 = time.time()
    grows = [structural_metrics(l) for l in _labels(gen)]
    rrows = [structural_metrics(l) for l in _labels(real)]
    print(f"[eval]   {len(grows)+len(rrows)} buildings in {time.time()-t0:.0f}s")

    res: dict = {"aggregate": {}, "per_condition": {}}

    # -- aggregate ----------------------------------------------------
    print("\n=== Pillars B + C  (generated vs condition-matched real) ===")
    print(f"  {'metric':26s} {'generated':>18s} {'real (matched)':>18s}")
    for key, lbl, hint in METRICS_BC:
        g, r = _agg(grows, key), _agg(rrows, key)
        res["aggregate"][key] = {"generated": g, "real": r, "hint": hint}
        gs = f"{g['mean']:.3f} ± {g['std']:.3f}" if g["mean"] is not None else "n/a"
        rs = f"{r['mean']:.3f} ± {r['std']:.3f}" if r["mean"] is not None else "n/a"
        print(f"  {lbl:26s} {gs:>18s} {rs:>18s}   ({hint})")

    # C1 is a pass/fail proportion rather than a continuous statistic.
    for nm, rows in (("generated", grows), ("real", rrows)):
        ok = [r["C1_order_ok"] for r in rows if not r.get("degenerate")]
        res["aggregate"].setdefault("C1_vertical_order", {})[nm] = {
            "n": len(ok), "pass_rate": float(np.mean(ok)) if ok else None}
    c1 = res["aggregate"]["C1_vertical_order"]
    print(f"  {'C1 ground<wall<roof':26s} "
          f"{100*c1['generated']['pass_rate']:17.1f}% "
          f"{100*c1['real']['pass_rate']:17.1f}%   (higher = better)")

    # B5 coverage: how many buildings were large enough to measure at all.
    for nm, rows in (("generated", grows), ("real", rrows)):
        meas = sum(1 for r in rows if np.isfinite(r.get("B5_fractal_dim", np.nan)))
        res["aggregate"].setdefault("B5_measurable", {})[nm] = {
            "measured": meas, "total": len(rows), "frac": meas / len(rows)}
    b5 = res["aggregate"]["B5_measurable"]
    print(f"  {'B5 measurable':26s} {100*b5['generated']['frac']:17.1f}% "
          f"{100*b5['real']['frac']:17.1f}%   (too-small buildings excluded)")

    # -- per condition ------------------------------------------------
    print(f"\n=== per condition ===")
    for name in sorted(set(rcond.tolist())):
        gsel = [grows[i] for i in np.where(gcond == name)[0]]
        rsel = [rrows[i] for i in np.where(rcond == name)[0]]
        d = {}
        for key, _lbl, _h in METRICS_BC:
            d[key] = {"generated": _agg(gsel, key), "real": _agg(rsel, key)}
        gok = [r["C1_order_ok"] for r in gsel if not r.get("degenerate")]
        rok = [r["C1_order_ok"] for r in rsel if not r.get("degenerate")]
        d["C1_vertical_order"] = {
            "generated": float(np.mean(gok)) if gok else None,
            "real": float(np.mean(rok)) if rok else None}
        res["per_condition"][name] = d
        print(f"  {name}")
        print(f"      C1 order  gen={100*(d['C1_vertical_order']['generated'] or 0):5.1f}%  "
              f"real={100*(d['C1_vertical_order']['real'] or 0):5.1f}%"
              f"   |  roof exposed gen={d['C2_roof_exposed']['generated']['mean'] or float('nan'):.3f}"
              f"  real={d['C2_roof_exposed']['real']['mean'] or float('nan'):.3f}")

    (out_dir / "metrics_pillarBC.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[eval] wrote {out_dir/'metrics_pillarBC.json'}")

    pd_path = out_dir / "plotdata_pillarBC.npz"
    arrs = {}
    for key, _l, _h in METRICS_BC:
        arrs[f"gen_{key}"] = np.array([r.get(key, np.nan) for r in grows], float)
        arrs[f"real_{key}"] = np.array([r.get(key, np.nan) for r in rrows], float)
    arrs["gen_cond"] = gcond
    arrs["real_cond"] = rcond
    arrs["gen_C1"] = np.array([r.get("C1_order_ok", False) for r in grows], bool)
    arrs["real_C1"] = np.array([r.get("C1_order_ok", False) for r in rrows], bool)
    np.savez_compressed(pd_path, **arrs)
    print(f"[eval] plot data -> {pd_path}")

    figures_pillarBC(pd_path, out_dir)
    return res


def figures_pillarBC(pd_path: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(pd_path, allow_pickle=True)
    figs, tabs = out_dir / "figures", out_dir / "tables"
    m = json.loads((out_dir / "metrics_pillarBC.json").read_text())

    # --- F5  structural metric distributions ------------------------
    panels = [("B2_wall_verticality", "wall verticality"),
              ("B3_ground_horizontality", "ground horizontality"),
              ("B4_mean_neighbours", "surface smoothness\n(mean same-class neighbours)"),
              ("C2_roof_exposed", "roof exposed from above")]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), dpi=140)
    for ax, (key, lbl) in zip(axes.ravel(), panels):
        g = z[f"gen_{key}"]; r = z[f"real_{key}"]
        g = g[np.isfinite(g)]; r = r[np.isfinite(r)]
        lo, hi = min(g.min(), r.min()), max(g.max(), r.max())
        bins = np.linspace(lo, hi, 35)
        ax.hist(r, bins=bins, alpha=0.55, color=C_REAL, label="real (matched)")
        ax.hist(g, bins=bins, alpha=0.55, color=C_GEN, label=_glabel())
        ax.axvline(r.mean(), color=C_REAL, ls="--", lw=1.1)
        ax.axvline(g.mean(), color=C_GEN, ls="--", lw=1.1)
        _fig_style(ax, lbl, "buildings")
        ax.legend(fontsize=8)
    _save(fig, figs / "eval_structural.png")

    # --- F6  fractal dimension --------------------------------------
    g = z["gen_B5_fractal_dim"]; r = z["real_B5_fractal_dim"]
    g = g[np.isfinite(g)]; r = r[np.isfinite(r)]
    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
    bins = np.linspace(1.8, 3.0, 45)
    ax.hist(r, bins=bins, alpha=0.55, color=C_REAL, label=f"real (n={len(r)}, mean {r.mean():.3f})")
    ax.hist(g, bins=bins, alpha=0.55, color=C_GEN, label=_glabel(f" (n={len(g)}, mean {g.mean():.3f})"))
    # Reference lines make the scale interpretable: a flat plane is 2, a
    # filled solid 3, and uniform noise measured 2.735 on this grid.
    for xv, txt in ((2.0, "flat plane"), (2.735, "uniform noise"), (3.0, "filled solid")):
        ax.axvline(xv, color=C_REF, ls=":", lw=1.0)
        ax.annotate(txt, (xv, ax.get_ylim()[1]*0.97), fontsize=7.5, color=C_REF,
                    rotation=90, ha="right", va="top")
    _fig_style(ax, "box-counting fractal dimension D", "buildings")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, figs / "eval_fractal_dim.png")

    # --- table -------------------------------------------------------
    agg = m["aggregate"]
    rows = []
    for key, lbl, hint in METRICS_BC:
        a = agg[key]
        gm, rm = a["generated"], a["real"]
        if gm["mean"] is None or rm["mean"] is None:
            continue
        rows.append([lbl,
                     f"{gm['mean']:.3f} $\\pm$ {gm['std']:.3f}",
                     f"{rm['mean']:.3f} $\\pm$ {rm['std']:.3f}",
                     hint])
    c1 = agg["C1_vertical_order"]
    rows.append(["vertical order ground$<$wall$<$roof",
                 f"{100*c1['generated']['pass_rate']:.1f}\\%",
                 f"{100*c1['real']['pass_rate']:.1f}\\%",
                 "higher = better"])
    _tex_table(
        tabs / "eval_BC_structural.tex",
        caption="Geometric and semantic structure of generated buildings against "
                "condition-matched real ones. Every quantity is measured on the real "
                "set as well, because voxelising a thin shell at 0.5\\,m introduces "
                "artefacts and the real distribution -- not a theoretical ideal -- is "
                "the correct benchmark.",
        label="tab:eval-structural",
        header=["Metric", "Generated", "Real (matched)", "Direction"],
        rows=rows,
        note=f"Fractal dimension measurable on "
             f"{100*agg['B5_measurable']['generated']['frac']:.0f}\\% of generated and "
             f"{100*agg['B5_measurable']['real']['frac']:.0f}\\% of real buildings; "
             f"smaller buildings admit too few box scales for a stable fit.",
    )


# ---------------------------------------------------------------------
# stage 4 — Pillars E and F
# ---------------------------------------------------------------------
def pillar_EF(cache: dict, out_dir: Path) -> dict:
    gen, real = cache["gen"], cache["real"]
    gcond = np.array([m["name"] for m in cache["gen_meta"]])
    rcond = np.array([m["name"] for m in cache["real_meta"]])

    print("\n[eval] computing topology + watertightness (~0.26 s/building) ...")
    t0 = time.time()
    grows = [topology_metrics(l) for l in _labels(gen)]
    print(f"[eval]   generated done ({time.time()-t0:.0f}s)")
    rrows = [topology_metrics(l) for l in _labels(real)]
    print(f"[eval]   {len(grows)+len(rrows)} buildings in {time.time()-t0:.0f}s")

    def rate(rows, key):
        v = [r[key] for r in rows if not r.get("degenerate")]
        return float(np.mean(v)) if v else None

    res: dict = {"aggregate": {}, "per_condition": {}}
    for nm, rows in (("generated", grows), ("real", rrows)):
        ok = [r for r in rows if not r.get("degenerate")]
        comps = np.array([r["E1_components"] for r in ok], float)
        cav = np.array([r["E3_cavities"] for r in ok], float)
        vol = np.array([r["F3_enclosed_volume"] for r in ok], float)
        sol = np.array([r["F4_solidity_bbox"] for r in ok], float)
        res["aggregate"][nm] = {
            "n": len(ok),
            "E1_single_component_rate": rate(rows, "E1_single_component"),
            "E1_components_mean": float(comps.mean()),
            "E1_components_median": float(np.median(comps)),
            "E1_components_max": float(comps.max()),
            "E3_cavities_mean": float(cav.mean()),
            "F1_watertight_rate": rate(rows, "F1_watertight"),
            "F3_enclosed_volume_mean": float(vol.mean()),
            "F4_solidity_mean": float(sol[np.isfinite(sol)].mean()),
        }

    g, r = res["aggregate"]["generated"], res["aggregate"]["real"]
    print("\n=== Pillars E + F ===")
    print(f"  {'metric':30s} {'generated':>14s} {'real (matched)':>16s}")
    print(f"  {'E1 single component':30s} {100*g['E1_single_component_rate']:13.1f}% "
          f"{100*r['E1_single_component_rate']:15.1f}%")
    print(f"  {'E1 components (mean)':30s} {g['E1_components_mean']:14.2f} "
          f"{r['E1_components_mean']:16.2f}")
    print(f"  {'E1 components (max)':30s} {g['E1_components_max']:14.0f} "
          f"{r['E1_components_max']:16.0f}")
    print(f"  {'E3 cavities (mean)':30s} {g['E3_cavities_mean']:14.2f} "
          f"{r['E3_cavities_mean']:16.2f}")
    print(f"  {'F1 watertight':30s} {100*g['F1_watertight_rate']:13.1f}% "
          f"{100*r['F1_watertight_rate']:15.1f}%")
    print(f"  {'F3 enclosed volume (voxels)':30s} {g['F3_enclosed_volume_mean']:14.0f} "
          f"{r['F3_enclosed_volume_mean']:16.0f}")
    print(f"  {'F4 solidity (vs bbox)':30s} {g['F4_solidity_mean']:14.3f} "
          f"{r['F4_solidity_mean']:16.3f}")

    print("\n=== per condition ===")
    print(f"  {'condition':38s} {'1-comp g/r':>14s} {'watertight g/r':>16s}")
    for name in sorted(set(rcond.tolist())):
        gs = [grows[i] for i in np.where(gcond == name)[0]]
        rs = [rrows[i] for i in np.where(rcond == name)[0]]
        d = {
            "generated": {"E1_single_component_rate": rate(gs, "E1_single_component"),
                          "F1_watertight_rate": rate(gs, "F1_watertight")},
            "real": {"E1_single_component_rate": rate(rs, "E1_single_component"),
                     "F1_watertight_rate": rate(rs, "F1_watertight")},
        }
        res["per_condition"][name] = d
        print(f"  {name:38s} "
              f"{100*d['generated']['E1_single_component_rate']:6.1f}%/"
              f"{100*d['real']['E1_single_component_rate']:5.1f}% "
              f"{100*d['generated']['F1_watertight_rate']:8.1f}%/"
              f"{100*d['real']['F1_watertight_rate']:5.1f}%")

    (out_dir / "metrics_pillarEF.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[eval] wrote {out_dir/'metrics_pillarEF.json'}")

    pd_path = out_dir / "plotdata_pillarEF.npz"
    np.savez_compressed(
        pd_path,
        gen_components=np.array([r.get("E1_components", np.nan) for r in grows], float),
        real_components=np.array([r.get("E1_components", np.nan) for r in rrows], float),
        gen_cavities=np.array([r.get("E3_cavities", np.nan) for r in grows], float),
        real_cavities=np.array([r.get("E3_cavities", np.nan) for r in rrows], float),
        gen_volume=np.array([r.get("F3_enclosed_volume", np.nan) for r in grows], float),
        real_volume=np.array([r.get("F3_enclosed_volume", np.nan) for r in rrows], float),
        gen_solidity=np.array([r.get("F4_solidity_bbox", np.nan) for r in grows], float),
        real_solidity=np.array([r.get("F4_solidity_bbox", np.nan) for r in rrows], float),
        gen_watertight=np.array([r.get("F1_watertight", False) for r in grows], bool),
        real_watertight=np.array([r.get("F1_watertight", False) for r in rrows], bool),
        gen_cond=gcond, real_cond=rcond,
    )
    print(f"[eval] plot data -> {pd_path}")
    figures_pillarEF(pd_path, out_dir)
    return res


def figures_pillarEF(pd_path: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(pd_path, allow_pickle=True)
    figs, tabs = out_dir / "figures", out_dir / "tables"
    m = json.loads((out_dir / "metrics_pillarEF.json").read_text())
    g, r = m["aggregate"]["generated"], m["aggregate"]["real"]

    # --- F7  components + watertightness summary --------------------
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), dpi=140)

    ax = axes[0]
    mx = int(max(np.nanmax(z["gen_components"]), np.nanmax(z["real_components"])))
    bins = np.arange(0.5, min(mx, 12) + 1.5)
    ax.hist(z["real_components"], bins=bins, alpha=0.55, color=C_REAL, label="real (matched)")
    ax.hist(z["gen_components"], bins=bins, alpha=0.55, color=C_GEN, label=_glabel())
    _fig_style(ax, "connected components  $b_0$", "buildings")
    ax.legend(fontsize=8)

    ax = axes[1]
    gs = z["gen_solidity"]; rs = z["real_solidity"]
    gs = gs[np.isfinite(gs)]; rs = rs[np.isfinite(rs)]
    bins = np.linspace(0, 1, 40)
    ax.hist(rs, bins=bins, alpha=0.55, color=C_REAL, label=f"real (mean {rs.mean():.3f})")
    ax.hist(gs, bins=bins, alpha=0.55, color=C_GEN, label=f"generated (mean {gs.mean():.3f})")
    _fig_style(ax, "solidity  (filled volume / bounding box)", "buildings")
    ax.legend(fontsize=8)

    ax = axes[2]
    labels = ["single\ncomponent", "watertight"]
    gv = [100*g["E1_single_component_rate"], 100*g["F1_watertight_rate"]]
    rv = [100*r["E1_single_component_rate"], 100*r["F1_watertight_rate"]]
    x = np.arange(2); w = 0.38
    ax.bar(x - w/2, rv, w, color=C_REAL, label="real (matched)")
    ax.bar(x + w/2, gv, w, color=C_GEN, label=_glabel())
    for i in range(2):
        ax.text(x[i]-w/2, rv[i], f"{rv[i]:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(x[i]+w/2, gv[i], f"{gv[i]:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_ylim(0, 108)
    _fig_style(ax, "", "% of buildings")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, figs / "eval_topology.png")

    _tex_table(
        tabs / "eval_EF_topology.tex",
        caption="Topological quality and watertightness. A building shell should form a "
                "single connected component enclosing one cavity. Both quantities are "
                "measured on the real set as well: voxelising a thin shell at 0.5\\,m can "
                "open pin-holes at oblique surface junctions, so the real rate -- not "
                "$100\\%$ -- is the attainable benchmark.",
        label="tab:eval-topology",
        header=["Metric", "Generated", "Real (matched)"],
        rows=[
            ["Single connected component ($b_0=1$)",
             f"{100*g['E1_single_component_rate']:.1f}\\%", f"{100*r['E1_single_component_rate']:.1f}\\%"],
            ["Components, mean", f"{g['E1_components_mean']:.2f}", f"{r['E1_components_mean']:.2f}"],
            ["Components, max", f"{g['E1_components_max']:.0f}", f"{r['E1_components_max']:.0f}"],
            ["Enclosed cavities ($b_2$), mean", f"{g['E3_cavities_mean']:.2f}", f"{r['E3_cavities_mean']:.2f}"],
            ["Watertight (encloses volume)",
             f"{100*g['F1_watertight_rate']:.1f}\\%", f"{100*r['F1_watertight_rate']:.1f}\\%"],
            ["Enclosed volume, mean (voxels)",
             f"{g['F3_enclosed_volume_mean']:.0f}", f"{r['F3_enclosed_volume_mean']:.0f}"],
            ["Solidity vs bounding box", f"{g['F4_solidity_mean']:.3f}", f"{r['F4_solidity_mean']:.3f}"],
        ],
        note="Solidity uses the axis-aligned bounding box rather than a convex hull "
             "(which would require an additional dependency); it is a comparative "
             "descriptor, meaningful only between the two columns.",
    )


# ---------------------------------------------------------------------
# stage 5 — Pillar D, conditional validity
# ---------------------------------------------------------------------
VOXEL_M = 0.5          # metres per voxel (shard attribute `voxel_size`)


def _height_m(lab: np.ndarray) -> float:
    """Building height in metres = voxel size x occupied vertical extent.

    Definition fixed by the supervisor annotation on exposé §4.5 and by
    claude.md §19.8: height = delta * (occupied Z-extent).
    """
    fg = lab > 0
    zs = np.where(fg.any(axis=(0, 1)))[0]
    return float((zs.max() - zs.min() + 1) * VOXEL_M) if zs.size else np.nan


def _roof_z_variance(lab: np.ndarray) -> float:
    """Spread of roof voxel heights. Flat roofs -> near 0; gabled -> large.

    Raw, in voxels. This quantity is CONFOUNDED WITH BUILDING HEIGHT --- a
    taller building has more vertical room for its roof to spread over ---
    so it must not be compared across conditions of differing height, and
    it is inflated by the §40.2 collapse. Use _roof_pitch() for the
    height-free comparison.
    """
    roof = lab == 2
    if roof.sum() < 4:
        return np.nan
    zi = np.arange(lab.shape[2])
    zs = np.repeat(zi, roof.sum(axis=(0, 1)))
    return float(zs.std())


def _roof_pitch(lab: np.ndarray) -> tuple[float, float]:
    """Height-free descriptors of roof shape.

    Returns (relative_extent, pitch_proxy).

    relative_extent = roof vertical extent / building vertical extent.
      Dimensionless; a flat roof gives ~0 whatever the building height.

    pitch_proxy = roof vertical extent / (half the roof's mean horizontal
      extent), i.e. rise/run --- an approximation of tan(pitch angle).
      This is the physically meaningful discriminator between roof types,
      because a gable's rise scales with the SPAN it covers, not with how
      tall the building underneath happens to be.
    """
    roof = lab == 2
    fg = lab > 0
    if roof.sum() < 4 or not fg.any():
        return np.nan, np.nan
    rz = np.where(roof.any(axis=(0, 1)))[0]
    bz = np.where(fg.any(axis=(0, 1)))[0]
    rise = float(rz.max() - rz.min() + 1)
    bldg = float(bz.max() - bz.min() + 1)
    rx = np.where(roof.any(axis=(1, 2)))[0]
    ry = np.where(roof.any(axis=(0, 2)))[0]
    run = 0.5 * 0.5 * ((rx.max() - rx.min() + 1) + (ry.max() - ry.min() + 1))
    return (rise / bldg if bldg else np.nan,
            rise / run if run else np.nan)


# Empirical height-per-storey bands, 5th-95th percentile per function
# class, measured on all 10,000 buildings of the training manifest:
#   pd.read_csv('tensorbuilding/shards/manifest.csv') -> measured_height
#   / storeys_above_ground, grouped by function_label.
#
# These exist because the advisor's stated band of 2.5-5.0 m per storey
# (exposé §4.5 margin note) covers only 66.0 % of REAL buildings, so on
# its own it penalises the model for reproducing the data. Both tests are
# reported: D1a is the advisor's criterion as stated, D1b is the same
# criterion calibrated so that ~90 % of real buildings pass by
# construction and the generated rate is therefore interpretable.
HPS_BAND_BY_FUNCTION: dict[str, tuple[float, float]] = {
    "agricultural_shed": (2.26, 10.04),
    "non_residential":   (2.18,  6.67),
    "residential":       (3.19,  8.36),
    "storage_building":  (2.22,  5.88),
    "unclassified":      (3.16,  7.90),
}
HPS_BAND_DEFAULT = (2.27, 7.54)      # all functions pooled


def pillar_D(cache: dict, out_dir: Path) -> dict:
    """D1 storey-height criterion, D2 roof-type control, D3 height control."""
    gen, real = cache["gen"], cache["real"]
    gmeta, rmeta = cache["gen_meta"], cache["real_meta"]
    lg, lr = _labels(gen), _labels(real)

    # The condition each generated sample was asked for. DEFAULT_CONDITIONS
    # carries the requested storeys/height, so the ask is known exactly.
    by_name = {c["name"]: c for c in DEFAULT_CONDITIONS}

    res: dict = {"D1_storey_height": {}, "D2_roof_type": {}, "D3_height": {}}

    # ---- D1  the advisor's criterion --------------------------------
    # "if I ask for a 3 storey building its height must be within a
    #  reasonable interval like 7m - 15m"  (exposé §4.5 margin note)
    # Generalised: a storey is taken as 2.5-5.0 m, so an s-storey building
    # should measure between 2.5s and 5s metres, and the 3-storey case
    # reproduces the stated 7.5-15 m band.
    LO_PER_STOREY, HI_PER_STOREY = 2.5, 5.0
    print("\n=== D1  storey -> height criterion ===")
    print("  D1a = advisor's stated band (2.5-5.0 m/storey)")
    print("  D1b = same test, band calibrated per function on the real data")
    print(f"\n  {'condition':38s} {'st':>3s} {'D1a band':>11s} {'gen':>7s} {'real':>7s}"
          f"   {'D1b band':>11s} {'gen':>7s} {'real':>7s}")
    d1_rows = []
    for name in sorted(by_name):
        cond = by_name[name]
        st = float(cond.get("storeys_above_ground", np.nan))
        sel = np.array([m["name"] == name for m in gmeta])
        if not sel.any() or not np.isfinite(st):
            continue
        h = np.array([_height_m(l) for l in lg[sel]])
        rsel = np.array([m["name"] == name for m in rmeta])
        hr = np.array([_height_m(l) for l in lr[rsel]]) if rsel.any() else np.array([])

        # D1a — the advisor's criterion exactly as stated
        lo, hi = LO_PER_STOREY * st, HI_PER_STOREY * st
        ok = float(((h >= lo) & (h <= hi)).mean())
        ok_r = float(((hr >= lo) & (hr <= hi)).mean()) if hr.size else np.nan

        # D1b — the same criterion, calibrated on real buildings of this
        # function class so that the test does not itself reject reality
        blo, bhi = HPS_BAND_BY_FUNCTION.get(
            str(cond.get("function_label", "")), HPS_BAND_DEFAULT)
        lo_b, hi_b = blo * st, bhi * st
        ok_b = float(((h >= lo_b) & (h <= hi_b)).mean())
        ok_rb = float(((hr >= lo_b) & (hr <= hi_b)).mean()) if hr.size else np.nan

        res["D1_storey_height"][name] = {
            "storeys_requested": st,
            "function": cond.get("function_label", ""),
            "D1a_band_m": [lo, hi], "D1a_gen_pass": ok, "D1a_real_pass": ok_r,
            "D1b_band_m": [lo_b, hi_b], "D1b_gen_pass": ok_b,
            "D1b_real_pass": ok_rb,
            "gen_height_mean": float(h.mean()), "gen_height_std": float(h.std()),
            "real_height_mean": float(hr.mean()) if hr.size else None,
        }
        d1_rows.append((name, st, lo, hi, h.mean(), h.std(), ok, ok_r))
        print(f"  {name:38s} {st:3.0f} {f'{lo:.1f}-{hi:.1f}':>11s} "
              f"{100*ok:6.1f}% {100*ok_r:6.1f}%   "
              f"{f'{lo_b:.1f}-{hi_b:.1f}':>11s} {100*ok_b:6.1f}% {100*ok_rb:6.1f}%")

    allh = np.array([_height_m(l) for l in lg])
    allhr = np.array([_height_m(l) for l in lr])
    acc: dict[str, list] = {"a_g": [], "a_r": [], "b_g": [], "b_r": []}
    for name, d in res["D1_storey_height"].items():
        st = d["storeys_requested"]
        hg = allh[np.array([m["name"] == name for m in gmeta])]
        hrr = allhr[np.array([m["name"] == name for m in rmeta])]
        lo, hi = d["D1a_band_m"]; lo_b, hi_b = d["D1b_band_m"]
        acc["a_g"].extend(((hg >= lo) & (hg <= hi)).tolist())
        acc["a_r"].extend(((hrr >= lo) & (hrr <= hi)).tolist())
        acc["b_g"].extend(((hg >= lo_b) & (hg <= hi_b)).tolist())
        acc["b_r"].extend(((hrr >= lo_b) & (hrr <= hi_b)).tolist())
    res["D1a_overall"] = {"gen": float(np.mean(acc["a_g"])),
                          "real": float(np.mean(acc["a_r"]))}
    res["D1b_overall"] = {"gen": float(np.mean(acc["b_g"])),
                          "real": float(np.mean(acc["b_r"]))}
    res["D1_overall_pass_rate"] = res["D1a_overall"]["gen"]   # back-compat
    print(f"  {'OVERALL':38s} {'':3s} {'':11s} "
          f"{100*res['D1a_overall']['gen']:6.1f}% {100*res['D1a_overall']['real']:6.1f}%   "
          f"{'':11s} {100*res['D1b_overall']['gen']:6.1f}% "
          f"{100*res['D1b_overall']['real']:6.1f}%")

    # ---- D2  roof-type control --------------------------------------
    # Does the requested roof type change the produced roof geometry?
    # Flat roofs should have low z-variance, gabled/monopitch higher.
    # Raw z-variance scales with building height, so under the §40.2
    # collapse it rises for reasons that have nothing to do with the roof.
    # The pitch proxy (rise/run) is height-free and is the figure to read.
    print("\n=== D2  roof-type control ===")
    print(f"  {'condition':38s} {'asked':>10s} "
          f"{'z-std (raw)':>14s} {'rel. extent':>14s} {'pitch rise/run':>16s}")
    for name in sorted(by_name):
        cond = by_name[name]
        gsel = np.array([m["name"] == name for m in gmeta])
        rsel = np.array([m["name"] == name for m in rmeta])

        def _stats(labels):
            v = np.array([_roof_z_variance(l) for l in labels], float)
            rp = np.array([_roof_pitch(l) for l in labels], float)
            f = lambda a: float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
            return f(v), f(rp[:, 0]), f(rp[:, 1])

        gz, gre, gp = _stats(lg[gsel])
        rz, rre, rp_ = _stats(lr[rsel]) if rsel.any() else (np.nan,)*3
        res["D2_roof_type"][name] = {
            "roof_requested": cond.get("roof_type_label", ""),
            "gen_roof_z_std": gz, "real_roof_z_std": rz,
            "gen_roof_rel_extent": gre, "real_roof_rel_extent": rre,
            "gen_roof_pitch": gp, "real_roof_pitch": rp_,
        }
        print(f"  {name:38s} {cond.get('roof_type_label',''):>10s} "
              f"{f'{gz:.2f} / {rz:.2f}':>14s} {f'{gre:.2f} / {rre:.2f}':>14s} "
              f"{f'{gp:.2f} / {rp_:.2f}':>16s}")
    print("                                                  "
          "  (generated / real)")

    # ---- D3  requested vs realised height ---------------------------
    print("\n=== D3  requested vs realised height ===")
    print(f"  {'condition':38s} {'asked (m)':>10s} {'gen (m)':>16s} {'error':>9s}")
    for name in sorted(by_name):
        want = float(by_name[name].get("measured_height", np.nan))
        sel = np.array([m["name"] == name for m in gmeta])
        if not sel.any() or not np.isfinite(want):
            continue
        h = allh[sel]
        res["D3_height"][name] = {
            "requested_m": want, "gen_mean_m": float(h.mean()),
            "mae_m": float(np.abs(h - want).mean()),
        }
        print(f"  {name:38s} {want:10.2f} {f'{h.mean():.2f} ± {h.std():.2f}':>16s} "
              f"{np.abs(h-want).mean():8.2f}m")

    (out_dir / "metrics_pillarD.json").write_text(json.dumps(res, indent=2, default=str))
    print(f"\n[eval] wrote {out_dir/'metrics_pillarD.json'}")

    pd_path = out_dir / "plotdata_pillarD.npz"
    np.savez_compressed(
        pd_path,
        gen_height=allh,
        real_height=np.array([_height_m(l) for l in lr], float),
        gen_roofz=np.array([_roof_z_variance(l) for l in lg], float),
        real_roofz=np.array([_roof_z_variance(l) for l in lr], float),
        gen_roofpitch=np.array([_roof_pitch(l) for l in lg], float),
        real_roofpitch=np.array([_roof_pitch(l) for l in lr], float),
        gen_cond=np.array([m["name"] for m in gmeta]),
        real_cond=np.array([m["name"] for m in rmeta]),
        storeys=np.array([by_name[m["name"]].get("storeys_above_ground", np.nan)
                          for m in gmeta], float),
    )
    print(f"[eval] plot data -> {pd_path}")
    figures_pillarD(pd_path, out_dir)
    return res


def figures_pillarD(pd_path: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(pd_path, allow_pickle=True)
    figs, tabs = out_dir / "figures", out_dir / "tables"
    m = json.loads((out_dir / "metrics_pillarD.json").read_text())

    # --- F8  THE headline figure: storeys -> height ------------------
    # This is the advisor's own acceptance criterion, and the most
    # immediately legible result in the whole evaluation.
    st, h, cond = z["storeys"], z["gen_height"], z["gen_cond"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=140)
    jit = (np.random.default_rng(0).random(st.size) - 0.5) * 0.18
    for name in sorted(set(cond.tolist())):
        s = cond == name
        ax.scatter(st[s] + jit[s], h[s], s=11, alpha=0.45, label=name)
    smin, smax = np.nanmin(st), np.nanmax(st)
    xs = np.linspace(smin - 0.5, smax + 0.5, 50)
    ax.fill_between(xs, 2.5*xs, 5.0*xs, color="green", alpha=0.10,
                    label="acceptable band (2.5-5.0 m per storey)")
    ax.plot(xs, 2.5*xs, color="green", ls="--", lw=1.0)
    ax.plot(xs, 5.0*xs, color="green", ls="--", lw=1.0)
    ax.set_xticks(sorted(set(st[np.isfinite(st)].tolist())))
    _fig_style(ax, "storeys requested", "realised height (m)")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title(f"D1a: generated {100*m['D1a_overall']['gen']:.1f}%  vs  "
                 f"real {100*m['D1a_overall']['real']:.1f}%", fontsize=11)
    _save(fig, figs / "eval_D1_storey_height.png")

    # --- F9  roof-type control --------------------------------------
    # Two panels: the raw z-spread (height-confounded, shown for
    # continuity) beside the height-free pitch proxy, which is the one
    # that actually isolates roof shape from the §40.2 collapse.
    names = sorted(m["D2_roof_type"])
    lbl = [f"{n}\n(asked: {m['D2_roof_type'][n]['roof_requested']})" for n in names]
    y = np.arange(len(names)); hgt = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), dpi=140, sharey=True)
    for ax, (gk, rk, xl) in zip(axes, [
            ("gen_roof_z_std", "real_roof_z_std",
             "roof z-spread (voxels) — confounded with height"),
            ("gen_roof_pitch", "real_roof_pitch",
             "roof pitch proxy (rise / run) — height-free")]):
        gv = [m["D2_roof_type"][n].get(gk) or 0 for n in names]
        rv = [m["D2_roof_type"][n].get(rk) or 0 for n in names]
        ax.barh(y + hgt/2, rv, hgt, color=C_REAL, label="real (matched)")
        ax.barh(y - hgt/2, gv, hgt, color=C_GEN, label=_glabel())
        _fig_style(ax, xl, "")
    axes[0].set_yticks(y); axes[0].set_yticklabels(lbl, fontsize=7.5)
    axes[1].legend(fontsize=8, loc="lower right")
    _save(fig, figs / "eval_D2_roof_control.png")

    rows = []
    for n in sorted(m["D1_storey_height"]):
        d = m["D1_storey_height"][n]
        rows.append([n.replace("_", "\\_"), f"{d['storeys_requested']:.0f}",
                     f"{d['gen_height_mean']:.2f} $\\pm$ {d['gen_height_std']:.2f}",
                     f"{d['D1a_band_m'][0]:.1f}--{d['D1a_band_m'][1]:.1f}",
                     f"\\textbf{{{100*d['D1a_gen_pass']:.1f}\\%}}",
                     f"{100*d['D1a_real_pass']:.1f}\\%",
                     f"{d['D1b_band_m'][0]:.1f}--{d['D1b_band_m'][1]:.1f}",
                     f"\\textbf{{{100*d['D1b_gen_pass']:.1f}\\%}}",
                     f"{100*d['D1b_real_pass']:.1f}\\%"])
    _tex_table(
        tabs / "eval_D1_storey_height.tex",
        caption="Conditional validity of the storey attribute, under two bands. "
                "\\textbf{D1a} applies the acceptance criterion set out by the "
                "advisor -- a three-storey request should yield roughly "
                "7--15\\,m, so a storey is taken as 2.5--5.0\\,m and an "
                "$s$-storey request should land in $[2.5s, 5s]$. That band, "
                "however, admits only 66.0\\,\\% of the real buildings in the "
                "training set, so it penalises the model for reproducing the "
                "data. \\textbf{D1b} therefore repeats the test with the band "
                "set to the 5th--95th percentile of height-per-storey measured "
                "on real buildings \\emph{of the same function class}, by "
                "construction admitting about 90\\,\\% of real buildings. "
                "Height is $0.5\\,\\mathrm{m}$ times the occupied vertical "
                "extent. The `real' columns apply the identical test to the "
                "matched real buildings.",
        label="tab:eval-d1",
        header=["Condition", "St.", "Realised height (m)",
                "D1a band", "Gen", "Real", "D1b band", "Gen", "Real"],
        rows=rows,
        note=f"Overall: D1a generated \\textbf{{{100*m['D1a_overall']['gen']:.1f}\\%}} "
             f"vs real {100*m['D1a_overall']['real']:.1f}\\%; "
             f"D1b generated \\textbf{{{100*m['D1b_overall']['gen']:.1f}\\%}} "
             f"vs real {100*m['D1b_overall']['real']:.1f}\\%.",
    )


# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# H — retrieval baseline and memorisation check
# ---------------------------------------------------------------------
# Every metric so far compares the model against real data. None asks
# the question a reviewer asks first: is the generator doing anything a
# lookup table could not?
#
# The baseline answers a request the same way a database would --- return
# the closest real building carrying those attributes. It will win on
# every realism metric, because it IS real data. That is the point: it
# fixes the ceiling, and isolates what generation actually buys, which
# is novelty and within-condition diversity rather than realism.
#
# The same machinery yields the memorisation check the evaluation
# otherwise lacks. With 13.7 M parameters over 10 000 training buildings,
# "is it copying?" is a live question. Distance from each generated
# sample to its nearest TRAINING building answers it: near zero means
# copying; comparable to the real-to-real floor means the model is
# generating rather than retrieving.
def _require_free_gpu(need_gib: float = 6.0, force: bool = False) -> None:
    """Refuse to start if another process is already using the GPU.

    Learned the hard way on 2026-08-01: this analysis was launched while a
    training run held 13.9 of 20 GB. It cycled CUDA OOM, the retry loop
    kept re-requesting memory under pressure, and the training process
    died -- 3,600 iterations lost with no checkpoint yet written.

    Retrying smaller was the wrong remedy: the fix is not to squeeze into
    contended memory but to decline and run afterwards. Analysis is
    minutes; a training run is hours.
    """
    if not torch.cuda.is_available():
        return
    free_b, total_b = torch.cuda.mem_get_info()
    free, total = free_b / 2**30, total_b / 2**30
    used = total - free
    if used > 1.0 and not force:
        raise SystemExit(
            f"[eval] REFUSING TO START: {used:.1f} GiB of {total:.1f} GiB "
            f"GPU memory is already in use by another process.\n"
            f"        Running heavy analysis alongside training has "
            f"previously killed the training run.\n"
            f"        Wait for it to finish, or pass --force if you are "
            f"certain the GPU is free enough.")
    if free < need_gib and not force:
        raise SystemExit(
            f"[eval] REFUSING TO START: only {free:.1f} GiB free, "
            f"need ~{need_gib:.0f} GiB. Pass --force to override.")


def retrieval_baseline(cache: dict, out_dir: Path, shards_dir: str,
                       manifest_path: str, device_str: str = "cuda",
                       force: bool = False) -> dict:
    _require_free_gpu(force=force)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    gen, real = cache["gen"], cache["real"]
    gmeta = cache["gen_meta"]

    ds = Building3DDataset(shards_dir=shards_dir, manifest_path=manifest_path)
    print(f"[eval] training pool: {len(ds):,} buildings")

    # --- the baseline itself -----------------------------------------
    # For each generated sample's condition, retrieve the single closest
    # real building. Retrieval is deterministic per condition, so a
    # request repeated N times returns the SAME building N times --- the
    # diversity failure is intrinsic to the method, not an artefact.
    by_name: dict[str, dict] = {c["name"]: c for c in DEFAULT_CONDITIONS}
    counts = Counter(m["name"] for m in gmeta)
    retr, retr_meta = [], []
    distinct: dict[str, int] = {}
    for name, n in counts.items():
        ranked = _rank_real_by_condition(ds, by_name[name], k=1)
        if not ranked:
            continue
        idx = ranked[0][0]
        distinct[name] = 1
        item = ds[idx]
        t = item["tensor"]
        t = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
        for _ in range(n):
            retr.append(t.astype(np.uint8))
            retr_meta.append({"name": name})
    retr = np.stack(retr, axis=0)
    print(f"[eval] retrieval set: {retr.shape}  "
          f"({sum(distinct.values())} distinct buildings for "
          f"{len(retr)} requests)")

    lr_, lg = _labels(retr), _labels(gen)
    res: dict = {}

    def _summ(lab, arr, tag):
        occ = _occupancy(arr)
        tops = [topology_metrics(l) for l in lab]
        ok = [t for t in tops if not t.get("degenerate")]
        return {
            "occupancy_mean": float(occ.mean()),
            "watertight_rate": float(np.mean([t["F1_watertight"] for t in ok])),
            "single_component_rate": float(np.mean([t["E1_single_component"] for t in ok])),
            "enclosed_volume_mean": float(np.mean([t["F3_enclosed_volume"] for t in ok])),
        }

    res["retrieval"] = _summ(lr_, retr, "retrieval")
    res["generated"] = _summ(lg, gen, "generated")
    res["real_matched"] = _summ(_labels(real), real, "real")

    print("\n=== H1  retrieval baseline vs generation ===")
    print(f"  {'metric':24s} {'retrieval':>12s} {'generated':>12s} {'real':>12s}")
    for k in ("occupancy_mean", "single_component_rate",
              "watertight_rate", "enclosed_volume_mean"):
        print(f"  {k:24s} {res['retrieval'][k]:12.4f} "
              f"{res['generated'][k]:12.4f} {res['real_matched'][k]:12.4f}")

    # --- diversity: the axis retrieval must lose on ------------------
    # Mean pairwise distance WITHIN one condition. Retrieval returns one
    # building per condition, so its within-condition diversity is
    # exactly 0 by construction.
    print("\n=== H2  within-condition diversity ===")
    print(f"  {'condition':38s} {'generated':>10s} {'retrieval':>10s} {'real':>10s}")
    div: dict = {}
    for name in sorted(counts):
        gsel = np.array([m["name"] == name for m in gmeta])
        rsel = np.array([m["name"] == name for m in cache["real_meta"]])
        g = gen[gsel][:40]
        d_gg = semantic_iou_matrix(g, g, device)
        iu = np.triu_indices(len(g), k=1)
        dg = float(d_gg[iu].mean()) if len(g) > 1 else np.nan
        r = real[rsel][:40]
        d_rr = semantic_iou_matrix(r, r, device)
        iu2 = np.triu_indices(len(r), k=1)
        dr = float(d_rr[iu2].mean()) if len(r) > 1 else np.nan
        div[name] = {"generated": dg, "retrieval": 0.0, "real": dr}
        print(f"  {name:38s} {dg:10.3f} {0.0:10.3f} {dr:10.3f}")
    res["H2_within_condition_diversity"] = div

    # --- memorisation -------------------------------------------------
    # Distance from each generated sample to its nearest TRAINING
    # building, against the real-to-real floor of the same measure.
    print("\n=== H3  memorisation check ===")
    # The FULL training set, deliberately not a subsample: this is a
    # MINIMUM distance, so omitting buildings can only inflate it, which
    # would make the model look more novel than it is -- exactly the
    # wrong direction for a memorisation check.
    #
    # But the pool must never be materialised. All 10,000 buildings as
    # one array is 17 GiB of host RAM (34 GiB peak while np.stack copies)
    # and 7.3 GiB of VRAM once packed, which locks up the machine.
    # Instead the pool is streamed in slices and only a running minimum
    # is kept: memory stays flat in the pool size, and the answer is
    # identical to the full computation.
    POOL_CHUNK = 250                      # ~0.4 GiB per slice
    nn_gen = np.full(len(gen), np.inf, dtype=np.float64)
    nn_real = np.full(len(real), np.inf, dtype=np.float64)
    n_pool = len(ds)
    for s in range(0, n_pool, POOL_CHUNK):
        sl = range(s, min(s + POOL_CHUNK, n_pool))
        chunk = np.stack([np.asarray(ds[i]["tensor"]).astype(np.uint8)
                          for i in sl], axis=0)
        d_g = semantic_iou_matrix(gen, chunk, device)
        np.minimum(nn_gen, d_g.min(axis=1), out=nn_gen)
        d_r = semantic_iou_matrix(real, chunk, device)
        # A real evaluation building is itself in the pool, so its own
        # zero-distance self-match must be excluded; take the smallest
        # strictly-positive distance in this slice instead.
        d_r_masked = np.where(d_r < 1e-9, np.inf, d_r)
        np.minimum(nn_real, d_r_masked.min(axis=1), out=nn_real)
        del chunk, d_g, d_r, d_r_masked
        if (s // POOL_CHUNK) % 8 == 0:
            print(f"[eval]   memorisation: {min(s + POOL_CHUNK, n_pool)}"
                  f"/{n_pool} pool buildings", flush=True)
    res["H3_memorisation"] = {
        "pool_streamed_in_chunks_of": POOL_CHUNK,
        "gen_to_train_nn_mean": float(nn_gen.mean()),
        "gen_to_train_nn_min": float(nn_gen.min()),
        "gen_exact_copies": int((nn_gen < 1e-9).sum()),
        "real_to_train_nn_mean": float(nn_real.mean()),
        "pool_size": n_pool,
    }
    m = res["H3_memorisation"]
    print(f"  generated -> nearest training building : {m['gen_to_train_nn_mean']:.3f} "
          f"(min {m['gen_to_train_nn_min']:.3f})")
    print(f"  real      -> nearest OTHER training    : {m['real_to_train_nn_mean']:.3f}")
    print(f"  exact copies among generated           : {m['gen_exact_copies']}")
    print("  A generated distance near 0 would mean copying; near the real")
    print("  figure means the model generates rather than retrieves.")

    (out_dir / "metrics_retrieval.json").write_text(
        json.dumps(res, indent=2, default=str))
    print(f"\n[eval] wrote {out_dir/'metrics_retrieval.json'}")
    return res


# ---------------------------------------------------------------------
# D6 — height extrapolation probe
# ---------------------------------------------------------------------
# The training set is capped at 16 m and is naturally short anyway
# (median 5.0 m, p95 13.1 m), so the model has effectively never seen a
# tall building. This asks whether the `measured_height` field
# extrapolates past the range it was trained on, or saturates.
#
# Retraining without the height cap would NOT answer this: the pool is
# only 1.4 % taller than 16 m, so a 10 000-building sample would contain
# roughly 140 tall buildings against 9 860 short ones. The model would
# still barely see them. Probing the existing model isolates
# extrapolation instead of confounding it with a marginally different
# data mix, and costs minutes rather than GPU-hours.
#
# Only `measured_height` is varied; every other field is held at the
# base condition. That is deliberate --- it isolates the one field ---
# but it does mean the tall requests are internally inconsistent (a 30 m
# building still asking for 3 storeys), which is noted with the results.
def height_probe(ckpt_path: str, out_dir: Path, heights: list[float],
                 n_per_height: int, n_steps: int, guidance: float,
                 seed0: int, base_name: str) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_checkpoint(ckpt_path, device)
    param = str(bundle["cfg"].get("parameterization", "eps"))
    base = next(c for c in DEFAULT_CONDITIONS if c["name"] == base_name)
    trained_max = 16.0                      # the training-set height cap

    print(f"[eval] device={device}   checkpoint: {ckpt_path}")
    print(f"[eval] D6 height probe on base condition '{base_name}', w={guidance}")
    print(f"[eval] requested heights: {heights}   ({n_per_height} samples each)")
    print(f"[eval] training data capped at {trained_max:.0f} m "
          f"-- anything above is extrapolation\n")

    rows = []
    t0 = time.time()
    for h in heights:
        cond = dict(base); cond["measured_height"] = float(h)
        seeds = [seed0 + int(h) * 1000 + j for j in range(n_per_height)]
        got = []
        for s in range(0, len(seeds), 16):
            x, _ = ddim_sample(
                bundle["unet"], bundle["cond_enc"], bundle["schedule"],
                conditions=[cond], vocabs=bundle["vocabs"],
                continuous_stats=bundle["continuous_stats"],
                seeds=seeds[s:s + 16], n_steps=n_steps,
                guidance_scale=guidance, device=device, parameterization=param)
            lab = _labels(to_hard_onehot(x).cpu().numpy().astype(np.uint8))
            got.extend(_height_m(l) for l in lab)
        got = np.array(got, float)
        rows.append({"requested": float(h), "mean": float(np.nanmean(got)),
                     "std": float(np.nanstd(got)),
                     "max": float(np.nanmax(got)),
                     "in_range": bool(h <= trained_max),
                     "realised": got.tolist()})
        tag = "" if h <= trained_max else "  <- extrapolation"
        print(f"[eval]   asked {h:5.1f} m -> got {np.nanmean(got):5.2f} "
              f"± {np.nanstd(got):4.2f} m  (max {np.nanmax(got):5.2f})"
              f"  ({time.time()-t0:.0f}s){tag}", flush=True)

    inr = [r for r in rows if r["in_range"]]
    out = [r for r in rows if not r["in_range"]]
    print(f"\n=== D6 summary ===")
    if len(inr) >= 2:
        # slope of realised against requested, in range and out of range
        x = np.array([r["requested"] for r in inr]); y = np.array([r["mean"] for r in inr])
        s_in = np.polyfit(x, y, 1)[0]
        print(f"  slope within trained range : {s_in:+.3f} m per m requested")
    if out:
        x = np.array([r["requested"] for r in rows]); y = np.array([r["mean"] for r in rows])
        s_all = np.polyfit(x, y, 1)[0]
        ceiling = max(r["mean"] for r in rows)
        print(f"  slope over the full sweep  : {s_all:+.3f} m per m requested")
        print(f"  highest mean realised      : {ceiling:.2f} m")
        print(f"  tallest single sample      : {max(r['max'] for r in rows):.2f} m")
        print("  A slope near 0 outside the trained range means the field does")
        print("  not extrapolate: the model saturates at the heights it saw.")

    res = {"checkpoint": str(ckpt_path), "base_condition": base_name,
           "guidance": guidance, "trained_max_m": trained_max, "rows": rows}
    (out_dir / "metrics_heightprobe.json").write_text(
        json.dumps(res, indent=2, default=str))
    print(f"\n[eval] wrote {out_dir/'metrics_heightprobe.json'}")
    return out_dir / "metrics_heightprobe.json"


# ---------------------------------------------------------------------
# D4 — condition ablation
# ---------------------------------------------------------------------
# Which of the ten conditioning fields actually steer the output? A field
# the model ignores is dead weight in the encoder, and --- more
# importantly for Chapter 7 --- a field the model ignores cannot be
# offered to a user as a control.
#
# Ablation uses the mechanism the model was TRAINED with rather than a
# bolt-on: dropping a key from the condition dict makes _encode_conditions
# map a categorical to its "(null)" vocabulary token (index 0, which
# appears in training wherever the source attribute was missing) and a
# continuous field to 0.0, which after z-scoring is exactly the dataset
# mean. Both are therefore the model's own "unknown" value, not an
# out-of-distribution poke.
#
# Paired seeds throughout: baseline and ablated share the initial noise,
# so the difference isolates the field.
_ABLATION_FIELDS = [
    "function_label", "roof_type_label", "height_cluster", "ratio_cluster",
    "estimatedConstructionPeriod", "constructionPeriodReliability",
    "measured_height", "length_to_width_ratio",
    "constructionPeriodConfidence", "storeys_above_ground",
]


def _paired_iou_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """1 - IoU of the occupied region, per matched pair. Shape (N,)."""
    a, b = A > 0, B > 0
    ax = tuple(range(1, a.ndim))
    inter = (a & b).sum(axis=ax).astype(np.float64)
    union = (a | b).sum(axis=ax).astype(np.float64)
    return 1.0 - np.divide(inter, union, out=np.ones_like(inter),
                           where=union > 0)


def cond_ablation(ckpt_path: str, out_dir: Path, n_per_cond: int,
                  n_steps: int, guidance: float, seed0: int) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_checkpoint(ckpt_path, device)
    param = str(bundle["cfg"].get("parameterization", "eps"))
    conds = list(DEFAULT_CONDITIONS)

    print(f"[eval] device={device}   checkpoint: {ckpt_path}")
    print(f"[eval] D4 condition ablation, guidance w={guidance}")
    if guidance == 0.0:
        print("[eval]   (w=0 by design: at w=1.5 the §40.2 collapse swamps "
              "the field effects being measured)")
    n_runs = len(conds) * (len(_ABLATION_FIELDS) + 2)
    print(f"[eval] {len(conds)} conditions x ({len(_ABLATION_FIELDS)} fields "
          f"+ baseline + control) x {n_per_cond} = {n_runs*n_per_cond} generated\n")

    def _gen(cond: dict, seeds: list[int]) -> np.ndarray:
        out = []
        for s in range(0, len(seeds), 16):
            x, _ = ddim_sample(
                bundle["unet"], bundle["cond_enc"], bundle["schedule"],
                conditions=[cond], vocabs=bundle["vocabs"],
                continuous_stats=bundle["continuous_stats"],
                seeds=seeds[s:s + 16], n_steps=n_steps,
                guidance_scale=guidance, device=device,
                parameterization=param,
            )
            out.append(_labels(to_hard_onehot(x).cpu().numpy().astype(np.uint8)))
        return np.concatenate(out, axis=0)

    rows: list[dict] = []
    t0 = time.time()
    for ci, cond in enumerate(conds):
        seeds = [seed0 + ci * 100_000 + j for j in range(n_per_cond)]
        base = _gen(cond, seeds)
        bh = np.array([_height_m(l) for l in base], float)
        bp = np.array([_roof_pitch(l)[1] for l in base], float)
        bo = np.array([(l > 0).mean() for l in base], float)

        # "(control)" re-runs the identical condition. It must give a
        # distance of exactly 0; anything else means sampling is not
        # deterministic and every number below is untrustworthy.
        for field in ["(control)"] + _ABLATION_FIELDS:
            c2 = dict(cond)
            if field != "(control)":
                if field not in c2:
                    continue                     # not set for this archetype
                c2.pop(field)
            abl = _gen(c2, seeds)
            rows.append({
                "cond": cond["name"], "field": field,
                "iou_dist": float(_paired_iou_dist(base, abl).mean()),
                "d_height": float(np.nanmean(
                    np.abs(np.array([_height_m(l) for l in abl]) - bh))),
                "d_pitch": float(np.nanmean(
                    np.abs(np.array([_roof_pitch(l)[1] for l in abl]) - bp))),
                "d_occ": float(np.nanmean(
                    np.abs(np.array([(l > 0).mean() for l in abl]) - bo))),
            })
        print(f"[eval]   {cond['name']:38s} done  ({time.time()-t0:.0f}s)",
              flush=True)

    ctl = [r["iou_dist"] for r in rows if r["field"] == "(control)"]
    print(f"\n[eval] determinism control: max IoU distance "
          f"{max(ctl):.2e} (must be 0)")

    # ---- report ------------------------------------------------------
    print("\n=== D4  condition ablation — mean paired IoU distance ===")
    print("  (0 = removing the field changed nothing; larger = more influence)\n")
    names = sorted({r["cond"] for r in rows})
    print(f"  {'field':32s} " + " ".join(f"{n.split('_')[0][:9]:>9s}"
                                         for n in names) + f" {'mean':>8s}")
    order = []
    for f in ["(control)"] + _ABLATION_FIELDS:
        cells, vals = [], []
        for n in names:
            v = [r["iou_dist"] for r in rows if r["cond"] == n and r["field"] == f]
            cells.append(f"{v[0]:9.3f}" if v else f"{'—':>9s}")
            vals.extend(v)
        if not vals:
            continue
        order.append((f, float(np.mean(vals))))
        print(f"  {f:32s} " + " ".join(cells) + f" {np.mean(vals):8.3f}")

    print("\n=== D4  fields ranked by influence ===")
    for f, v in sorted([o for o in order if o[0] != "(control)"],
                       key=lambda t: -t[1]):
        dh = np.mean([r["d_height"] for r in rows if r["field"] == f])
        bar = "#" * int(round(40 * v / max(x[1] for x in order if x[0] != "(control)")))
        print(f"  {f:32s} {v:6.3f}  Δheight {dh:5.2f}m  {bar}")

    res = {"guidance": guidance, "n_per_condition": n_per_cond,
           "control_max_iou_dist": max(ctl),
           "ranking": [{"field": f, "iou_dist": v} for f, v in
                       sorted(order, key=lambda t: -t[1])],
           "rows": rows}
    (out_dir / "metrics_ablation.json").write_text(
        json.dumps(res, indent=2, default=str))
    pd_path = out_dir / "plotdata_ablation.npz"
    np.savez_compressed(
        pd_path,
        cond=np.array([r["cond"] for r in rows]),
        field=np.array([r["field"] for r in rows]),
        iou_dist=np.array([r["iou_dist"] for r in rows], float),
        d_height=np.array([r["d_height"] for r in rows], float),
        d_pitch=np.array([r["d_pitch"] for r in rows], float),
        d_occ=np.array([r["d_occ"] for r in rows], float),
    )
    print(f"\n[eval] wrote {out_dir/'metrics_ablation.json'}")
    print(f"[eval] plot data -> {pd_path}")
    figures_ablation(pd_path, out_dir)
    return pd_path


def figures_ablation(pd_path: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(pd_path, allow_pickle=True)
    fld, dist = z["field"], z["iou_dist"]
    keep = [f for f in dict.fromkeys(fld.tolist()) if f != "(control)"]
    means = {f: float(np.mean(dist[fld == f])) for f in keep}
    order = sorted(means, key=lambda f: means[f])

    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=140)
    y = np.arange(len(order))
    ax.barh(y, [means[f] for f in order], 0.62, color=C_GEN)
    for i, f in enumerate(order):
        for c in sorted(set(z["cond"].tolist())):
            v = dist[(fld == f) & (z["cond"] == c)]
            if v.size:
                ax.scatter(v, [i], s=9, color="black", alpha=0.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("_", " ") for f in order], fontsize=8)
    _fig_style(ax, "mean paired IoU distance when the field is removed", "")
    ax.set_title("D4: which conditioning fields steer the output?", fontsize=11)
    _save(fig, out_dir / "fig_ablation.png")

    _tex_table(
        out_dir / "tab_ablation.tex",
        caption="Condition ablation. Each field is removed in turn and the "
                "building regenerated from the identical initial noise; the "
                "value is the mean IoU distance between the paired outputs, "
                "so $0$ means the field had no effect. Removal sets a "
                "categorical field to the \\texttt{(null)} token and a "
                "continuous field to the dataset mean --- the same "
                "``unknown'' values the model saw during training.",
        label="tab:eval-d4",
        header=["Field", "IoU distance", "$\\Delta$ height (m)"],
        rows=[[f.replace("_", "\\_"), f"{means[f]:.3f}",
               f"{np.mean(z['d_height'][fld == f]):.2f}"]
              for f in sorted(means, key=lambda x: -means[x])])


# ---------------------------------------------------------------------
# D5 — classifier-free-guidance sweep
# ---------------------------------------------------------------------
# §40.2 found that 41.4 % of v4 samples collapse into a second, much
# taller mode (>= 15 m) that §41.2 showed is also fragmented. The
# leading suspect is the guidance weight: sampling used w = 1.5, and
# CFG is known to trade diversity for condition adherence, over-shooting
# the conditional mode when w is large.
#
# This is a PAIRED experiment: the same seed produces the same initial
# noise at every w, so any difference between guidance levels is
# attributable to w alone and not to the noise draw. With the
# implementation in sample.py --- (1+w)*cond - w*null --- w = 0 is plain
# conditional sampling with no guidance at all, which is the control.
COLLAPSE_M = 15.0          # height threshold separating the two modes (§40.2)


def cfg_sweep(ckpt_path: str, out_dir: Path, weights: list[float],
              n_per_cond: int, n_steps: int, seed0: int) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_checkpoint(ckpt_path, device)
    cfg = bundle["cfg"]
    param = str(cfg.get("parameterization", "eps"))
    conds = list(DEFAULT_CONDITIONS)

    print(f"[eval] device={device}   checkpoint: {ckpt_path}")
    print(f"[eval] D5 guidance sweep: w in {weights}")
    print(f"[eval] {len(conds)} conditions x {n_per_cond} samples x "
          f"{len(weights)} weights = {len(conds)*n_per_cond*len(weights)} generated")
    print(f"[eval] paired design: identical seeds at every w\n")

    rows: list[dict] = []
    t0 = time.time()
    for w in weights:
        for ci, cond in enumerate(conds):
            seeds = [seed0 + ci * 100_000 + j for j in range(n_per_cond)]
            for s in range(0, len(seeds), 16):
                chunk = seeds[s:s + 16]
                x, _ = ddim_sample(
                    bundle["unet"], bundle["cond_enc"], bundle["schedule"],
                    conditions=[cond], vocabs=bundle["vocabs"],
                    continuous_stats=bundle["continuous_stats"],
                    seeds=chunk, n_steps=n_steps, guidance_scale=w,
                    device=device, parameterization=param,
                )
                lab = _labels(to_hard_onehot(x).cpu().numpy().astype(np.uint8))
                for seed, l in zip(chunk, lab):
                    t = topology_metrics(l)
                    rows.append({
                        "w": float(w), "cond": cond["name"], "seed": int(seed),
                        "height_m": _height_m(l),
                        "occupancy": float((l > 0).sum() / l.size),
                        "components": t.get("E1_components", np.nan),
                        "single": bool(t.get("E1_single_component", False)),
                        "watertight": bool(t.get("F1_watertight", False)),
                        "storeys": float(cond.get("storeys_above_ground", np.nan)),
                    })
        done = [r for r in rows if r["w"] == w]
        h = np.array([r["height_m"] for r in done], float)
        print(f"[eval]   w={w:<4.1f} n={len(done):4d}  "
              f"height={np.nanmean(h):5.2f}m  "
              f"collapsed={100*np.nanmean(h >= COLLAPSE_M):5.1f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate ---------------------------------------------------
    res: dict = {"weights": weights, "n_per_condition": n_per_cond,
                 "collapse_threshold_m": COLLAPSE_M, "overall": {},
                 "per_condition": {}}

    print("\n=== D5  guidance sweep — overall ===")
    print(f"  {'w':>5s} {'height (m)':>14s} {'collapsed':>10s} {'occupancy':>10s} "
          f"{'1-comp':>8s} {'watertight':>11s} {'D1 pass':>8s}")
    for w in weights:
        sel = [r for r in rows if r["w"] == w]
        h = np.array([r["height_m"] for r in sel], float)
        st = np.array([r["storeys"] for r in sel], float)
        d1 = (h >= 2.5 * st) & (h <= 5.0 * st)
        rec = {
            "height_mean_m": float(np.nanmean(h)),
            "height_std_m": float(np.nanstd(h)),
            "collapse_rate": float(np.nanmean(h >= COLLAPSE_M)),
            "occupancy_mean": float(np.mean([r["occupancy"] for r in sel])),
            "single_component_rate": float(np.mean([r["single"] for r in sel])),
            "watertight_rate": float(np.mean([r["watertight"] for r in sel])),
            "D1_pass_rate": float(np.nanmean(d1)),
        }
        res["overall"][str(w)] = rec
        print(f"  {w:5.1f} {f'{rec['height_mean_m']:.2f} ± {rec['height_std_m']:.2f}':>14s} "
              f"{100*rec['collapse_rate']:9.1f}% {100*rec['occupancy_mean']:9.2f}% "
              f"{100*rec['single_component_rate']:7.1f}% "
              f"{100*rec['watertight_rate']:10.1f}% {100*rec['D1_pass_rate']:7.1f}%")

    print("\n=== D5  collapse rate per condition ===")
    print(f"  {'condition':38s} " + " ".join(f"{'w='+str(w):>8s}" for w in weights))
    for cond in sorted({r["cond"] for r in rows}):
        cells, per_w = [], {}
        for w in weights:
            h = np.array([r["height_m"] for r in rows
                          if r["w"] == w and r["cond"] == cond], float)
            cr = float(np.nanmean(h >= COLLAPSE_M))
            per_w[str(w)] = cr
            cells.append(f"{100*cr:7.1f}%")
        res["per_condition"][cond] = per_w
        print(f"  {cond:38s} " + " ".join(cells))

    (out_dir / "metrics_cfgsweep.json").write_text(
        json.dumps(res, indent=2, default=str))
    pd_path = out_dir / "plotdata_cfgsweep.npz"
    np.savez_compressed(
        pd_path,
        w=np.array([r["w"] for r in rows], float),
        cond=np.array([r["cond"] for r in rows]),
        seed=np.array([r["seed"] for r in rows], int),
        height=np.array([r["height_m"] for r in rows], float),
        occupancy=np.array([r["occupancy"] for r in rows], float),
        components=np.array([r["components"] for r in rows], float),
        single=np.array([r["single"] for r in rows], bool),
        watertight=np.array([r["watertight"] for r in rows], bool),
        storeys=np.array([r["storeys"] for r in rows], float),
    )
    print(f"\n[eval] wrote {out_dir/'metrics_cfgsweep.json'}")
    print(f"[eval] plot data -> {pd_path}")
    figures_cfgsweep(pd_path, out_dir)
    return pd_path


def figures_cfgsweep(pd_path: Path, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(pd_path, allow_pickle=True)
    w, h = z["w"], z["height"]
    ws = sorted(set(w.tolist()))

    # --- fig 1: collapse rate + height vs w --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    cr = [100 * np.nanmean(h[w == x] >= COLLAPSE_M) for x in ws]
    axes[0].plot(ws, cr, "o-", color=C_GEN, lw=1.8, ms=5)
    axes[0].axvline(1.5, color=C_REF, ls=":", lw=1.2)
    axes[0].annotate("v4 setting", xy=(1.5, max(cr) * 0.92), fontsize=7,
                     color="grey", ha="right", rotation=90)
    _fig_style(axes[0], "guidance weight $w$", "collapsed samples (%)",
               "Mode collapse vs guidance")

    for x in ws:
        axes[1].scatter([x] * (w == x).sum(), h[w == x], s=3, alpha=0.15,
                        color=C_GEN, edgecolors="none")
    axes[1].plot(ws, [np.nanmean(h[w == x]) for x in ws], "o-",
                 color="black", lw=1.6, ms=5, label="mean")
    axes[1].axhline(COLLAPSE_M, color=C_REF, ls="--", lw=1.0,
                    label=f"collapse threshold ({COLLAPSE_M:.0f} m)")
    axes[1].legend(fontsize=7, frameon=False)
    _fig_style(axes[1], "guidance weight $w$", "height (m)",
               "Height distribution vs guidance")
    _save(fig, out_dir / "fig_cfgsweep_collapse.png")

    # --- fig 2: per-condition collapse + structural quality ----------
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    for cond in sorted(set(z["cond"].tolist())):
        m = z["cond"] == cond
        axes[0].plot(ws, [100 * np.nanmean(h[m & (w == x)] >= COLLAPSE_M)
                          for x in ws], "o-", lw=1.4, ms=4,
                     label=cond.replace("_", " "))
    axes[0].legend(fontsize=6, frameon=False)
    _fig_style(axes[0], "guidance weight $w$", "collapsed (%)",
               "Collapse by condition")

    axes[1].plot(ws, [100 * z["single"][w == x].mean() for x in ws], "o-",
                 lw=1.6, ms=5, label="single component")
    axes[1].plot(ws, [100 * z["watertight"][w == x].mean() for x in ws], "s-",
                 lw=1.6, ms=5, label="watertight")
    axes[1].legend(fontsize=7, frameon=False)
    _fig_style(axes[1], "guidance weight $w$", "rate (%)",
               "Structural quality vs guidance")
    _save(fig, out_dir / "fig_cfgsweep_structure.png")

    _tex_table(
        out_dir / "tab_cfgsweep.tex",
        caption=("Classifier-free guidance sweep. Identical seeds at every "
                 "guidance weight, so differences are attributable to $w$ "
                 "alone. Collapse is defined as a realised height of at "
                 f"least {COLLAPSE_M:.0f}\\,m."),
        label="tab:cfgsweep",
        header=["$w$", "Height (m)", "Collapsed", "Single comp.",
                "Watertight", "D1 pass"],
        rows=[[f"{x:.1f}",
               f"{np.nanmean(h[w==x]):.2f}",
               f"{100*np.nanmean(h[w==x] >= COLLAPSE_M):.1f}\\%",
               f"{100*z['single'][w==x].mean():.1f}\\%",
               f"{100*z['watertight'][w==x].mean():.1f}\\%",
               f"{100*np.nanmean((h[w==x] >= 2.5*z['storeys'][w==x]) & (h[w==x] <= 5.0*z['storeys'][w==x])):.1f}\\%"]
              for x in ws])


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("pillar-a", help="stage 2: distributional realism + diversity")
    pa.add_argument("--eval-dir", required=True,
                    help="directory containing eval_samples.npz")

    pbc = sub.add_parser("pillar-bc", help="stage 3: geometry + semantic structure")
    pbc.add_argument("--eval-dir", required=True)

    pef = sub.add_parser("pillar-ef", help="stage 4: topology + watertightness")
    pef.add_argument("--eval-dir", required=True)

    pdd = sub.add_parser("pillar-d", help="stage 5: conditional validity")
    pdd.add_argument("--eval-dir", required=True)

    rb = sub.add_parser("retrieval", help="stage 9 (H): retrieval baseline + memorisation")
    rb.add_argument("--eval-dir", required=True)
    rb.add_argument("--shards-dir", default="tensorbuilding/shards")
    rb.add_argument("--manifest-path", default="tensorbuilding/shards/manifest.csv")
    rb.add_argument("--force", action="store_true",
                    help="run even if the GPU is busy (see _require_free_gpu)")

    hp = sub.add_parser("height-probe", help="stage 8 (D6): height extrapolation")
    hp.add_argument("--ckpt", required=True)
    hp.add_argument("--out", required=True)
    hp.add_argument("--heights", default="8,12,16,20,25,30")
    hp.add_argument("--n-per-height", type=int, default=24)
    hp.add_argument("--n-steps", type=int, default=50)
    hp.add_argument("--guidance", type=float, default=0.0)
    hp.add_argument("--seed0", type=int, default=5000)
    hp.add_argument("--base", default="gruenderzeit_res_gabled_pre1919",
                    help="condition to vary measured_height on")

    ab = sub.add_parser("cond-ablation", help="stage 7 (D4): condition ablation")
    ab.add_argument("--ckpt", required=True)
    ab.add_argument("--out", required=True, help="output directory")
    ab.add_argument("--n-per-condition", type=int, default=16)
    ab.add_argument("--n-steps", type=int, default=50)
    ab.add_argument("--guidance", type=float, default=0.0,
                    help="CFG weight; 0 by default so the §40.2 collapse "
                         "does not swamp the field effects")
    ab.add_argument("--seed0", type=int, default=1000)

    cs = sub.add_parser("cfg-sweep", help="stage 6 (D5): guidance weight sweep")
    cs.add_argument("--ckpt", required=True)
    cs.add_argument("--out", required=True, help="output directory")
    cs.add_argument("--weights", default="0,0.5,1.0,1.5,3.0",
                    help="comma-separated CFG weights")
    cs.add_argument("--n-per-condition", type=int, default=40)
    cs.add_argument("--n-steps", type=int, default=50)
    cs.add_argument("--seed0", type=int, default=1000,
                    help="same default as build-samples, so w=1.5 reproduces "
                         "the main sample set's noise draws")

    b = sub.add_parser("build-samples", help="stage 1: generate + cache the sample set")
    b.add_argument("--ckpt", required=True)
    b.add_argument("--out", required=True, help="output directory")
    b.add_argument("--n-per-condition", type=int, default=125,
                   help="generated samples per condition (4 conditions -> 4x this)")
    b.add_argument("--n-steps", type=int, default=50, help="DDIM steps")
    b.add_argument("--guidance", type=float, default=1.5, help="CFG weight")
    b.add_argument("--seed0", type=int, default=1000)
    b.add_argument("--shards-dir", default=None,
                   help="override the shards dir recorded in the checkpoint")
    b.add_argument("--manifest-path", default=None)

    a = ap.parse_args()
    if a.cmd == "build-samples":
        build_sample_set(a.ckpt, Path(a.out), a.n_per_condition,
                         a.n_steps, a.guidance, a.seed0,
                         a.shards_dir, a.manifest_path)
    elif a.cmd == "pillar-a":
        d = Path(a.eval_dir)
        cache = load_cache(d / "eval_samples.npz")
        print(f"[eval] cache: {len(cache['gen'])} generated, {len(cache['real'])} real")
        print(f"[eval] from : {cache['provenance']['ckpt']}")
        pillar_A(cache, d)
    elif a.cmd == "pillar-bc":
        d = Path(a.eval_dir)
        cache = load_cache(d / "eval_samples.npz")
        print(f"[eval] cache: {len(cache['gen'])} generated, {len(cache['real'])} real")
        pillar_BC(cache, d)
    elif a.cmd == "pillar-ef":
        d = Path(a.eval_dir)
        cache = load_cache(d / "eval_samples.npz")
        print(f"[eval] cache: {len(cache['gen'])} generated, {len(cache['real'])} real")
        pillar_EF(cache, d)
    elif a.cmd == "retrieval":
        d = Path(a.eval_dir)
        cache = load_cache(d / "eval_samples.npz")
        print(f"[eval] cache: {len(cache['gen'])} generated, {len(cache['real'])} real")
        retrieval_baseline(cache, d, a.shards_dir, a.manifest_path,
                           force=a.force)
    elif a.cmd == "height-probe":
        height_probe(a.ckpt, Path(a.out),
                     [float(x) for x in a.heights.split(",")],
                     a.n_per_height, a.n_steps, a.guidance, a.seed0, a.base)
    elif a.cmd == "cond-ablation":
        cond_ablation(a.ckpt, Path(a.out), a.n_per_condition,
                      a.n_steps, a.guidance, a.seed0)
    elif a.cmd == "cfg-sweep":
        cfg_sweep(a.ckpt, Path(a.out),
                  [float(x) for x in a.weights.split(",")],
                  a.n_per_condition, a.n_steps, a.seed0)
    elif a.cmd == "pillar-d":
        d = Path(a.eval_dir)
        cache = load_cache(d / "eval_samples.npz")
        print(f"[eval] cache: {len(cache['gen'])} generated, {len(cache['real'])} real")
        pillar_D(cache, d)


if __name__ == "__main__":
    _cli()
