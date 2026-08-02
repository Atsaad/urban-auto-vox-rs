"""
sample.py -- end-to-end DDIM sampler for the AutoVox diffusion model.

Wraps `DiffusionSchedule.ddim_step` (from diffusion.py) with:
  - Classifier-Free Guidance (CFG) via the two-forward-pass trick.
  - Fixed-seed reproducibility (so tracker progression samples are directly
    comparable across training checkpoints).
  - Batched sampling: K conditions x M seeds -> (K*M, 6, D, D, D) tensor.

Called from two places:
  1. train.py `_sample_hook` -- every `sample_every` iters, produces the
     fixed progression tracker.
  2. CLI (`python -m model.src.sample --ckpt ... --out ...`) for
     inference on a trained checkpoint after training.

Design note (why DDIM eta=0):
    Deterministic (eta=0) means the sample is entirely a function of
    (initial noise, condition, model weights). We hold the first two fixed
    across checkpoints; the only thing that changes is the model. That gives
    the clean "shape crystallises" progression view.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .conditions import ConditionEncoder
from .dataset import CATEGORICAL_COLS, CONTINUOUS_COLS, NULL_TOK
from .diffusion import DiffusionSchedule
from .unet3d import UNet3D


# ---- default progression conditions ----------------------------------------
# Four archetypes covering the diversity of the 5cities_balanced dataset.
# The dict values are RAW human-readable strings; sample.py will look them
# up in the vocabs stored on the loaded checkpoint.
DEFAULT_CONDITIONS: list[dict] = [
    {
        "name": "munich_res_gabled_1960s",
        "function_label": "residential",
        "roof_type_label": "gabled",
        "height_cluster": "medium",
        "ratio_cluster": "regular",
        "estimatedConstructionPeriod": "1960_1969",
        "constructionPeriodReliability": "high",
        "measured_height": 8.5,
        "length_to_width_ratio": 1.5,
        "storeys_above_ground": 2.0,
        "constructionPeriodConfidence": 0.9,
    },
    {
        "name": "gruenderzeit_res_gabled_pre1919",
        "function_label": "residential",
        "roof_type_label": "gabled",
        "height_cluster": "tall",
        "ratio_cluster": "regular",
        "estimatedConstructionPeriod": "before_1919",
        "constructionPeriodReliability": "high",
        "measured_height": 12.0,
        "length_to_width_ratio": 1.4,
        "storeys_above_ground": 3.0,
        "constructionPeriodConfidence": 0.95,
    },
    {
        "name": "modern_nonres_flat_2000s",
        "function_label": "non_residential",
        "roof_type_label": "flat",
        "height_cluster": "medium",
        "ratio_cluster": "regular",
        "estimatedConstructionPeriod": "2000_2009",
        "constructionPeriodReliability": "medium",
        "measured_height": 10.0,
        "length_to_width_ratio": 1.8,
        "storeys_above_ground": 3.0,
        "constructionPeriodConfidence": 0.6,
    },
    {
        "name": "rural_agricultural_monopitch_1980s",
        "function_label": "agricultural_shed",
        "roof_type_label": "monopitch",
        "height_cluster": "short",
        "ratio_cluster": "elongated",
        "estimatedConstructionPeriod": "1980_1989",
        "constructionPeriodReliability": "medium",
        "measured_height": 6.0,
        "length_to_width_ratio": 3.5,
        "storeys_above_ground": 1.0,
        "constructionPeriodConfidence": 0.5,
    },
]


# ---- ckpt loading ----------------------------------------------------------
def load_checkpoint(ckpt_path: str, device: torch.device) -> dict:
    """Load the .pt file written by train.py and rebuild the models.

    Returns: dict {unet, cond_enc, schedule, vocabs, continuous_stats, cfg}
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    cond_enc = ConditionEncoder(
        vocab_sizes={c: len(v) for c, v in ckpt["vocabs"].items()},
        categorical_cols=CATEGORICAL_COLS,
        n_continuous=len(CONTINUOUS_COLS),
        cond_dim=int(cfg.get("cond_dim", 256)),
    ).to(device)

    # Channel count is read from the checkpoint's own stem weight rather
    # than hardcoded: v4 targets have 6 channels, v5 adds a 7th `interior`
    # channel (claude.md §47). Hardcoding 6 here would load a v5
    # checkpoint into a 6-channel net and fail with a shape error --- or,
    # worse, silently sample the wrong shape.
    _stem = (ckpt.get("ema") or {}).get("shadow", ckpt["unet"])
    n_ch = int(_stem["stem.weight"].shape[1]) if "stem.weight" in _stem \
        else int(cfg.get("in_channels", 6))

    unet = UNet3D(
        in_channels=n_ch,
        base_channels=int(cfg.get("base_channels", 64)),
        channel_mults=tuple(cfg.get("channel_mults", (1, 2, 4, 4))),
        cond_dim=int(cfg.get("cond_dim", 256)),
        time_embed_dim=int(cfg.get("time_embed_dim", 256)),
        blocks_per_level=int(cfg.get("blocks_per_level", 2)),
    ).to(device)

    # Prefer EMA weights for sampling (cleaner outputs).
    if "ema" in ckpt and ckpt["ema"] is not None:
        unet.load_state_dict(ckpt["ema"]["shadow"])
    else:
        unet.load_state_dict(ckpt["unet"])
    cond_enc.load_state_dict(ckpt["cond_enc"])
    unet.eval(); cond_enc.eval()

    schedule = DiffusionSchedule.cosine(
        int(cfg.get("n_timesteps", 1000)), device=device
    )
    return {
        "unet": unet, "cond_enc": cond_enc, "schedule": schedule,
        "vocabs": ckpt["vocabs"],
        "continuous_stats": ckpt["continuous_stats"],
        "cfg": cfg,
        "n_channels": n_ch,
    }


# ---- condition encoding ---------------------------------------------------
def _encode_conditions(
    conditions: list[dict],
    vocabs: dict[str, dict],
    continuous_stats: dict[str, tuple[float, float]],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Turn user-facing dicts into (cat_ids, cont) tensors ready for the encoder."""
    B = len(conditions)
    cat_ids = {}
    for col in CATEGORICAL_COLS:
        ids = []
        for cond in conditions:
            raw = str(cond.get(col, "") or "").strip() or NULL_TOK
            ids.append(vocabs[col].get(raw, 0))
        cat_ids[col] = torch.tensor(ids, dtype=torch.long, device=device)

    cont = torch.zeros(B, len(CONTINUOUS_COLS), dtype=torch.float32, device=device)
    for k, col in enumerate(CONTINUOUS_COLS):
        m, s = continuous_stats.get(col, (0.0, 1.0))
        for i, cond in enumerate(conditions):
            v = cond.get(col)
            if v is None or v == "":
                continue
            try:
                fv = float(v)
                if np.isfinite(fv):
                    cont[i, k] = (fv - m) / (s if s else 1.0)
            except (ValueError, TypeError):
                pass
    return cat_ids, cont


# ---- DDIM subsampling schedule --------------------------------------------
def _ddim_timesteps(T: int, n_steps: int) -> list[int]:
    """Evenly spaced subsequence of [0, T-1], length n_steps, descending."""
    idx = np.linspace(0, T - 1, n_steps, dtype=np.int64)
    return list(reversed(idx.tolist()))


# ---- the sampler ----------------------------------------------------------
@torch.no_grad()
def sample(
    unet: UNet3D,
    cond_enc: ConditionEncoder,
    schedule: DiffusionSchedule,
    conditions: list[dict],
    vocabs: dict,
    continuous_stats: dict,
    seeds: Iterable[int] = (42,),
    n_steps: int = 50,
    guidance_scale: float = 1.5,
    device: torch.device | None = None,
    shape: tuple[int, ...] | None = None,   # defaults to the model's own channels
    dtype: torch.dtype = torch.float32,
    parameterization: str = "eps",
) -> tuple[torch.Tensor, list[dict]]:
    """Generate a batch of buildings for the (condition x seed) product grid.

    Returns:
        tensors: shape (K*M, C, D, D, D) float in [0,1]-ish argmax space
        meta:    K*M dicts recording {condition, seed}
    """
    if device is None:
        device = next(unet.parameters()).device
    if shape is None:
        # Take the channel count from the net itself so a 7-channel v5
        # model is not silently sampled at 6 channels.
        shape = (int(unet.stem.weight.shape[1]), 64, 64, 64)
    unet.eval(); cond_enc.eval()

    K = len(conditions)
    M = len(list(seeds)) if not isinstance(seeds, (list, tuple)) else len(seeds)
    seeds = list(seeds)

    # Build (K*M) row batch: same condition repeated M times back-to-back.
    all_conditions: list[dict] = []
    for cond in conditions:
        for _ in range(M):
            all_conditions.append(cond)
    B = K * M

    cat_ids, cont = _encode_conditions(
        all_conditions, vocabs, continuous_stats, device
    )

    # Sample the starting noise x_T. Fixed by seed per sample.
    x_shape = (B, *shape)
    x = torch.empty(x_shape, dtype=dtype, device=device)
    for i, cond in enumerate(conditions):
        for j, seed in enumerate(seeds):
            g = torch.Generator(device=device).manual_seed(int(seed))
            row = i * M + j
            x[row] = torch.randn(shape, dtype=dtype, device=device, generator=g)

    # Precompute conditional and unconditional condition vectors.
    drop_no  = torch.zeros(B, dtype=torch.bool, device=device)
    drop_yes = torch.ones (B, dtype=torch.bool, device=device)
    c_cond = cond_enc(cat_ids, cont, drop=drop_no)
    c_null = cond_enc(cat_ids, cont, drop=drop_yes)

    # DDIM subsampled trajectory: descending timesteps.
    ts = _ddim_timesteps(schedule.n_timesteps, n_steps)

    for step_idx, t_int in enumerate(ts):
        t_prev_int = ts[step_idx + 1] if step_idx + 1 < len(ts) else -1
        t      = torch.full((B,), t_int,      dtype=torch.long, device=device)
        t_prev = torch.full((B,), t_prev_int, dtype=torch.long, device=device)

        # CFG: two forward passes.
        # Under x0-parameterization the model output is logits over the
        # 6 classes. CFG is applied on the RAW LOGITS (linear space); the
        # softmax happens after CFG to produce the soft x0 prediction.
        out_cond = unet(x, t, c_cond)
        if guidance_scale != 0.0:
            out_null = unet(x, t, c_null)
            model_out = (1.0 + guidance_scale) * out_cond - guidance_scale * out_null
        else:
            model_out = out_cond

        if parameterization == "x0":
            x0_pred = torch.softmax(model_out, dim=1)
            x = schedule.ddim_step(x, t, t_prev, x0_pred=x0_pred, eta=0.0)
        else:
            x = schedule.ddim_step(x, t, t_prev, eps=model_out, eta=0.0)

    # Meta index (for saving alongside the tensor).
    meta = []
    for cond in conditions:
        for seed in seeds:
            m = dict(cond)
            m["seed"] = int(seed)
            meta.append(m)
    return x, meta


# ---- utility: convert continuous 6-channel prediction to hard argmax ------
def to_hard_onehot(x: torch.Tensor) -> torch.Tensor:
    """Argmax over channel dim -> hard one-hot uint8. Preserves shape."""
    idx = x.argmax(dim=1)                                    # (B, D, D, D)
    hard = torch.nn.functional.one_hot(idx, num_classes=x.shape[1])   # (B, D, D, D, C)
    hard = hard.permute(0, 4, 1, 2, 3).contiguous().to(torch.uint8)
    return hard


# ---- CLI ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="DDIM sampler for AutoVox diffusion.")
    ap.add_argument("--ckpt", required=True, help="Path to a .pt checkpoint from train.py")
    ap.add_argument("--out",  required=True, help="Output directory")
    ap.add_argument("--conditions", default=None,
                    help="JSON file with a list of condition dicts. "
                         "Defaults to the four archetypes in DEFAULT_CONDITIONS.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45])
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--guidance", type=float, default=1.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sample] loading {args.ckpt} on {device}")
    ckpt = load_checkpoint(args.ckpt, device)

    conditions = DEFAULT_CONDITIONS
    if args.conditions:
        with open(args.conditions) as f:
            conditions = json.load(f)

    print(f"[sample] generating: {len(conditions)} conditions x {len(args.seeds)} seeds "
          f"= {len(conditions) * len(args.seeds)} buildings")
    x, meta = sample(
        ckpt["unet"], ckpt["cond_enc"], ckpt["schedule"],
        conditions, ckpt["vocabs"], ckpt["continuous_stats"],
        seeds=args.seeds, n_steps=args.n_steps,
        guidance_scale=args.guidance, device=device,
        parameterization=str(ckpt["cfg"].get("parameterization", "eps")),
    )
    hard = to_hard_onehot(x)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"tensors": hard.cpu(), "meta": meta,
                "n_steps": args.n_steps, "guidance_scale": args.guidance},
               out / "samples.pt")
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"[sample] wrote {out / 'samples.pt'} ({hard.shape}) + meta.json")


if __name__ == "__main__":
    main()
