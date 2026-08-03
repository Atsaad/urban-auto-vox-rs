"""
train.py — DDPM training loop for the AutoVox semantic voxel diffusion model.

Phase A (smoke):  tiny U-Net, small subset, verify the loss decreases.
Phase B onwards: scale up via the YAML config -- no code change needed.

Usage:
    model/.venv/bin/python -m model.src.train --config model/configs/phase_a.yaml
    model/.venv/bin/python -m model.src.train --config model/configs/phase_a.yaml --override iters=500 batch_size=4

All paths in the YAML are interpreted relative to the repo root (the cwd at
launch time), matching how `build_tensors.py` and `chunk-process.sh` work.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader, Subset

from .conditions import ConditionEncoder
from .dataset import (CATEGORICAL_COLS, CONTINUOUS_COLS,
                      Building3DDataset, collate_batch)
from .diffusion import DiffusionSchedule
from .ema import EMA
from .render import render_grid
from .sample import DEFAULT_CONDITIONS, sample as ddim_sample, to_hard_onehot
from .unet3d import UNet3D


# ---- config -------------------------------------------------------------
def _load_config(path: str, overrides: list[str]) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for o in overrides:
        if "=" not in o:
            raise ValueError(f"--override expects key=value, got {o}")
        k, v = o.split("=", 1)
        # naive cast: try int, then float, then leave as str
        for caster in (int, float):
            try:
                v = caster(v); break
            except ValueError:
                pass
        cfg[k] = v
    return cfg


# ---- helpers ------------------------------------------------------------
def _seed_all(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _human_n(n: int) -> str:
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


# ---- main ---------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[],
                    help="key=value overrides applied AFTER yaml load")
    ap.add_argument("--resume", default="",
                    help="resume from a checkpoint: a path to a .pt file, or "
                         "'auto' to pick the highest-numbered ckpt_*.pt in "
                         "out_dir. Restores model, EMA, optimiser, iteration "
                         "counter and the loss/grad-norm history.")
    args = ap.parse_args()

    cfg = _load_config(args.config, args.override)
    _seed_all(int(cfg.get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device} torch={torch.__version__}")

    # -- Dataset --------------------------------------------------------
    ds = Building3DDataset(
        shards_dir=cfg["shards_dir"],
        manifest_path=cfg.get("manifest_path"),
    )
    print(f"[train] dataset: {len(ds):,} samples, tensor_shape={ds.tensor_shape}")
    print(f"[train] vocab_sizes: {ds.vocab_sizes}")
    if (n_sub := int(cfg.get("subset", 0))) > 0 and n_sub < len(ds):
        ds = Subset(ds, list(range(n_sub)))
        print(f"[train] using subset of {n_sub} samples")

    loader = DataLoader(
        ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 2)),
        collate_fn=collate_batch,
        pin_memory=(device.type == "cuda"),
        persistent_workers=int(cfg.get("num_workers", 2)) > 0,
        drop_last=True,
    )

    # -- Model ----------------------------------------------------------
    raw_ds = ds.dataset if isinstance(ds, Subset) else ds   # to read vocabs
    cond_enc = ConditionEncoder(
        vocab_sizes=raw_ds.vocab_sizes,
        categorical_cols=CATEGORICAL_COLS,
        n_continuous=len(CONTINUOUS_COLS),
        cond_dim=int(cfg.get("cond_dim", 256)),
    ).to(device)

    base_channels = int(cfg.get("base_channels", 32))
    ch_mults = tuple(cfg.get("channel_mults", (1, 2, 4, 8)))
    unet = UNet3D(
        in_channels=raw_ds.tensor_shape[0],
        base_channels=base_channels,
        channel_mults=ch_mults,
        cond_dim=int(cfg.get("cond_dim", 256)),
        time_embed_dim=int(cfg.get("time_embed_dim", 256)),
        blocks_per_level=int(cfg.get("blocks_per_level", 1)),
    ).to(device)

    n_params = sum(p.numel() for p in unet.parameters()) + sum(
        p.numel() for p in cond_enc.parameters()
    )
    print(f"[train] params: {_human_n(n_params)} (UNet + condition encoder)")

    # -- Schedule / optim / amp ----------------------------------------
    schedule = DiffusionSchedule.cosine(int(cfg.get("n_timesteps", 1000)), device=device)
    optim = torch.optim.AdamW(
        list(unet.parameters()) + list(cond_enc.parameters()),
        lr=float(cfg.get("lr", 2e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    ema = EMA(unet, decay=float(cfg.get("ema_decay", 0.999)))

    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    amp_ctx = (lambda: torch.autocast(device_type="cuda", dtype=amp_dtype)) if use_amp else nullcontext

    p_drop_cond = float(cfg.get("p_drop_cond", 0.1))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    # Foreground weighting of the MSE loss (see diffusion.p_loss). 0 = disabled.
    fg_loss_weight = float(cfg.get("fg_loss_weight", 0.0))
    # Parameterization: "eps" (default, DDPM MSE) or "x0" (Improved-DDPM, CE loss).
    parameterization = str(cfg.get("parameterization", "eps"))
    # v6 topology terms; absent or zero -> identical to v4/v5
    topo_cfg = {k: cfg[c] for k, c in
                (("w_cldice", "topo_w_cldice"), ("w_leak", "topo_w_leak"),
                 ("skel_iters", "topo_skel_iters"),
                 ("flood_steps", "topo_flood_steps")) if c in cfg}
    if not any(topo_cfg.get(k, 0) for k in ("w_cldice", "w_leak")):
        topo_cfg = None
    if parameterization not in ("eps", "x0"):
        raise ValueError(f"unknown parameterization: {parameterization!r}")
    print(f"[train] parameterization: {parameterization}"
          + (f"  (fg_weight={fg_loss_weight})" if fg_loss_weight > 0 and parameterization == "eps" else ""))

    # LR warmup:  linear 0 -> peak_lr over `warmup_iters` iterations.
    # Prevents the early gradient explosion pattern observed in the first
    # Phase-B run (loss spikes at it=1500 / 5100 / 9500, then permanent
    # collapse). See understanding_guide §11.9 for the post-mortem.
    peak_lr = float(cfg.get("lr", 2e-4))
    warmup_iters = int(cfg.get("warmup_iters", 0))

    def _lr_for_step(step: int) -> float:
        if warmup_iters <= 0 or step >= warmup_iters:
            return peak_lr
        return peak_lr * (step + 1) / warmup_iters

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # -- Training loop --------------------------------------------------
    n_iters = int(cfg["iters"])
    log_every = int(cfg.get("log_every", 25))
    save_every = int(cfg.get("save_every", 1000))
    sample_every = int(cfg.get("sample_every", 0))
    sample_seeds = list(cfg.get("sample_seeds", [42, 43, 44, 45]))
    sample_steps = int(cfg.get("sample_n_steps", 50))
    sample_guidance = float(cfg.get("sample_guidance", 1.5))
    samples_dir = out_dir / "samples"
    if sample_every > 0:
        samples_dir.mkdir(parents=True, exist_ok=True)
        print(f"[train] progression tracker: every {sample_every} iters, "
              f"{len(DEFAULT_CONDITIONS)} conditions x {len(sample_seeds)} seeds "
              f"= {len(DEFAULT_CONDITIONS) * len(sample_seeds)} samples per checkpoint")
    losses: list[float] = []
    grad_norms: list[float] = []
    start_it = 0

    # -- Resume ---------------------------------------------------------
    # Restores model / EMA / optimiser / iteration counter AND the
    # loss+grad-norm history. The history matters: losses.json is written
    # from the in-memory list, so resuming without restoring it would
    # truncate the curve to post-resume iterations only and quietly ruin
    # the training figures.
    #
    # NOT restored (deliberately): the DataLoader shuffle position and the
    # RNG stream. A resumed run therefore sees a different sample order
    # than an uninterrupted one would. For diffusion training on 1.15 M
    # samples this is immaterial, but it does mean a resumed run is not
    # bit-identical to an unbroken one — state that when reporting.
    if args.resume:
        if args.resume == "auto":
            cands = sorted(out_dir.glob("ckpt_*.pt"),
                           key=lambda p: int(p.stem.split("_")[1]))
            if not cands:
                raise SystemExit(f"--resume auto: no ckpt_*.pt found in {out_dir}")
            resume_path = cands[-1]
        else:
            resume_path = Path(args.resume)
            if not resume_path.is_file():
                raise SystemExit(f"--resume: not a file: {resume_path}")

        print(f"[train] resuming from {resume_path}")
        ck = torch.load(resume_path, map_location=device, weights_only=False)

        # Guard: a checkpoint from a different architecture will either
        # fail cryptically inside load_state_dict or, worse, load partially.
        # Compare the shape-determining keys and refuse early if they differ.
        prev_cfg = ck.get("config", {})
        for k in ("base_channels", "channel_mults", "cond_dim",
                  "time_embed_dim", "blocks_per_level"):
            old, new = prev_cfg.get(k), cfg.get(k)
            if old is not None and str(old) != str(new):
                raise SystemExit(
                    f"--resume: architecture mismatch on '{k}': checkpoint has "
                    f"{old!r}, config has {new!r}. Resuming would corrupt the "
                    f"model. Use a matching config or start a fresh run.")

        unet.load_state_dict(ck["unet"])
        cond_enc.load_state_dict(ck["cond_enc"])
        ema.load_state_dict(ck["ema"])
        optim.load_state_dict(ck["optim"])
        start_it = int(ck.get("iter", 0))
        losses = list(ck.get("losses", []))
        grad_norms = list(ck.get("grad_norms", []))

        if not losses:
            # Checkpoint predates history-in-checkpoint; fall back to the
            # sidecar files so the curve is still continuous.
            for name, tgt in (("losses.json", losses),
                              ("grad_norms.json", grad_norms)):
                p = out_dir / name
                if p.is_file():
                    tgt.extend(json.loads(p.read_text())[:start_it])
            print(f"[train] (history recovered from sidecar json)")

        print(f"[train] resumed at iter {start_it:,}/{n_iters:,} "
              f"— {len(losses):,} loss points restored")
        if start_it >= n_iters:
            raise SystemExit(
                f"--resume: checkpoint is already at iter {start_it:,} >= "
                f"iters {n_iters:,}. Nothing to do; raise `iters` to continue.")

    def _dump_history() -> None:
        (out_dir / "losses.json").write_text(json.dumps(losses))
        (out_dir / "grad_norms.json").write_text(json.dumps(grad_norms))

    def _sample_hook(step: int) -> None:
        """Generate fixed-seed progression samples with the EMA weights."""
        t0 = time.time()
        x, meta = ddim_sample(
            ema.shadow, cond_enc, schedule,
            conditions=DEFAULT_CONDITIONS,
            vocabs=raw_ds.vocabs,
            continuous_stats=raw_ds.continuous_stats,
            seeds=sample_seeds,
            n_steps=sample_steps,
            guidance_scale=sample_guidance,
            device=device,
            parameterization=parameterization,
        )
        hard = to_hard_onehot(x).cpu().numpy()
        step_dir = samples_dir / f"step_{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        # PNG grid: rows=conditions, cols=seeds (K x M).
        K, M = len(DEFAULT_CONDITIONS), len(sample_seeds)
        titles = [f"{m['name']}\nseed={m['seed']}" for m in meta]
        render_grid(list(hard), titles=titles,
                    out_path=str(step_dir / "grid.png"),
                    cols=M, suptitle=f"{Path(cfg['out_dir']).name} progression @ step {step:,}")
        # Also stash the raw tensors for future re-analysis.
        import numpy as np_
        np_.savez_compressed(step_dir / "samples.npz",
                             tensors=hard, meta=json.dumps(meta, default=str))
        print(f"[train] progression sample @ step {step:,} -> {step_dir} "
              f"({time.time()-t0:.1f}s)", flush=True)

    t_start = time.time()
    it = start_it
    loader_iter = iter(loader)
    while it < n_iters:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        x0 = batch["tensor"].to(device, non_blocking=True)        # (B,C,D,D,D)
        cont = batch["cont"].to(device, non_blocking=True)
        cat = {c: batch[f"cat_{c}"].to(device, non_blocking=True)
               for c in CATEGORICAL_COLS}

        B = x0.shape[0]
        t = torch.randint(0, schedule.n_timesteps, (B,), device=device)
        drop = (torch.rand(B, device=device) < p_drop_cond)

        with amp_ctx():
            cond_vec = cond_enc(cat, cont, drop=drop)

            def model_call(x_t, t_, drop_):
                # `drop_` already baked into cond_vec; UNet only needs (x, t, c).
                return unet(x_t, t_, cond_vec)

            loss = schedule.p_loss(model_call, x0, t, drop_mask=drop,
                                   topo=topo_cfg,
                                   fg_weight=fg_loss_weight,
                                   parameterization=parameterization)

        # Apply LR warmup for this step.
        lr_now = _lr_for_step(it)
        for pg in optim.param_groups:
            pg["lr"] = lr_now

        optim.zero_grad(set_to_none=True)
        loss.backward()
        # Capture pre-clip grad norms so we can monitor gradient stability
        # and catch the early explosions that killed Phase-B v1.
        gn_unet = torch.nn.utils.clip_grad_norm_(unet.parameters(), grad_clip)
        gn_cond = torch.nn.utils.clip_grad_norm_(cond_enc.parameters(), grad_clip)
        gn_unet_f = float(gn_unet)
        gn_cond_f = float(gn_cond)
        optim.step()
        ema.update(unet)

        losses.append(float(loss.detach().cpu()))
        grad_norms.append(gn_unet_f)
        it += 1

        # Fail-fast if gradients become NaN/Inf: training has diverged and
        # every further optimiser step corrupts the AdamW state further.
        if not math.isfinite(gn_unet_f) or not math.isfinite(gn_cond_f):
            print(f"[train] FATAL: non-finite grad norm at it={it} "
                  f"(unet={gn_unet_f}, cond={gn_cond_f}); halting.",
                  flush=True)
            break

        if it % log_every == 0 or it == 1:
            elapsed = time.time() - t_start
            recent = losses[-log_every:]
            avg = sum(recent) / max(1, len(recent))
            gn_recent = grad_norms[-log_every:]
            gn_avg = sum(gn_recent) / max(1, len(gn_recent))
            print(f"[train] it={it:>5d}/{n_iters}  loss={avg:.4f}  "
                  f"recent_min={min(recent):.4f}  "
                  f"gn={gn_avg:.3f} (max={max(gn_recent):.2f})  "
                  f"lr={lr_now:.2e}  "
                  f"it/s={it/elapsed:.2f}  elapsed={elapsed:.0f}s",
                  flush=True)

        if save_every > 0 and (it % save_every == 0 or it == n_iters):
            ckpt = {
                "iter": it,
                "unet": unet.state_dict(),
                "cond_enc": cond_enc.state_dict(),
                "ema": ema.state_dict(),
                "optim": optim.state_dict(),
                "config": cfg,
                "vocabs": raw_ds.vocabs,
                "continuous_stats": raw_ds.continuous_stats,
                # History travels WITH the checkpoint so that --resume
                # reconstructs a continuous curve. ~8 bytes/iter, so even a
                # 200 K-iter run adds only ~3 MB to the file.
                "losses": losses,
                "grad_norms": grad_norms,
            }
            ckpt_path = out_dir / f"ckpt_{it:06d}.pt"
            torch.save(ckpt, ckpt_path)
            # Mirror the history to json on every save, not just at the end,
            # so an aborted run still leaves a usable curve on disk.
            _dump_history()
            print(f"[train] saved {ckpt_path}")

        # Progression tracker -- generate fixed-seed samples at every
        # `sample_every` iterations (see §26 in claude.md).
        if sample_every > 0 and (it % sample_every == 0 or it == n_iters):
            _sample_hook(it)

    # Final loss + grad-norm log dump.
    _dump_history()
    _n = min(log_every, len(losses)) or 1
    print(f"[train] done in {time.time() - t_start:.1f}s — final mean loss "
          f"(last {_n}): {sum(losses[-_n:]) / _n:.4f}")


if __name__ == "__main__":
    main()
