"""
Diffusion schedule + DDPM training loss + DDIM sampler.

Schedule: cosine (Nichol & Dhariwal 2021, Improved Denoising Diffusion
Probabilistic Models). 1000 steps by default.

Forward (q):
    x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε,   ε ~ N(0, I)

Training loss (ε-prediction, claude.md §9):
    L = E_{t, x_0, ε} || ε - ε_θ(x_t, t, c) ||^2

Classifier-free guidance:
    The drop mask is applied by the ConditionEncoder. This module is
    architecture-agnostic; it just routes a `drop` tensor through to the
    encoder via the model wrapper.

DDIM sampling (η=0 deterministic, claude.md §10):
    x_{t-1} = sqrt(ᾱ_{t-1}) x̂_0 + sqrt(1 - ᾱ_{t-1}) ε̂_t
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---- cosine schedule -----------------------------------------------------
def cosine_alpha_bar(t_frac: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """ᾱ(t) for t ∈ [0, 1].  Improved-DDPM Eq. (17)."""
    return torch.cos(((t_frac + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2


def make_cosine_betas(n_timesteps: int = 1000, max_beta: float = 0.999) -> torch.Tensor:
    """β_t schedule from the cosine ᾱ_t."""
    t = torch.arange(n_timesteps + 1, dtype=torch.float64) / n_timesteps
    ab = cosine_alpha_bar(t)
    betas = 1.0 - (ab[1:] / ab[:-1])
    return betas.clamp(max=max_beta).to(torch.float32)


# ---- main schedule object ------------------------------------------------
@dataclass
class DiffusionSchedule:
    """All the per-step precomputed quantities, stored on a target device."""

    n_timesteps: int
    betas: torch.Tensor             # (T,)
    alphas: torch.Tensor            # (T,)
    alpha_bars: torch.Tensor        # (T,)
    sqrt_alpha_bars: torch.Tensor   # (T,)
    sqrt_one_minus_alpha_bars: torch.Tensor  # (T,)

    @classmethod
    def cosine(cls, n_timesteps: int = 1000, device: torch.device = torch.device("cpu")):
        betas = make_cosine_betas(n_timesteps).to(device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        return cls(
            n_timesteps=n_timesteps,
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            sqrt_alpha_bars=torch.sqrt(alpha_bars),
            sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
        )

    # ---------------------------------------------------------------- forward
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε."""
        a = _gather(self.sqrt_alpha_bars, t, x0.ndim)
        b = _gather(self.sqrt_one_minus_alpha_bars, t, x0.ndim)
        return a * x0 + b * noise

    # ---------------------------------------------------------------- training loss
    def p_loss(
        self,
        model_call,                       # callable: (x_t, t, drop_mask) -> model output
        x0: torch.Tensor,
        t: torch.Tensor,
        drop_mask: torch.Tensor | None,
        fg_weight: float = 0.0,
        parameterization: str = "eps",
        topo: dict | None = None,
    ) -> torch.Tensor:
        """DDPM training loss.

        Two parameterizations supported:

        - "eps" (default, Ho et al. 2020 DDPM):
            Model output = ε̂  (predicted noise, R^{B x C x D x D x D}).
            Loss = MSE(ε̂, ε) with optional foreground weighting via `fg_weight`.
            See understanding_guide §11.10 for the fg-weighting rationale.

        - "x0" (Improved DDPM / v3+ pivot for sparse voxel data):
            Model output = logits over the C classes (raw, un-normalised).
            Loss = softmax cross-entropy against x0.argmax(dim=1).
            Naturally handles the 99.7 %-empty class distribution because the
            softmax normalisation gives the model a free bias toward the
            training marginal (mostly "empty"). See §11.11 for the derivation.

        For eps-param, `fg_weight = 0` keeps loss byte-identical with the
        original unweighted DDPM MSE.
        For x0-param, fg_weight is ignored (CE handles imbalance intrinsically).
        """
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        model_out = model_call(x_t, t, drop_mask)

        if parameterization == "x0":
            # Model output is (B, C, D, D, D) logits.  Cross-entropy expects
            # target as class-index tensor (B, D, D, D).
            target = x0.argmax(dim=1)                       # (B, D, D, D)
            loss = F.cross_entropy(model_out, target)

            # v6: optional topology terms. Cross-entropy is per-voxel and
            # cannot express connectivity or closure -- established by v5,
            # where a dense interior target improved watertightness 2.4x
            # and then stopped. These act on the predicted field as a
            # whole. Off by default, so v4/v5 behaviour is unchanged.
            if topo:
                from .topo_loss import topology_loss
                extra, parts = topology_loss(model_out, x0, **topo)
                loss = loss + extra
                self.last_topo_parts = parts
            return loss

        # -- eps parameterization (default) ----------------------------------
        eps_pred = model_out
        if fg_weight <= 0.0:
            return F.mse_loss(eps_pred, noise)

        sq_err = (eps_pred - noise) ** 2                     # (B, C, D, D, D)
        # Foreground mask from x0's argmax, broadcast across the C channels
        # so all 6 output channels of a foreground voxel get the same weight.
        fg_mask = (x0.argmax(dim=1, keepdim=True) != 0).float()  # (B, 1, D, D, D)
        w = 1.0 + fg_weight * fg_mask                        # (B, 1, D, D, D)
        # Normalised weighted mean:  keeps the loss scale ~[0, 1] so plateau
        # heuristics from unweighted MSE still apply.
        return (sq_err * w).sum() / (w.expand_as(sq_err).sum())

    # ---------------------------------------------------------------- DDIM step
    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,            # (B,) current timestep
        t_prev: torch.Tensor,       # (B,) previous timestep (smaller), or -1
        eps: torch.Tensor | None = None,       # for eps-parameterization
        x0_pred: torch.Tensor | None = None,   # for x0-parameterization
        eta: float = 0.0,
    ) -> torch.Tensor:
        """Deterministic DDIM (η=0) update.

        Provide either `eps` (ε-param) or `x0_pred` (x0-param), not both.
        Under the hood the DDIM step needs BOTH an ε and an x0 estimate;
        if we're given one we compute the other from the forward-process
        relationship x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε.
        """
        if (eps is None) == (x0_pred is None):
            raise ValueError("Pass exactly one of eps / x0_pred to ddim_step")

        ab_t = _gather(self.alpha_bars, t, x_t.ndim)
        # ab_prev: 1.0 where t_prev < 0 (final step)
        ab_prev = torch.where(
            t_prev[(...,) + (None,) * (x_t.ndim - 1)] >= 0,
            _gather(self.alpha_bars, t_prev.clamp(min=0), x_t.ndim),
            torch.ones_like(ab_t),
        )
        sqrt_ab_t = torch.sqrt(ab_t).clamp(min=1e-12)
        sqrt_one_minus_ab_t = torch.sqrt(1.0 - ab_t).clamp(min=1e-12)

        if x0_pred is None:
            # eps-param path: derive x0 from eps.
            x0_pred = (x_t - sqrt_one_minus_ab_t * eps) / sqrt_ab_t
        else:
            # x0-param path: derive eps from the given x0 prediction.
            eps = (x_t - sqrt_ab_t * x0_pred) / sqrt_one_minus_ab_t

        # Direction pointing to x_t
        sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t).clamp(min=1e-12) * (1 - ab_t / ab_prev))
        dir_xt = torch.sqrt((1.0 - ab_prev - sigma ** 2).clamp(min=0.0)) * eps
        x_prev = torch.sqrt(ab_prev) * x0_pred + dir_xt
        if eta > 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)
        return x_prev


# ---- helper -------------------------------------------------------------
def _gather(buffer_1d: torch.Tensor, t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Pick `buffer_1d[t]` and broadcast to `ndim` dimensions like (B,1,1,1,1)."""
    out = buffer_1d.to(t.device)[t]                       # (B,)
    return out.view(-1, *([1] * (ndim - 1)))
