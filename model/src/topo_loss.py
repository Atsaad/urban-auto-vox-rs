"""Differentiable topology terms for v6.

Two defects remain after v5, and they are not the same defect:

  Pillar E  84.0 % single-component (real 99.4 %)  -- CONNECTIVITY
  Pillar F  40.3 % watertight       (real 92.1 %)  -- CLOSURE

The distinction matters for the loss. A shell with one missing voxel is
still fully connected -- b0 = 1 -- but no longer watertight, b2 = 0. So a
connectivity loss cannot see the larger defect, and a closure loss does
not directly address fragmentation. v6 therefore carries one term for
each.

`soft_cldice` follows Shit et al., CVPR 2021 (clDice), built from soft
morphology so it is differentiable end to end. It targets connectivity.

`leak_loss` is the closure term and is specific to this problem. It
floods the grid exterior by iterated soft dilation against predicted free
space, then penalises whatever exterior mass reaches voxels the model
labelled `interior`. A sealed shell blocks the flood; a holed one does
not. This is the differentiable surrogate for the flood-fill test that
Pillar F performs exactly, proposed in the thesis conclusion.

Both consume the softmax probabilities, never the argmax, so gradients
flow.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


# ---------------------------------------------------------------- morphology
def soft_dilate6(x: torch.Tensor) -> torch.Tensor:
    """6-connected soft dilation: max over face neighbours and self.

    Three axis-aligned pools rather than one 3x3x3 kernel. A cubic kernel
    is 26-connected, which lets the flood travel diagonally between
    voxels sharing only an edge -- exactly the mismatch that made the v5
    data prep report two thirds of real buildings as open until it was
    caught (claude.md 49.1). The evaluation defines both connectivity and
    watertightness on 6-neighbourhoods, so the loss must agree.
    """
    return torch.maximum(
        torch.maximum(F.max_pool3d(x, (3, 1, 1), 1, (1, 0, 0)),
                      F.max_pool3d(x, (1, 3, 1), 1, (0, 1, 0))),
        F.max_pool3d(x, (1, 1, 3), 1, (0, 0, 1)))


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Soft erosion = dual of dilation."""
    return -soft_dilate6(-x)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    return soft_dilate6(soft_erode(x))


def soft_skel(x: torch.Tensor, iters: int = 5) -> torch.Tensor:
    """Iterative soft skeletonisation (Shit et al. 2021, Alg. 1)."""
    skel = F.relu(x - soft_open(x))
    for _ in range(iters):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        # union without double-counting the overlap
        skel = skel + F.relu(delta - skel * delta)
    return skel


# ---------------------------------------------------------------- the terms
def soft_cldice(pred: torch.Tensor, true: torch.Tensor,
                iters: int = 5) -> torch.Tensor:
    """1 - clDice. Connectivity term; expects (B,1,D,D,D) in [0,1]."""
    sp, st = soft_skel(pred, iters), soft_skel(true, iters)
    tprec = (torch.sum(sp * true, dim=(1, 2, 3, 4)) + EPS) / \
            (torch.sum(sp, dim=(1, 2, 3, 4)) + EPS)
    tsens = (torch.sum(st * pred, dim=(1, 2, 3, 4)) + EPS) / \
            (torch.sum(st, dim=(1, 2, 3, 4)) + EPS)
    return (1.0 - 2.0 * tprec * tsens / (tprec + tsens + EPS)).mean()


def leak_loss(p_shell: torch.Tensor, p_interior: torch.Tensor,
              steps: int = 40,
              mask: torch.Tensor | None = None) -> torch.Tensor:
    """Closure term: how much exterior reaches the claimed interior.

    Floods inward from the grid boundary and reports the flood's overlap
    with voxels the model labelled `interior`. Zero when the shell seals;
    large when it leaks.

    The flood is blocked by the SHELL ONLY -- channels 1..5 -- and not by
    the interior class. Blocking on all non-empty voxels looks right and
    is wrong: the interior blob then blocks the flood itself, so punching
    a hole in the shell changes nothing and the loss reads zero for a
    holed building exactly as for a sealed one. Verified: with the shell
    as blocker the term separates them; with `1 - p_empty` it does not.

    `steps` must be large enough for the flood to cross the empty space
    and enter through a hole. The grid is 64 wide and a building sits
    near its centre, so 40 steps saturates; fewer risks reporting zero
    leakage merely because the flood never arrived.
    """
    free = (1.0 - p_shell).clamp(0, 1)

    # seed: the six faces of the grid, which are outside every building
    flood = torch.zeros_like(free)
    flood[:, :, 0, :, :] = 1.0; flood[:, :, -1, :, :] = 1.0
    flood[:, :, :, 0, :] = 1.0; flood[:, :, :, -1, :] = 1.0
    flood[:, :, :, :, 0] = 1.0; flood[:, :, :, :, -1] = 1.0
    flood = torch.minimum(flood, free)

    for _ in range(steps):
        # min, not multiply. Multiplying compounds: with diffuse early
        # predictions free ~= 0.29 everywhere, and 0.29^32 is zero, so the
        # flood dies before travelling and the term has no gradient at all
        # -- measured inert even at weight 100. min() lets free space cap
        # the flood without attenuating it step over step.
        flood = torch.minimum(soft_dilate6(flood), free)

    # exterior that has reached voxels the model calls interior
    leaked = flood * p_interior
    per_sample = (leaked.sum(dim=(1, 2, 3, 4))
                  / (p_interior.sum(dim=(1, 2, 3, 4)) + EPS))

    # v8: closure is enforced only on buildings whose REAL shell was already
    # watertight. Averaging over the masked-in samples only -- not over the
    # batch -- keeps the term's scale independent of how many buildings the
    # gate happens to admit in a given batch, so the calibrated weight still
    # means what it meant in v6/v7.
    if mask is None:
        return per_sample.mean()
    m = mask.to(per_sample.dtype)
    n = m.sum()
    if n < 1:
        # No gated-in sample in this batch. Return a real zero that still
        # carries the graph, so AMP and DDP see a consistent set of grads.
        return per_sample.sum() * 0.0
    return (per_sample * m).sum() / n


def topology_loss(logits: torch.Tensor, x0: torch.Tensor,
                  w_cldice: float = 0.0, w_leak: float = 0.0,
                  skel_iters: int = 5, flood_steps: int = 40,
                  closure_mask: torch.Tensor | None = None
                  ) -> tuple[torch.Tensor, dict]:
    """Combined term added to the cross-entropy. Returns (loss, parts).

    Channel layout is the v5 seven-class target: 0 empty, 1..5 boundary
    surfaces, 6 interior. `solid` is everything that is not empty, which
    is what the flood must be blocked by -- a building's interior blocks
    the exterior just as its walls do.
    """
    if w_cldice <= 0 and w_leak <= 0:
        z = logits.sum() * 0.0
        return z, {}

    p = torch.softmax(logits, dim=1)
    # connectivity is a property of everything the building occupies;
    # closure is a property of the shell alone (see leak_loss)
    p_solid = (1.0 - p[:, 0:1]).clamp(0, 1)
    p_shell = p[:, 1:6].sum(dim=1, keepdim=True).clamp(0, 1)
    parts: dict[str, float] = {}
    total = logits.sum() * 0.0

    if w_cldice > 0:
        t_solid = (1.0 - x0[:, 0:1]).clamp(0, 1).float()
        l = soft_cldice(p_solid, t_solid, skel_iters)
        total = total + w_cldice * l
        parts["cldice"] = float(l.detach())

    if w_leak > 0 and logits.shape[1] > 6:
        # clDice is NOT gated: connectivity is expected of every building,
        # open or closed. Only closure is conditional on the target.
        l = leak_loss(p_shell, p[:, 6:7], flood_steps, mask=closure_mask)
        total = total + w_leak * l
        parts["leak"] = float(l.detach())
        if closure_mask is not None:
            parts["leak_n"] = float(closure_mask.sum())

    return total, parts
