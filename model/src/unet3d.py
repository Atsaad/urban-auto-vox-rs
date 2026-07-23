"""
UNet3D — 3D U-Net with FiLM conditioning for DDPM on (C, D, D, D) tensors.

Architecture (claude.md §9 / §10):
- Channels-first 3D conv backbone.
- 4 resolution levels at D=64: 64 -> 32 -> 16 -> 8, with a bottleneck at 8.
- Per level: 1 ResBlock3D (sufficient for the Phase A smoke; trivially scaled
  up by passing more `blocks_per_level`).
- Time embedding: sinusoidal -> MLP, broadcast through every ResBlock.
- Conditioning: a (B, cond_dim) vector from ConditionEncoder is summed with
  the time embedding and used to FiLM every ResBlock.
- Downsample = stride-2 Conv3d ; upsample = trilinear + Conv3d (cheap, no
  checkerboard artifacts on the empty / surface-only target distribution).

Output channels = input channels (DDPM ε-prediction has the same shape as x).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- time embedding ------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    """Standard Transformer-style sinusoidal embedding of integer timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:  # t: (B,) long or float
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / max(1, half - 1)
        )                                                # (half,)
        args = t.float()[:, None] * freqs[None, :]      # (B, half)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)


# ---- FiLM ----------------------------------------------------------------
class FiLM(nn.Module):
    """h' = (1 + γ(c)) ⊙ h + β(c), per channel.

    The "1 + γ" parameterisation matches Improved-DDPM and is more stable at
    init than the raw γ form (γ starts near 0 -> identity).
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gb = self.proj(c)                                # (B, 2C)
        g, b = gb.chunk(2, dim=-1)                       # (B, C), (B, C)
        g = g.view(g.shape[0], -1, 1, 1, 1)
        b = b.view(b.shape[0], -1, 1, 1, 1)
        return h * (1.0 + g) + b


# ---- residual block ------------------------------------------------------
class ResBlock3D(nn.Module):
    """GroupNorm -> SiLU -> Conv3d -> FiLM -> GroupNorm -> SiLU -> Conv3d (+ skip).

    Robust to in_ch != out_ch via a 1x1x1 skip projection.
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, groups: int = 8):
        super().__init__()
        g = min(groups, in_ch) if in_ch % groups else groups
        self.norm1 = nn.GroupNorm(g, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.film = FiLM(cond_dim, out_ch)
        g2 = min(groups, out_ch) if out_ch % groups else groups
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film(h, c)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# ---- down / up samplers --------------------------------------------------
class Downsample3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="trilinear", align_corners=False)
        return self.conv(x)


# ---- full U-Net ----------------------------------------------------------
class UNet3D(nn.Module):
    """3D U-Net.  Input shape (B, C, D, D, D), output shape (B, C, D, D, D).

    For Phase A (smoke) defaults: D=64, C=6, channels (32, 64, 128, 256),
    1 block per level, total ~ a few M params -> fits the RTX 4000 Ada at
    batch 8-16 in bf16.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int | None = None,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        cond_dim: int = 256,
        time_embed_dim: int = 256,
        blocks_per_level: int = 1,
    ):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        chs = [base_channels * m for m in channel_mults]

        # Time -> embed (added to cond before every FiLM).
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Stem.
        self.stem = nn.Conv3d(in_channels, chs[0], kernel_size=3, padding=1)

        # Encoder.
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = chs[0]
        skip_channels: list[int] = []
        for li, out_ch in enumerate(chs):
            level = nn.ModuleList(
                [ResBlock3D(in_ch if k == 0 else out_ch, out_ch, cond_dim)
                 for k in range(blocks_per_level)]
            )
            self.down_blocks.append(level)
            skip_channels.append(out_ch)
            in_ch = out_ch
            if li < len(chs) - 1:
                self.downsamples.append(Downsample3D(in_ch))

        # Bottleneck — two ResBlocks at the lowest resolution.
        self.mid1 = ResBlock3D(in_ch, in_ch, cond_dim)
        self.mid2 = ResBlock3D(in_ch, in_ch, cond_dim)

        # Decoder — mirrors encoder, concatenates skips.
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for li in reversed(range(len(chs))):
            out_ch = chs[li]
            level = nn.ModuleList()
            for k in range(blocks_per_level):
                cat_ch = in_ch + (skip_channels[li] if k == 0 else 0)
                level.append(ResBlock3D(cat_ch, out_ch, cond_dim))
                in_ch = out_ch
            self.up_blocks.append(level)
            if li > 0:
                self.upsamples.append(Upsample3D(in_ch))

        # Output head -- normalise + project back to C channels.
        self.out_norm = nn.GroupNorm(min(8, chs[0]), chs[0])
        self.out_conv = nn.Conv3d(chs[0], out_channels, kernel_size=3, padding=1)
        # Zero-init the final layer so the network initially predicts zero
        # noise. Standard diffusion-training trick (Nichol & Dhariwal 2021;
        # also DiT, ADM) -- prevents initial-loss explosion + makes the early
        # training signal a clean denoising target.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # -------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                 # (B, C, D, D, D)
        t: torch.Tensor,                 # (B,) int64 or float32 timesteps
        cond: torch.Tensor,              # (B, cond_dim) from ConditionEncoder
    ) -> torch.Tensor:                   # (B, C, D, D, D)
        c = cond + self.time_embed(t)    # merged conditioning vector

        h = self.stem(x)

        skips: list[torch.Tensor] = []
        for li, level in enumerate(self.down_blocks):
            for block in level:
                h = block(h, c)
            skips.append(h)
            if li < len(self.downsamples):
                h = self.downsamples[li](h)

        h = self.mid1(h, c)
        h = self.mid2(h, c)

        # Decoder traverses the chs list in reverse.
        n_levels = len(self.down_blocks)
        up_iter = iter(range(n_levels))
        for ui, level in enumerate(self.up_blocks):
            li = (n_levels - 1) - ui     # corresponding encoder level
            for k, block in enumerate(level):
                if k == 0:
                    h = torch.cat([h, skips[li]], dim=1)
                h = block(h, c)
            if ui < len(self.upsamples):
                h = self.upsamples[ui](h)

        h = F.silu(self.out_norm(h))
        return self.out_conv(h)
