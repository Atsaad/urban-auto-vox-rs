"""
ConditionEncoder — turn (categorical IDs + continuous scalars) into a single
fixed-dim conditioning vector for the U-Net's FiLM layers.

Design (claude.md §8 + §9):
- Each categorical feature gets a learned embedding (vocab_size includes the
  "(null)" token at id 0, which the dataset uses for empty strings).
- Continuous features are already standardised in the Dataset (z-score) and
  get a small MLP.
- The two are concatenated and projected to `cond_dim`.

CFG support is built-in:
- A learned `null_embedding` of shape (cond_dim,) replaces the per-sample
  conditioning vector whenever the `drop` mask says so. This is the standard
  classifier-free guidance trick (Ho & Salimans 2022).
- The training loop chooses which rows to drop (Bernoulli(p_drop)); the
  encoder simply applies the mask. At inference time we form both ε(x, c) and
  ε(x, ∅) with the same module by toggling the mask.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict[str, int],          # {feature_name: vocab_size}
        categorical_cols: Iterable[str],      # canonical order
        n_continuous: int,                    # length of the cont vector
        embed_dim: int = 32,                  # per-feature embedding width
        cont_hidden: int = 64,                # MLP hidden width
        cond_dim: int = 256,                  # final conditioning width
    ):
        super().__init__()
        self.categorical_cols = list(categorical_cols)
        self.cond_dim = cond_dim

        # Per-feature embeddings.
        self.embeds = nn.ModuleDict({
            c: nn.Embedding(vocab_sizes[c], embed_dim)
            for c in self.categorical_cols
        })

        # Continuous MLP.
        self.cont_mlp = nn.Sequential(
            nn.Linear(n_continuous, cont_hidden),
            nn.SiLU(),
            nn.Linear(cont_hidden, cont_hidden),
        )

        # Fuse: concat all embeddings + cont hidden, then project to cond_dim.
        in_dim = embed_dim * len(self.categorical_cols) + cont_hidden
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Learned "no condition" vector for CFG inference.
        self.null_embedding = nn.Parameter(torch.zeros(cond_dim))
        nn.init.normal_(self.null_embedding, std=0.02)

    def forward(
        self,
        cat_ids: dict[str, torch.Tensor],   # {feature: (B,) long}
        cont: torch.Tensor,                 # (B, n_continuous) float
        drop: torch.Tensor | None = None,   # (B,) bool — replace with ∅
    ) -> torch.Tensor:                      # (B, cond_dim)
        emb = [self.embeds[c](cat_ids[c]) for c in self.categorical_cols]
        emb = torch.cat(emb, dim=-1)                            # (B, embed_dim*K)
        cont_h = self.cont_mlp(cont)                            # (B, cont_hidden)
        c = self.fuse(torch.cat([emb, cont_h], dim=-1))         # (B, cond_dim)

        if drop is not None and drop.any():
            null = self.null_embedding.unsqueeze(0).expand_as(c)
            c = torch.where(drop.unsqueeze(-1), null, c)
        return c
