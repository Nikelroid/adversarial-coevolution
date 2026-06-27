"""Custom SB3 feature extractors for the Gin Rummy architecture sweep (Phase 8).

Each extractor reads ONLY ``obs["observation"]`` (the (4, 52) card-plane tensor) and returns
a feature vector. The 110-dim action mask is read separately by the masked policy
(``FiniteMaskedPolicy``) and applied to the logits downstream, so **masking is unaffected** by
the choice of extractor -- exactly like the existing ``EmbedExtractor`` in ``phase5_compare``.

This module is intentionally standalone (only torch + SB3) so that ``PPO.load`` / ``TRPO.load``
can resolve these classes when a checkpoint trained with them is reloaded for eval or as an
opponent. Keep it import-light.

Observation layout (from ``gym_wrapper``): a Dict space with
  ``observation``: Box(0, 1, shape=(4, 52))  -- 4 card planes x 52 cards
  ``action_mask``: Box(0, 1, shape=(110,))
"""
from __future__ import annotations

import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Conv1DCardExtractor(BaseFeaturesExtractor):
    """1-D convolutions over the card axis: treat (4, 52) as 4 channels over a length-52 card
    sequence. Local kernels capture short-range card structure (e.g. runs) **when the deck is
    rank-ordered within suit** -- a weak prior we disclose in the paper. ~140K params."""

    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)
        self.net = th.nn.Sequential(
            th.nn.Conv1d(4, 32, kernel_size=3, padding=1), th.nn.ReLU(),
            th.nn.Conv1d(32, 64, kernel_size=3, padding=1), th.nn.ReLU(),
            th.nn.AdaptiveAvgPool1d(8), th.nn.Flatten(),
            th.nn.Linear(64 * 8, features_dim), th.nn.ReLU(),
        )

    def forward(self, obs):
        x = obs["observation"].float()          # (B, 4, 52)
        return self.net(x)


class DeepSetsCardExtractor(BaseFeaturesExtractor):
    """Permutation-invariant set encoder over the 52 cards (Deep Sets, Zaheer et al. 2017).

    Each card is a token = its 4 plane-bits concatenated with a learned identity embedding (so
    invariance over the *set* does not erase *which* card it is); a shared phi maps every token,
    sum-pooling aggregates, and rho produces the features. The most principled encoder for an
    unordered hand of cards. ~15K params (lightest)."""

    def __init__(self, observation_space, features_dim: int = 128, id_dim: int = 16,
                 hidden: int = 64):
        super().__init__(observation_space, features_dim=features_dim)
        self.id = th.nn.Parameter(th.zeros(52, id_dim))
        th.nn.init.normal_(self.id, std=0.1)
        self.phi = th.nn.Sequential(
            th.nn.Linear(4 + id_dim, hidden), th.nn.ReLU(),
            th.nn.Linear(hidden, hidden), th.nn.ReLU(),
        )
        self.rho = th.nn.Linear(hidden, features_dim)

    def forward(self, obs):
        x = obs["observation"].float().permute(0, 2, 1)        # (B, 52, 4)
        ids = self.id.unsqueeze(0).expand(x.shape[0], -1, -1)  # (B, 52, id_dim)
        tok = th.cat([x, ids], dim=-1)                          # (B, 52, 4+id_dim)
        h = self.phi(tok).sum(dim=1)                            # (B, hidden) -- sum-pool
        return self.rho(h)


class SetAttentionExtractor(BaseFeaturesExtractor):
    """Light self-attention (transformer) encoder over the 52 card tokens. Each token =
    Linear(4 -> d_model) + a learned positional embedding; ``layers`` TransformerEncoder layers;
    mean-pool -> features. Slowest per step on CPU (~20-40%), budget for it. ~80K params."""

    def __init__(self, observation_space, features_dim: int = 128, d_model: int = 64,
                 nhead: int = 4, layers: int = 2, dim_ff: int = 128):
        super().__init__(observation_space, features_dim=features_dim)
        self.proj = th.nn.Linear(4, d_model)
        self.pos = th.nn.Parameter(th.zeros(52, d_model))
        th.nn.init.normal_(self.pos, std=0.1)
        enc_layer = th.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            batch_first=True, activation="gelu")
        self.enc = th.nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.out = th.nn.Linear(d_model, features_dim)

    def forward(self, obs):
        x = obs["observation"].float().permute(0, 2, 1)    # (B, 52, 4)
        h = self.proj(x) + self.pos.unsqueeze(0)           # (B, 52, d_model)
        h = self.enc(h)                                    # (B, 52, d_model)
        return self.out(h.mean(dim=1))                     # (B, features_dim)
