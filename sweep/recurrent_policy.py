"""Masked recurrent (LSTM) policy for Gin Rummy -- the Phase-8 recurrence side-study.

Subclasses sb3-contrib's dict-obs recurrent policy and re-applies the legal-action mask to the
action logits, exactly like the feed-forward MaskedGinRummyPolicy does, but inside the recurrent
forward path. The mask is read from obs['action_mask'], stashed, and applied in
_get_action_dist_from_latent so it covers forward(), get_distribution() AND evaluate_actions()
(all three route through that method). -inf is used (RecurrentPPO is PPO, which handles it).
"""
from __future__ import annotations

import torch as th
from sb3_contrib.common.recurrent.policies import RecurrentMultiInputActorCriticPolicy


class MaskedRecurrentPolicy(RecurrentMultiInputActorCriticPolicy):
    _pending_mask = None

    def _stash_mask(self, obs):
        m = obs.get("action_mask") if isinstance(obs, dict) else None
        if m is not None and not isinstance(m, th.Tensor):
            m = th.as_tensor(m, device=self.device)
        self._pending_mask = m

    def _get_action_dist_from_latent(self, latent_pi):
        logits = self.action_net(latent_pi)
        m = self._pending_mask
        if m is not None and tuple(m.shape) == tuple(logits.shape):
            mask = m.to(dtype=th.bool, device=logits.device)
            # Finite floor (-1e8), NOT -inf: RecurrentPPO PADS sequences, so padded timesteps have an
            # all-zero mask -> with -inf they become an all-(-inf) row that Categorical rejects. A
            # finite floor makes such rows a valid uniform (the loss masks padded entries anyway),
            # while real timesteps still pick only legal actions (argmax/sample over real vs -1e8).
            logits = th.where(mask, logits, th.full_like(logits, -1e8))
            logits = th.nan_to_num(logits, nan=-1e8, posinf=-1e8, neginf=-1e8)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, lstm_states, episode_starts, deterministic: bool = False):
        self._stash_mask(obs)
        return super().forward(obs, lstm_states, episode_starts, deterministic)

    def get_distribution(self, obs, lstm_states, episode_starts):
        self._stash_mask(obs)
        return super().get_distribution(obs, lstm_states, episode_starts)

    def evaluate_actions(self, obs, actions, lstm_states, episode_starts):
        self._stash_mask(obs)
        return super().evaluate_actions(obs, actions, lstm_states, episode_starts)
