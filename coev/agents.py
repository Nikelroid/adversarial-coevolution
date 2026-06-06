"""Opponent / evaluation agents for any PettingZoo AEC environment.

An agent reads the current observation from ``env.last()`` and returns one action. ``set_player``
records which seat it is playing (turn-based games give each agent its own observation when it is
that agent's turn, so a single instance can play any seat).
"""
import numpy as np
import torch as th


def masked_argmax(model, obs):
    """Highest-probability LEGAL action from a trained model's masked action distribution. Never a
    random fallback. Works whether or not the observation has an ``action_mask`` (if it does not,
    this is a plain argmax over the policy's probabilities)."""
    with th.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        probs = model.policy.get_distribution(obs_t).distribution.probs
        probs = probs.detach().cpu().numpy().reshape(-1)
    mask = obs.get("action_mask") if isinstance(obs, dict) else None
    if mask is None:
        return int(probs.argmax())
    mask = np.asarray(mask).astype(bool)
    masked = np.where(mask, probs, -np.inf)
    if not np.isfinite(masked).any():                 # no legal mass (shouldn't happen)
        masked = np.where(mask, 1.0, -np.inf)
    return int(np.argmax(masked))


class BaseAgent:
    def __init__(self, env):
        self.env = env
        self.seat = None

    def set_player(self, seat):
        self.seat = seat

    def _obs(self):
        return self.env.last()[0]

    def do_action(self) -> int:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    """Plays a uniformly random legal move (the easiest opponent / a baseline)."""
    def do_action(self) -> int:
        obs = self._obs()
        mask = obs.get("action_mask") if isinstance(obs, dict) else None
        if mask is not None:
            legal = np.where(np.asarray(mask) == 1)[0]
            return int(np.random.choice(legal))
        seat = self.seat or self.env.agent_selection
        return int(self.env.action_space(seat).sample())


class PolicyAgent(BaseAgent):
    """Wraps a trained Stable-Baselines3 model (PPO or TRPO) as an opponent. Plays the
    highest-probability legal move via masked_argmax."""
    def __init__(self, env, model):
        super().__init__(env)
        self.model = model

    def do_action(self) -> int:
        return masked_argmax(self.model, self._obs())
