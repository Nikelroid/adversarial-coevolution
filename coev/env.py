"""Turn a multi-agent PettingZoo AEC environment into a single-agent Gymnasium environment that
Stable-Baselines3 can train on.

One seat is the learner. Every other seat is played by an opponent (random, a model from the
curriculum pool, or the current model for self-play). This works for ANY turn-based PettingZoo AEC
game (connect four, tic-tac-toe, Gin Rummy, Leduc, ...) and for your own environment, as long as it
follows the AEC API: ``reset``, ``agent_selection``, ``agents``, ``last()``, ``step()``, and (for
masking) an observation dict that contains ``action_mask``.

It is game-agnostic: it never assumes seat names, a player count, or any game-specific reward. You
pass ``env_fn`` (a callable returning a fresh AEC env) and, optionally, a ``reward_transform`` to
shape the reward.
"""
import random

import gymnasium as gym
import numpy as np

from coev.agents import RandomAgent, PolicyAgent


class MaskedCoevEnv(gym.Env):
    """Single-agent view of an AEC game for one learner seat."""

    def __init__(self, env_fn, opponent_policy=None, curriculum=None, env_kwargs=None,
                 randomize_position=True, turns_limit=1000, reward_transform=None, rank=0):
        super().__init__()
        self.env_fn = env_fn
        self.env_kwargs = env_kwargs or {}
        self.env = env_fn(**self.env_kwargs)
        self.opponent_factory = opponent_policy or RandomAgent   # class instantiated with the env
        self.curriculum = curriculum
        self.randomize_position = randomize_position
        self.turns_limit = turns_limit
        self.reward_transform = reward_transform
        self.rank = rank
        self.opponent = None
        self.training_agent = None

        # infer the spaces from the env's first live agent
        self.env.reset()
        agent = self.env.agent_selection
        self.observation_space = self.env.observation_space(agent)
        self.action_space = self.env.action_space(agent)
        self.env.reset()

    # ---------------------------------------------------- opponent selection
    def _select_opponent(self):
        if self.curriculum is None:
            self.opponent = self.opponent_factory(self.env)
            return
        kind, _phase, _step = self.curriculum.get_opponent_type()
        if kind == "pool":
            model = self.curriculum.get_policy_from_pool()
            self.opponent = (PolicyAgent(self.env, model) if model is not None
                             else RandomAgent(self.env))
        elif kind == "self":
            model = self.curriculum.get_selfplay_policy()
            self.opponent = (PolicyAgent(self.env, model) if model is not None
                             else RandomAgent(self.env))
        else:
            self.opponent = RandomAgent(self.env)

    def _opponent_step(self):
        _obs, _r, term, trunc, _info = self.env.last()
        if term or trunc:
            self.env.step(None)
            return
        self.opponent.set_player(self.env.agent_selection)
        self.env.step(self.opponent.do_action())

    def _reward(self):
        """Training agent's reward right now (PettingZoo keeps a per-agent reward dict)."""
        return float(self.env.rewards.get(self.training_agent, 0.0))

    def _shape(self, reward, obs, done, info):
        if self.reward_transform is not None:
            return float(self.reward_transform(reward, obs, done, info))
        return float(reward)

    # ---------------------------------------------------- gym API
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.reset(seed=seed)
            random.seed(seed); np.random.seed(seed)
        else:
            self.env.reset()
        self.turn_num = 0
        agents = list(self.env.agents)
        self._select_opponent()
        self.training_agent = random.choice(agents) if self.randomize_position else agents[0]
        while self.env.agent_selection != self.training_agent:
            self._opponent_step()
        obs, _, _, _, _ = self.env.last()
        return obs, {}

    def step(self, action):
        obs, _reward, term, trunc, info = self.env.last()
        if not term and not trunc:
            mask = obs.get("action_mask") if isinstance(obs, dict) else None
            if mask is not None and not mask[action]:
                # illegal move: small penalty + a random legal move (teaches the agent to mask).
                legal = np.where(np.asarray(mask))[0]
                action = int(np.random.choice(legal))
                self.env.step(action)
                self.turn_num += 1
                return self._after_action(term_override=False, penalty=-1.0)
        self.env.step(action)
        self.turn_num += 1
        return self._after_action()

    def _after_action(self, term_override=None, penalty=0.0):
        # did the learner's own move end the game?
        _o, _r, term, trunc, info = self.env.last()
        if self.turn_num > self.turns_limit:
            trunc = True
        if term or trunc:
            return self._finish(info, penalty)
        # let opponents play until it is the learner's turn again
        while True:
            if self.env.agent_selection == self.training_agent:
                obs, _r, term, trunc, info = self.env.last()
                done = term or trunc or self.turn_num > self.turns_limit
                if done:
                    return self._finish(info, penalty)
                return obs, self._shape(self._reward() + penalty, obs, False, info), False, False, info
            self._opponent_step()
            _o, _r, term, trunc, info = self.env.last()
            if term or trunc:
                return self._finish(info, penalty)

    def _finish(self, info, penalty=0.0):
        obs, _r, _t, _tr, _info = self.env.last()
        reward = self._shape(self._reward() + penalty, obs, True, info)
        if self.curriculum is not None:
            self.curriculum.episode_complete()
        self._select_opponent()          # pick next episode's opponent now (before vec reset)
        return obs, reward, True, False, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
