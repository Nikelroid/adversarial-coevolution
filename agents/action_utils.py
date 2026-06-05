"""Shared action selection for trained policies.

The one correct way for a trained model to ACT (as an opponent or under evaluation) is to
take the highest-probability LEGAL action from its masked action distribution -- never to
predict() and then substitute a RANDOM legal action when the predicted one is illegal. The
random fallback silently makes the agent play a random move in those states, which understates
a hero's true strength at evaluation and weakens pool/self opponents during training. This
helper guarantees the best legal move always, and re-applies the legal mask so it is correct
whether or not the policy masks internally.
"""
import numpy as np
import torch as th


def masked_argmax(model, obs):
    """Highest-probability legal action for `model` given the env dict observation `obs`
    (which carries an 'action_mask'). Returns an int action id."""
    mask = np.asarray(obs["action_mask"]).astype(bool)
    with th.no_grad():
        obs_t, _ = model.policy.obs_to_tensor(obs)
        probs = model.policy.get_distribution(obs_t).distribution.probs
        probs = probs.detach().cpu().numpy().reshape(-1)
    masked = np.where(mask, probs, -np.inf)            # consider only legal actions
    if not np.isfinite(masked).any():                  # degenerate: no legal prob mass
        masked = np.where(mask, 1.0, -np.inf)           # -> any legal action
    return int(np.argmax(masked))
