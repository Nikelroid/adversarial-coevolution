"""Action-masked actor-critic policy that works for both PPO and TRPO, on any environment whose
observation is a dict that contains an ``action_mask``. If the observation has no mask, it behaves
like an ordinary policy (every action is legal).

The mask is applied to the action logits, setting illegal actions to a large negative number
(``-1e8``) instead of ``-inf``. PPO tolerates ``-inf``, but TRPO's conjugate-gradient / KL math
produces NaNs on infinities, so a large finite value gives near-zero probability while keeping the
same policy usable by both algorithms. This is the proven masking from the Gin Rummy work, made
game-agnostic.
"""
import torch as th
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class MaskedPolicy(ActorCriticPolicy):
    MASK_FILL = -1e8

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_entropy = None

    # -------------------------------------------------- mask helpers
    def _mask(self, obs):
        if isinstance(obs, dict) and "action_mask" in obs:
            m = obs["action_mask"]
            if not isinstance(m, th.Tensor):
                m = th.as_tensor(m, device=self.device)
            return m
        return None

    def _mask_logits(self, logits, mask):
        if mask is None:
            return logits
        mask = mask.to(dtype=th.bool, device=logits.device)
        return th.where(mask, logits, th.full_like(logits, self.MASK_FILL))

    def _latents(self, obs):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            return self.mlp_extractor(features)              # (latent_pi, latent_vf)
        pi_features, vf_features = features
        return (self.mlp_extractor.forward_actor(pi_features),
                self.mlp_extractor.forward_critic(vf_features))

    # -------------------------------------------------- SB3 hooks
    def forward(self, obs, deterministic: bool = False):
        mask = self._mask(obs)
        latent_pi, latent_vf = self._latents(obs)
        logits = self._mask_logits(self.action_net(latent_pi), mask)
        dist = CategoricalDistribution(self.action_space.n).proba_distribution(action_logits=logits)
        self.last_entropy = dist.entropy().mean().item()
        actions = dist.get_actions(deterministic=deterministic)
        return actions, self.value_net(latent_vf), dist.log_prob(actions)

    def get_distribution(self, obs):
        mask = self._mask(obs)
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi = self.mlp_extractor.forward_actor(features)
        else:
            latent_pi = self.mlp_extractor.forward_actor(features[0])
        logits = self._mask_logits(self.action_net(latent_pi), mask)
        return CategoricalDistribution(self.action_space.n).proba_distribution(action_logits=logits)

    def evaluate_actions(self, obs, actions):
        mask = self._mask(obs)
        latent_pi, latent_vf = self._latents(obs)
        logits = self._mask_logits(self.action_net(latent_pi), mask)
        dist = CategoricalDistribution(self.action_space.n).proba_distribution(action_logits=logits)
        return self.value_net(latent_vf), dist.log_prob(actions), dist.entropy()
