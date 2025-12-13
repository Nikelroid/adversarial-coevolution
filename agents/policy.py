import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution

class MaskedGinRummyPolicy(ActorCriticPolicy):
    """
    Masked Policy for Gin Rummy that handles action masking.
    Shared implementation for PPO and DAgger.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _extract_obs_and_mask(self, obs):
        obs_tensor = obs['observation']
        mask_tensor = obs.get('action_mask', None)
        if not isinstance(obs_tensor, th.Tensor):
            obs_tensor = th.as_tensor(obs_tensor, device=self.device).float()
        if mask_tensor is not None and not isinstance(mask_tensor, th.Tensor):
            mask_tensor = th.as_tensor(mask_tensor, device=self.device)
        return obs_tensor, mask_tensor

    def _apply_action_mask(self, logits, action_mask):
        if action_mask is None: return logits
        mask = action_mask.to(dtype=th.bool, device=logits.device)
        logits = th.where(mask, logits, th.tensor(float('-inf'), device=logits.device, dtype=logits.dtype))
        return logits

    def forward(self, obs, deterministic: bool = False):
        _, action_mask = self._extract_obs_and_mask(obs)
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        
        logits = self.action_net(latent_pi)
        masked_logits = self._apply_action_mask(logits, action_mask)
        distribution = CategoricalDistribution(self.action_space.n).proba_distribution(action_logits=masked_logits)
        
        values = self.value_net(latent_vf)
        actions = distribution.sample() if not deterministic else th.argmax(masked_logits, dim=1)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def get_action_logits(self, obs):
        """Helper to get logits for supervised learning"""
        _, action_mask = self._extract_obs_and_mask(obs)
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features) if self.share_features_extractor else (self.mlp_extractor.forward_actor(features[0]), None)
        logits = self.action_net(latent_pi)
        masked_logits = self._apply_action_mask(logits, action_mask)
        return masked_logits
