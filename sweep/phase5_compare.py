"""Phase 5: compare observation REPRESENTATIONS for RL on Gin Rummy.

Three conditions, identical RL otherwise:
  * sparse   : raw 208-d (4x52) observation via the standard CombinedExtractor.
  * llm      : the LLM-similarity-judged dense embedding (frozen).
  * temporal : the self-supervised temporal-context dense embedding (frozen).

For a given EMBEDDER ('sparse' or a path to an embedder.pt), warm-nothing: train PPO
self-play vs the frozen champion for RL_STEPS, then eval vs gold / champion / random.
Write a result JSON tagged by condition -> compare sample-efficiency + final win rates
(esp. vs the gold yardstick). CPU.

Env: EMBEDDER (sparse | path.pt), TAG, RL_STEPS, NUM_ENV, N_STEPS, EVAL_GAMES, SEED.
"""
import os, sys, json, time, functools
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
try:
    import cv2  # noqa: F401  (parent import before fork; see llmplay_one)
except Exception:
    pass
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor

from gym_wrapper import GinRummySB3Wrapper
from agents.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
import ppo_train  # noqa: F401  registers MaskedGinRummyPolicy
from sweep.embed_train import Embedder
from sweep.llm_dagger_one import evaluate, KNOCK, GIN   # reuse the eval harness


class EmbedExtractor(BaseFeaturesExtractor):
    """Frozen learned embedding of the 208-d observation as the policy's features.
    The action mask is read separately by MaskedGinRummyPolicy, so masking is intact."""
    def __init__(self, observation_space, embedder_path=None):
        ckpt = th.load(embedder_path, map_location="cpu")
        cfg = ckpt["cfg"]
        super().__init__(observation_space, features_dim=cfg["emb_dim"])
        self.emb = Embedder(**cfg)
        self.emb.load_state_dict(ckpt["state_dict"])
        for p in self.emb.parameters():
            p.requires_grad_(False)
        self.emb.eval()

    def forward(self, obs):
        x = obs["observation"].float()
        x = x.reshape(x.shape[0], -1)
        with th.no_grad():
            return self.emb(x)


def main():
    embedder = os.environ.get("EMBEDDER", "sparse")
    tag = os.environ.get("TAG", "sparse" if embedder == "sparse" else "embed")
    rl_steps = int(os.environ.get("RL_STEPS", 2_000_000))
    num_env = int(os.environ.get("NUM_ENV", 64))
    n_steps = int(os.environ.get("N_STEPS", 256))
    eval_games = int(os.environ.get("EVAL_GAMES", 400))
    champ_path = os.environ.get("CHAMP_MODEL", "game/model/ppo_gin_rummy_selfplay.zip")
    seed = int(os.environ.get("SEED", 0))
    th.manual_seed(seed); np.random.seed(seed)

    print(f"=== phase5 tag={tag} embedder={embedder} rl_steps={rl_steps} ===", flush=True)
    champ = PPO.load(champ_path, device="cpu")

    def _mk(rank):
        def _t():
            e = GinRummySB3Wrapper(
                opponent_policy=functools.partial(PPOAgent, model=champ),
                randomize_position=True, turns_limit=200, curriculum_manager=None,
                rank=rank, knock_reward=KNOCK, gin_reward=GIN)
            e.reset(seed=seed + rank)
            return Monitor(e)
        return _t
    env = SubprocVecEnv([_mk(i) for i in range(num_env)], start_method="fork")

    net_arch = dict(pi=[256, 128], vf=[256, 128])    # same head for all conditions
    if embedder == "sparse":
        pkw = dict(features_extractor_class=CombinedExtractor, net_arch=net_arch,
                   activation_fn=th.nn.Tanh, ortho_init=True)
    else:
        pkw = dict(features_extractor_class=EmbedExtractor,
                   features_extractor_kwargs=dict(embedder_path=embedder),
                   net_arch=net_arch, activation_fn=th.nn.Tanh, ortho_init=True)
    model = PPO(ppo_train.MaskedGinRummyPolicy, env, n_steps=n_steps, device="cpu",
                verbose=0, policy_kwargs=pkw)

    t0 = time.time()
    model.learn(total_timesteps=rl_steps)
    secs = time.time() - t0
    env.close()

    res = {k: evaluate(model, k, champ, eval_games) for k in ("random", "champion", "gold")}
    out_dir = os.path.join(PROJECT_ROOT, "sweep", "phase5")
    os.makedirs(out_dir, exist_ok=True)
    result = dict(tag=tag, embedder=embedder, rl_steps=rl_steps, train_seconds=secs,
                  vs_random=res["random"], vs_champion=res["champion"], vs_gold=res["gold"])
    json.dump(result, open(os.path.join(out_dir, f"phase5_{tag}.json"), "w"), indent=2)
    print(f"[phase5 {tag}] vs_random={res['random']['win_rate']:.3f} "
          f"vs_champion={res['champion']['win_rate']:.3f} "
          f"vs_gold={res['gold']['win_rate']:.3f} (in {secs:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
