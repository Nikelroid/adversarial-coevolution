"""Hyperparameter grid for the PPO Gin Rummy sweep vs RandomAgent.

The 8 configs sweep learning_rate x ent_coef (the two most impactful HPs for
masked PPO on card games), plus two probes around rollout length and update
frequency at the most defensible defaults.

Every run stays under 5M timesteps so the curriculum is always in Phase 1
(100% random opponents) — pure signal for "vs random" performance.
"""

BASE = dict(
    total_timesteps=2_000_000,
    num_env=96,
    n_steps=512,
    batch_size=1024,
    n_epochs=4,
    learning_rate=1e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    weight_decay=0.0,
    seed=100,
    eval_episodes=1000,
    turns_limit=200,
)

VARIANTS = [
    dict(learning_rate=3e-4, ent_coef=0.01),
    dict(learning_rate=3e-4, ent_coef=0.03),
    dict(learning_rate=1e-4, ent_coef=0.01),
    dict(learning_rate=1e-4, ent_coef=0.03),
    dict(learning_rate=5e-5, ent_coef=0.01),
    dict(learning_rate=5e-5, ent_coef=0.03),
    dict(learning_rate=1e-4, ent_coef=0.01, n_steps=1024),
    dict(learning_rate=1e-4, ent_coef=0.01, n_epochs=10),
]


def get_config(index: int) -> dict:
    cfg = BASE.copy()
    cfg.update(VARIANTS[index])
    cfg["config_id"] = index
    return cfg


N_CONFIGS = len(VARIANTS)
