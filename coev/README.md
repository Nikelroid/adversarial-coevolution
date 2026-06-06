# coev: a universal RL-vs-opponent training pipeline

`coev` is the Gin Rummy pipeline made game-agnostic. It trains a masked PPO or TRPO agent against an
opponent curriculum in **any** PettingZoo AEC game, or in your own AEC environment. Gin Rummy was
just the first example.

## What it does

Given a function that builds your game, it:

1. turns the multi-agent game into a single-agent training problem (one seat learns, the rest are
   opponents);
2. masks illegal moves, so the agent only ever plays legal actions (and acts by taking the best
   legal move, never a random fallback);
3. schedules opponents from random, to a growing pool of past agents, to self-play;
4. keeps the best checkpoint (training often drifts past its peak);
5. evaluates against random play and an optional expert you supply, and writes `result.json`.

Nothing in the core is tied to one game. The only required input is `env_fn`.

## Use it

```python
from pettingzoo.classic import connect_four_v3
from coev import CoevConfig, train

cfg = CoevConfig(env_fn=connect_four_v3.env, env_id="connect_four",
                 algo="trpo", total_steps=2_000_000)
train(cfg)
```

Run the bundled examples:

```bash
python -m coev.examples.connect_four     # any PettingZoo game, no game-specific code
python -m coev.examples.gin_rummy        # same pipeline + a gold-standard benchmark and reward shaping
```

## Your own environment

Pass `env_fn` returning any environment that follows the PettingZoo AEC API: `reset`, `agents`,
`agent_selection`, `last()`, `step()`. For action masking, the per-step observation should be a dict
with an `action_mask`; without one, every action is treated as legal.

## Config (the parts you will touch most)

| field | meaning |
|---|---|
| `env_fn`, `env_kwargs` | builds your game (required) |
| `algo` | `"ppo"` or `"trpo"` |
| `total_steps`, `num_envs` | training budget |
| `stages` | the opponent schedule (defaults to random then pool then self) |
| `seed_models` | paths to prior agents to practise against |
| `benchmark_agent` | an optional expert to grade against (and to keep-best on) |
| `reward_transform` | optional `(reward, obs, done, info) -> reward` shaping |
| `init_model` | warm-start from a prior model |
| `keep_best` | ship the best checkpoint, not the final one |

## Files

`policy.py` masked policy, `agents.py` opponents, `env.py` the single-agent wrapper, `curriculum.py`
the opponent schedule, `train.py` the entrypoint and evaluator, `config.py` the config.
