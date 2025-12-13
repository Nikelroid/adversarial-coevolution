# 🗡️ DAgger Training Guide

DAgger (**D**ataset **Agg**regation) is an Imitation Learning algorithm where the agent learns to mimic an **Expert**. This is often faster and more stable than pure Reinforcement Learning (PPO) for games like Gin Rummy.

In this project, we use a heuristic **Expert Agent** as the teacher.

---

## 🚀 Quick Start

To start DAgger training, run:

```bash
python dagger_train.py --timesteps 100000 --episodes-per-iter 50 --epochs 4
```

### Arguments
*   `--timesteps`: Total number of training steps (data samples collected). Suggestion: `100000` to `500000`.
*   `--episodes-per-iter`: Number of games to play per iteration to collect data. Suggestion: `50` or `100`.
*   `--epochs`: Number of supervised learning passes over the aggregated dataset per iteration. Suggestion: `4`-`10`.

---

## 🧠 How It Works

1.  **Collection Phase**:
    *   The student (your model) plays against an opponent.
    *   For every state, the **Expert** calculates the best action (label).
    *   **Teacher Forcing**: Initially, the agent executes the Expert's action (Beta=1.0). Over time, Beta decays, and the agent executes its own action but still learns from the Expert's "advice".

2.  **Aggregation Phase**:
    *   All observed `(state, expert_action)` pairs are added to a growing dataset.

3.  **Training Phase**:
    *   The model is trained using Supervised Learning (Cross Entropy Loss) on the entire dataset.

---

## 📊 Viewing Results

Results are logged to **Weights & Biases (WandB)** project `Adversarial-CoEvolution`.

**Metrics to watch:**
*   `dagger/loss`: Should decrease over time (indicates model is learning to predict expert moves).
*   `dagger/win_rate`: Win rate against the random opponent during data collection. Should increase.
*   `dagger/beta`: Defaults to decaying from 1.0 to 0.0 over 20 iterations.

---

## 🧪 Validating the Trained Model

Once training is complete, the model runs are saved in `artifacts/models/dagger/`.
You can evaluate the final model using `main.py`:

```bash
python main.py --player1 ppo --player2 random --model ./artifacts/models/dagger/final_model --tournament --num-games 1000
```
*(Note: Use `--player1 ppo` because the DAgger model is compatible with the PPO agent architecture)*
