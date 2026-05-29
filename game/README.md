# 🃏 Play Gin Rummy vs the trained PPO agent

A small **local web app** to play Gin Rummy in your browser against the
reinforcement-learning agent trained in this project.

It is designed to be **bug-free by construction**: the server runs the *exact*
PettingZoo `gin_rummy_v4` environment the agent was trained on, the environment
enforces every rule, and the browser can only ever send moves the environment
has marked legal. No game logic is re-implemented on the client.

```
  Browser (renders cards, sends clicks)
        │  HTTP/JSON
        ▼
  Python server  ──►  gin_rummy_v4 env  ◄──  trained PPO agent (the opponent)
  (game/server.py)     (authoritative rules)
```

---

## Prerequisites

- **Python 3.10 or 3.11**
- **git**
- ~2 GB free disk (for PyTorch). **No GPU needed** — it runs fine on CPU.

---

## 1. Get the code

```bash
# first time:
git clone https://github.com/Nikelroid/Adversarial-CoEvolution.git
cd Adversarial-CoEvolution

# or, if you already have it:
cd Adversarial-CoEvolution
git pull
```

## 2. Create an isolated Python environment

<details open><summary><b>Option A — venv (built in)</b></summary>

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```
</details>

<details><summary><b>Option B — conda</b></summary>

```bash
conda create -n ginrummy python=3.11 -y
conda activate ginrummy
```
</details>

## 3. Install dependencies

```bash
pip install -r game/requirements.txt
```

> The web server itself uses only Python's standard library. These packages are
> what the trained model and the game environment need (numpy, torch,
> stable-baselines3, gymnasium, pettingzoo[classic]). If `pip` installs a CUDA
> build of torch you don't want, install the CPU build first from
> <https://pytorch.org/get-started/locally/>.

## 4. Run it

From the **project root**:

```bash
python game/server.py
```

You'll see:

```
Loading 'winrate' opponent: .../game/model/ppo_gin_rummy_winrate.zip
Loading 'reward' opponent: .../game/model/ppo_gin_rummy_reward.zip
Models loaded. Starting a game session...
============================================================
  Gin Rummy vs PPO is running at:  http://localhost:8000
  Open that URL in your browser. Ctrl-C to stop.
============================================================
```

Open **http://localhost:8000** in your browser and play. 🎉

### Choose your opponent

A dropdown at the top of the page lets you pick which trained agent to play
against (switching it starts a fresh game). Both ship in `game/model/`:

| Opponent | Model | Notes |
|---|---|---|
| **Self-play champion** (default) | self-play gen-1 (fine-tuned from run_5) | **strongest so far** — beats the base agents ~61%, 98.7% vs random |
| **Highest win rate** | `run_5` | 99.6% wins vs random |
| **Highest reward** | `run_2` | 0.54 avg reward |

### Options

```bash
python game/server.py --port 8001          # use a different port
GIN_OPPONENT=reward python game/server.py   # start on the "highest reward" agent
```

---

## How to play

**Goal:** arrange your 10 cards into **melds** and keep your **deadwood** low.
- **Set** — 3 or 4 cards of the same rank (e.g. 7♠ 7♥ 7♦).
- **Run** — 3+ consecutive cards of the same suit (e.g. 5♣ 6♣ 7♣).
- **Deadwood** — points left in cards that aren't in a meld (A=1, 2–10 = face
  value, J/Q/K = 10).

**Each turn:**
1. **Draw** — click the **Stock** pile (face-down) to draw a fresh card, or
   click the **Discard** pile to take the visible top card.
2. **Discard** — click one of your cards to throw it away.

**Ending the round:**
- **Knock** — when your deadwood would be **≤ 10**, click **Knock**, then click
  the card to discard. Lower deadwood than the opponent wins the round.
- **Gin** — when *all* your cards are melded (deadwood 0), click **Gin!**.

**Reading the board:**
- A **gold outline** marks cards currently in a meld.
- The **deadwood** badge (top-right) shows your current count.
- The PPO agent (top of the table, face-down) plays its turns automatically.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Model not found` | Ensure `game/model/ppo_gin_rummy_winrate.zip` and `ppo_gin_rummy_reward.zip` exist (they ship with the repo). |
| `Address already in use` | Another process holds the port — run with `--port 8001`. |
| `ModuleNotFoundError` | Activate your environment and re-run `pip install -r game/requirements.txt`. |
| First move feels slow | The model loads once at startup (a few seconds); play is instant after. |
| Blank page / 404s | Make sure you launched from the **project root** (`python game/server.py`), not from inside `game/`. |

---

## How it stays bug-free

- **Authoritative server.** All rules come from `gin_rummy_v4`; the browser only
  renders state and posts a chosen action id.
- **Legal moves only.** Every clickable affordance is built from the env's
  `action_mask`, and the server re-checks the mask before stepping — an illegal
  move returns `400` and never reaches the game.
- **One request at a time.** The client serializes requests, so the UI can't
  desync from the server.
