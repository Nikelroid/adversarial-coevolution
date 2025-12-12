# Adversarial Co-Evolution of RL and VLM/LLM Agents in Multiplayer Games

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Stable Baselines3](https://img.shields.io/badge/Stable%20Baselines3-2.3+-green.svg)
![PettingZoo](https://img.shields.io/badge/PettingZoo-Gin%20Rummy-purple.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Dashboard-009688.svg)

## 📖 Overview

This project explores the intersection of **Reinforcement Learning (RL)** and **Large Language Models (LLMs)** in complex, imperfect-information environments (Gin Rummy). It addresses the challenge of training RL agents without reliable opponents or expensive human feedback by establishing an **adversarial co-evolutionary loop**.

We utilize LLMs (Llama 3, Gemma, GPT) as zero-shot strategic opponents to guide the training of efficient PPO agents. The system employs a **3-phase curriculum learning** approach to distill the broad, "common-sense" strategic knowledge of LLMs into a fast, compact RL policy.

## 🚀 Key Features & Achievements

* **High-Performance RL Pipeline:** Engineered a high-throughput, 64-96 core, multi-process PPO training pipeline with a custom action-masked policy using Stable Baselines 3 and PyTorch.
* **Curriculum Learning System:** Built a robust 3-phase curriculum learning system (Random → Self-Play → Adversarial) with a fully cached RAM model-pool API, achieving a **99.12% win rate** vs. baseline agents.
* **LLM Knowledge Distillation:** Architected a scalable API framework to integrate LLM strategic insights as a policy prior, enabling agents to learn from models like Llama 3 and GPT-OSS via Ollama/HuggingFace.
* **Interactive Evaluation Suite:** Designed and built a custom evaluation environment (PettingZoo) and Web UI for critical live human-vs-agent testing and qualitative validation of learned strategies.
* **Scientific Control Deck:** A comprehensive Web Dashboard to monitor training, manage distributed workers, edit prompts, and visualize real-time logs.

## 🛠️ System Architecture

The project consists of three main components:

1.  **The RL Agent (PPO):** A custom implementation of Proximal Policy Optimization with valid action masking, trained to handle the partial observability of Gin Rummy.
2.  **The LLM Agent:** A sophisticated wrapper that parses game states into text prompts (Chain-of-Thought) and parses LLM responses back into valid game actions.
3.  **Distributed Inference Engine:** A Master-Worker architecture allowing multiple LLMs (e.g., Llama-70B for reasoning, Llama-3B for speed) to serve inference requests asynchronously, decoupling generation from the training loop.

## 🖥️ Scientific Control Deck (Dashboard)

We provide a **Scientific Web Dashboard** to serve as the unified command center for the project.

- **One-Click Launch:** `./start_dashboard.sh`
- **Port:** `http://localhost:8001`

### Features
- **Unified Process Management:** Start/Stop Training, Master Server, Workers, and user Game sessions from one interface.
- **Hyperparameter Tuning:** Adjust `Learning Rate`, `Batch Size`, and `Entropy Coefficient` on the fly.
- **Prompt Engineering:** Integrated **Prompt Editor** to modify the System Prompt (`config/prompt.txt`) and immediately see its effect on the agent.
- **Real-Time Telemetry:** Live log streaming via WebSockets for all active processes.
- **Model Selection:** Dynamically switch between `Llama-3.1-70B`, `Gemma-27B`, `Qwen-32B`, and others.

## 🔧 Installation & Quick Start

### Prerequisites
- Python 3.10+
- `ollama` (for local LLM inference)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-repo/Adversarial-CoEvolution.git
cd Adversarial-CoEvolution

# Install Dependencies
pip install -r requirements.txt
# (Includes stable-baselines3, pettingzoo, fastapi, uvicorn, etc.)
```

### 🚀 Launching the System

1.  **Start the Dashboard**:
    ```bash
    ./start_dashboard.sh
    ```
2.  **Open Browser**: Go to [http://localhost:8001](http://localhost:8001).
3.  **Start Components** (via Dashboard):
    *   **Master Node**: Click "Start Master".
    *   **Worker Node**: Select Model (e.g., `llama-3b`) and click "Spawn Worker".
    *   **Training**: Configure parameters and click "Start Training".

## ⚙️ Configuration (`config.yaml`)

All system settings are centralized in `config/config.yaml`.

```yaml
training:
  total_timesteps: 1000000
  num_env: 100

distributed:
  master:
    host: "0.0.0.0"
    port: 8000
  worker:
    default_model: "llama-3b"

models:
  llama-3b: "meta-llama/Llama-3.2-3B"
  llama-70b: "meta-llama/Meta-Llama-3.1-70B"
  # ... other models
```

## 🧠 Distributed Architecture

To handle the computational load of LLMs during training, we use a **Master-Worker** pattern:

*   **Master Node (`model_server.py`)**: Manages job queues (`fast` for actions, `slow` for enhancement) and routes requests.
*   **Worker Nodes (`model_worker.py`)**: Connect to the Master and process jobs using local or remote LLMs.
*   **Prompt Enhancement**: When the agent fails, it triggers a "Prompt Enhancement" job. A powerful model (e.g., Llama-70B) analyzes the failure and rewrites the prompt in `config/prompt.txt`.

## 📂 Project Structure

```
├── agents/             # PPO, Expert, and LLM Agent implementations
├── config/             # Configuration files (config.yaml, prompt.txt)
├── dashboard/          # FastAPI Backend & HTML/CSS Frontend
├── game/               # Gin Rummy Logic & Pygame UI
├── llm/                # API Clients & Validators
├── models/             # Saved PPO Models
├── ppo_train.py        # Main Training Script
├── model_server.py     # Distributed Master Node
├── model_worker.py     # Distributed Worker Node
└── start_dashboard.sh  # Unified Launcher
```

## License
MIT License.
