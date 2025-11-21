
# Adversarial Co-Evolution of RL and VLM/LLM Agents in Multiplayer Games

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Stable Baselines3](https://img.shields.io/badge/Stable%20Baselines3-2.3+-green.svg)
![PettingZoo](https://img.shields.io/badge/PettingZoo-Gin%20Rummy-purple.svg)

## 📖 Overview

This project explores the intersection of **Reinforcement Learning (RL)** and **Large Language Models (LLMs)** in complex, imperfect-information environments (Gin Rummy). It addresses the challenge of training RL agents without reliable opponents or expensive human feedback by establishing an **adversarial co-evolutionary loop**.

We utilize LLMs (Llama 3, Gemma, GPT) as zero-shot strategic opponents to guide the training of efficient PPO agents. The system employs a **3-phase curriculum learning** approach to distill the broad, "common-sense" strategic knowledge of LLMs into a fast, compact RL policy.

## 🚀 Key Features & Achievements

* **High-Performance RL Pipeline:** Engineered a high-throughput, 64-96 core, multi-process PPO training pipeline with a custom action-masked policy using Stable Baselines 3 and PyTorch.
* **Curriculum Learning System:** Built a robust 3-phase curriculum learning system (Random → Self-Play → Adversarial) with a fully cached RAM model-pool API, achieving a **99.12% win rate** vs. baseline agents.
* **LLM Knowledge Distillation:** Architected a scalable API framework to integrate LLM strategic insights as a policy prior, enabling agents to learn from models like Llama 3 and GPT-OSS via Ollama/HuggingFace.
* **Interactive Evaluation Suite:** Designed and built a custom evaluation environment (PettingZoo) and Web UI for critical live human-vs-agent testing and qualitative validation of learned strategies.

## 🛠️ System Architecture

The project consists of three main components:

1.  **The RL Agent (PPO):** A custom implementation of Proximal Policy Optimization with valid action masking, trained to handle the partial observability of Gin Rummy.
2.  **The LLM Agent:** A sophisticated wrapper that parses game states into text prompts (Chain-of-Thought) and parses LLM responses back into valid game actions.
3.  **The Orchestrator:** Manages the training curriculum, switching opponents between random agents, prior model checkpoints, and live LLM inferences based on training progress.

## 📂 Project Structure


.
├── agents/                 # Agent implementations (PPO, Random, LLM, Human)
├── artifacts/              # Trained models and checkpoints
├── config/                 # Configuration files (paths, prompts.yaml)
├── controller/             # Game logic and orchestration
├── game/                   # Gin Rummy environment wrappers and assets
├── llm/                    # API handlers for Ollama/HuggingFace interaction
├── src/                    # Utilities, logging, and UI components
├── templates/              # HTML templates for the Web UI
├── app.py                  # Flask application for web-based play
├── eval.py                 # Evaluation scripts
├── main.py                 # Main entry point
├── ppo_train.py            # PPO training pipeline script
├── environment.yml         # Conda environment definition
└── requirements.txt        # Python dependencies


## 💻 Installation

### Prerequisites

  * Python 3.10+
  * Conda (recommended)
  * [Ollama](https://ollama.com/) (for local LLM inference)

### Setup

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/nikelroid/adversarial-coevolution.git](https://github.com/nikelroid/adversarial-coevolution.git)
    cd adversarial-coevolution
    ```

2.  **Create the environment:**

    ```bash
    conda env create -f environment.yml
    conda activate rl-llm-env
    ```

    *Alternatively, using pip:*

    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## 🏃‍♂️ Usage

### 1\. Training the RL Agent

To start the PPO training pipeline with the default configuration (Curriculum Phase 1 & 2):

```bash
python ppo_train.py
```

*Check `config/` to adjust hyperparameters or curriculum stages.*

### 2\. Running the LLM Interaction

Ensure your Ollama server is running (default port 11434). To test an LLM agent:

```bash
python llm_test.py
```

### 3\. Web Interface (Human vs. Agent)

Launch the web application to play against the trained models:

```bash
python app.py
```

Open your browser at `http://localhost:5000`.

### 4\. Evaluation

To benchmark the current model against a random agent or an LLM:

```bash
python eval.py --model artifacts/models/ppo_gin_rummy/ppo_gin_rummy_final.zip
```

## 📊 Results

| Agent Type | Opponent | Win Rate | Notes |
|:---:|:---:|:---:|:---:|
| **PPO (Baseline)** | Random | 98.9% | High win rate, but prone to local optima (Gin-biased). |
| **PPO (Curriculum)** | Random | **99.1%** | Balanced strategy (Knock vs. Gin). |
| **GPT-OSS (20B)** | Random | 100% | Zero-shot performance (5-0 match). |
| **GPT-OSS (20B)** | PPO (Knock) | 60% | Competitive match (3-2 score). |

## 👥 Authors

  * **Nima Kelidari** - *Lead Engineer & RL Architecture* - [kelidari@usc.edu](mailto:kelidari@usc.edu)
  * **Mahdi Salmani** - *LLM Integration & Evaluation* - [salmanis@usc.edu](mailto:salmanis@usc.edu)
  * **Mohammadsaeed Haghi** - *Game Environment & API* - [haghim@usc.edu](mailto:haghim@usc.edu)

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgments

  * [PettingZoo](https://pettingzoo.farama.org/) for the Multi-Agent RL environments.
  * [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for reliable PPO implementations.
  * [RLCard](https://github.com/datamllab/rlcard) for game logic inspiration.

<!-- end list -->

```
```
