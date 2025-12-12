
# ☁️ Server / Google Colab Deployment Guide

This guide explains how to deploy the Adversarial Co-Evolution project to a remote server (e.g., AWS, GCP, University Cluster) or Google Colab.

## Option 1: Google Colab (Free GPU)

Google Colab is the easiest way to run training if you don't have a powerful GPU locally.

### Step 1: Prepare the Code
1.  Zip your entire project folder:
    ```bash
    zip -r project.zip . -x "*.git*" "artifacts/*" "logs/*"
    ```
2.  Upload `project.zip` to your Google Drive.

### Step 2: Create a Colab Notebook
1.  Open [Google Colab](https://colab.research.google.com/).
2.  Create a new notebook.
3.  Change Runtime to **GPU** (Runtime > Change runtime type > T4 GPU).

### Step 3: Run the following cells

**Cell 1: Connect to Drive & Unzip**
```python
from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/project.zip /content/
!unzip -q project.zip -d /content/project
%cd /content/project
```

**Cell 2: Install Dependencies**
```python
!pip install stable-baselines3[extra] pettingzoo[classic] shimmy gymnasium pygame
!pip install -r requirements.txt
# If using Ollama, you might need to install it:
!curl -fsSL https://ollama.com/install.sh | sh
```

**Cell 3: Start Ollama (Background)**
```python
import subprocess
# Start Ollama server in background
subprocess.Popen(["ollama", "serve"])
# Pull model
!ollama pull llama3.2:1b
```

**Cell 4: Run Training**
```python
!python ppo_train.py --train --timesteps 100000 --num-env 8 --evaluator expert
```

**Cell 5: Download Results**
```python
!zip -r results.zip artifacts/ logs/
from google.colab import files
files.download('results.zip')
```

---

## Option 2: Remote Server (SSH)

For persistent training on a university server or cloud instance (e.g., EC2 `g4dn.xlarge`).

### Step 1: Upload Code
User `scp` or `rsync` to upload your code:
```bash
rsync -avz --exclude '.git' --exclude 'artifacts' ./ user@server_ip:~/adversarial-coevolution
```

### Step 2: Setup Environment (Miniconda)
SSH into the server:
```bash
ssh user@server_ip
cd adversarial-coevolution

# Install Miniconda if needed
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Create Env
conda create -n rl_env python=3.10 -y
conda activate rl_env
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
pip install "stable-baselines3[extra]" "pettingzoo[classic]"
```

### Step 4: Run with Dashboard (Recommended)
You can use `screen` or `tmux` to keep the dashboard running.

```bash
# Start a new screen session
screen -S dashboard

# Run the launch script
chmod +x start_dashboard.sh
./start_dashboard.sh
```
*Press `Ctrl+A` then `D` to detach.*

### Accessing the Dashboard
If the server has a public IP, allow port `8001` in the firewall.
If not, use **SSH Tunneling**:
```bash
# Run this on your LOCAL machine
ssh -L 8001:localhost:8001 user@server_ip
```
Now open `http://localhost:8001` on your local laptop!

---

## Generating Results (Plotting)

After training, log files are saved in `logs/`. run:
```bash
python utils/plot_results.py
```
This generates `training_results.png`.
