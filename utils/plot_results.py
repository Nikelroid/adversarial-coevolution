
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_training_results(log_dir="./logs/", output_file="training_results.png"):
    """
    Reads monitor.csv files from Stable Baselines3 logs and plots rewards.
    """
    print(f"Looking for logs in {log_dir}...")
    
    # SB3 monitor files are named like 0.monitor.csv, 1.monitor.csv, etc.
    monitor_files = [f for f in os.listdir(log_dir) if f.endswith(".monitor.csv")]
    
    if not monitor_files:
        print("No monitor files found. Have you run training?")
        return
        
    print(f"Found {len(monitor_files)} log files. Aggregating...")
    
    data_frames = []
    for file in monitor_files:
        path = os.path.join(log_dir, file)
        try:
            # Skip first line (header info)
            df = pd.read_csv(path, skiprows=1)
            data_frames.append(df)
        except Exception as e:
            print(f"Skipping {file}: {e}")
            
    if not data_frames:
        print("No valid data found.")
        return

    # Combine all data
    full_df = pd.concat(data_frames, ignore_index=True)
    
    # Sort by wall time or steps? Usually rolling mean over episodes is best.
    # But since it's multiple envs, 'l' is length, 'r' is reward, 't' is time.
    
    # Plotting Moving Average of Reward
    plt.figure(figsize=(10, 5))
    
    # Rolling window
    window = 100
    if len(full_df) > window:
        full_df['rolling_reward'] = full_df['r'].rolling(window=window).mean()
        plt.plot(full_df['rolling_reward'], label=f'Reward ({window}-ep avg)')
    else:
        plt.plot(full_df['r'], label='Reward', alpha=0.3)

    plt.title("Training Progress: Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
if __name__ == "__main__":
    plot_training_results()
