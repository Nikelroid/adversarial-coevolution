import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['text.color'] = '#333333'

def plot_analysis_charts():
    fig = plt.figure(figsize=(15, 5))
    
    # --- CHART 1: Prompt Efficacy (Bar Chart) ---
    ax1 = fig.add_subplot(131)
    
    # Data: Valid Action %
    prompts = ['Simple', 'Complex (CoT)']
    validity = [79.3, 98.2] # Averages from your table
    colors = ['#F87171', '#34D399'] # Red, Green
    
    bars = ax1.bar(prompts, validity, color=colors, alpha=0.9, width=0.6)
    
    ax1.set_ylim(0, 110)
    ax1.set_ylabel("Valid Action Rate (%)", fontsize=11, weight='bold')
    ax1.set_title("Impact of Chain-of-Thought\non Rule Adherence", fontsize=12, weight='bold', pad=10)
    
    # Add labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                 f'{height}%', ha='center', va='bottom', fontsize=12, weight='bold')

    # --- CHART 2: Latency vs. Utility (Scatter Plot) ---
    ax2 = fig.add_subplot(132)
    
    # Data points (Latency, Success/Utility Score - subjective based on results)
    # Models: [Llama-70b, GPT-OSS-20b, Gemma-27b, Qwen-VL-32b]
    latency = [27.0, 9.5, 9.4, 145.0]
    utility = [95, 90, 85, 10] # Utility is rough proxy for "Is it a good teacher?"
    sizes = [700, 200, 270, 320] # Bubble size = Model Params (scaled)
    labels = ['Llama-70b', 'GPT-OSS-20b', 'Gemma-27b', 'Qwen-VL']
    colors = ['#60A5FA', '#10B981', '#F59E0B', '#9CA3AF']

    scatter = ax2.scatter(latency, utility, s=sizes, c=colors, alpha=0.7, edgecolors='w', linewidth=2)
    
    # Labels
    for i, txt in enumerate(labels):
        ax2.text(latency[i]+5, utility[i], txt, fontsize=10, weight='bold')
        
    ax2.set_xlabel("Inference Latency (s)", fontsize=11, weight='bold')
    ax2.set_ylabel("Strategic Utility (Est.)", fontsize=11, weight='bold')
    ax2.set_title("The Trade-off:\nLatency vs. Utility", fontsize=12, weight='bold', pad=10)
    ax2.set_xlim(0, 160)
    
    # Annotation for the "Sweet Spot"
    rect = plt.Rectangle((5, 80), 30, 25, fill=False, edgecolor='#10B981', linewidth=2, linestyle='--')
    ax2.add_patch(rect)
    ax2.text(20, 75, "Optimal Teacher\nZone", ha='center', color='#10B981', fontsize=9, weight='bold')

    # --- CHART 3: Head-to-Head Result (Pie/Donut) ---
    ax3 = fig.add_subplot(133)
    
    # Exp 16 Data: LLM vs RL (Best of 5)
    # LLM won 3, RL won 2
    sizes = [3, 2]
    labels = ['LLM (GPT-OSS)', 'RL Agent']
    colors = ['#8B5CF6', '#F472B6'] # Violet, Pink
    
    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct=lambda p: '{:.0f} Wins'.format(p * sum(sizes) / 100),
                                       startangle=90, colors=colors, 
                                       wedgeprops={'width': 0.4, 'edgecolor': 'w'},
                                       textprops={'fontsize': 11, 'weight': 'bold'})
    
    ax3.text(0, 0, 'Match\nResult', ha='center', va='center', fontsize=12, weight='bold')
    ax3.set_title("Zero-Shot LLM vs.\nTrained RL Agent", fontsize=12, weight='bold', pad=10)

    # --- FINALIZE ---
    plt.tight_layout()
    plt.savefig('modern_analysis_panel.png', dpi=300)
    print("Generated 'modern_analysis_panel.png'")

if __name__ == "__main__":
    plot_analysis_charts()