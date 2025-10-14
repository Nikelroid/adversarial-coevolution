import numpy as np
import matplotlib.pyplot as plt

# Define the function and its gradient
def f(w):
    """Objective function: f(w) = w^2 - 3w + 11"""
    return w**2 - 3*w + 11

def gradient_f(w):
    """Gradient of f(w): f'(w) = 2w - 3"""
    return 2*w - 3

# SGD implementation
def sgd(learning_rate, initial_w=0.0, num_iterations=100):
    """
    Perform gradient descent optimization
    
    Args:
        learning_rate: Step size for gradient descent
        initial_w: Starting value of w
        num_iterations: Number of iterations to run
    
    Returns:
        w_history: List of w values at each iteration
        f_history: List of function values at each iteration
    """
    w = initial_w
    w_history = [w]
    f_history = [f(w)]
    
    for i in range(num_iterations):
        # Compute gradient
        grad = gradient_f(w)
        
        # Update w using gradient descent rule
        w = w - learning_rate * grad
        
        # Store history
        w_history.append(w)
        f_history.append(f(w))
    
    return w_history, f_history

# Test with different learning rates
learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5, 0.8,1]
initial_w = 10.0  # Starting point
num_iterations = 50

# Store results for each learning rate
results = {}
for lr in learning_rates:
    w_hist, f_hist = sgd(lr, initial_w, num_iterations)
    results[lr] = {'w': w_hist, 'f': f_hist}
    print(f"Learning rate: {lr:.2f} -> Final w: {w_hist[-1]:.4f}, Final f(w): {f_hist[-1]:.4f}")

# Analytical solution (for comparison)
w_optimal = 1.5  # Derivative = 0 when 2w - 3 = 0, so w = 1.5
f_optimal = f(w_optimal)
print(f"\nOptimal solution: w* = {w_optimal}, f(w*) = {f_optimal}")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Function value convergence
ax1 = axes[0]
for lr in learning_rates:
    ax1.plot(results[lr]['f'], label=f'LR = {lr}', linewidth=2)
ax1.axhline(y=f_optimal, color='red', linestyle='--', linewidth=2, label='Optimal f(w)')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('f(w)', fontsize=12)
ax1.set_title('Convergence of Function Value', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Parameter convergence
ax2 = axes[1]
for lr in learning_rates:
    ax2.plot(results[lr]['w'], label=f'LR = {lr}', linewidth=2)
ax2.axhline(y=w_optimal, color='red', linestyle='--', linewidth=2, label='Optimal w')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('w', fontsize=12)
ax2.set_title('Convergence of Parameter w', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional visualization: Convergence landscape
fig2, ax = plt.subplots(figsize=(10, 6))

# Plot the function
w_range = np.linspace(-2, 12, 300)
f_range = [f(w) for w in w_range]
ax.plot(w_range, f_range, 'k-', linewidth=2, label='f(w) = w² - 3w + 11')

# Plot convergence paths for selected learning rates
selected_lrs = [0.01, 0.1, 0.5]
colors = ['blue', 'green', 'orange']

for lr, color in zip(selected_lrs, colors):
    w_hist = results[lr]['w']
    f_hist = results[lr]['f']
    ax.plot(w_hist, f_hist, 'o-', color=color, alpha=0.6, 
            markersize=4, linewidth=1.5, label=f'Path (LR={lr})')

# Mark optimal point
ax.plot(w_optimal, f_optimal, 'r*', markersize=20, label='Optimal point')

ax.set_xlabel('w', fontsize=12)
ax.set_ylabel('f(w)', fontsize=12)
ax.set_title('Optimization Paths on Function Landscape', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()