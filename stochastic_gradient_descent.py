import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Data
X = np.linspace(1, 10, 10)
Y = 2*X + 1

# Parameters
lr = 0.01
m = 0.0
b = 0.0
epochs = 100

print("Training Linear Regression with SGD")
print("Target: y = 2x + 1")
print("-" * 30)

history = []

for epoch in range(epochs):
    combined = list(zip(X, Y))
    random.shuffle(combined)
    
    epoch_loss = 0
    
    for xi, yi in combined:
        # Store parameters before each update
        history.append((m, b))
        
        y_pred = m*xi + b
        
        loss = (yi - y_pred)**2
        epoch_loss += loss
        
        grad_m = -2 * xi * (yi - y_pred)
        grad_b = -2 * (yi - y_pred)
        
        m = m - lr * grad_m
        b = b - lr * grad_b
    
    if (epoch + 1) % 20 == 0:
        avg_loss = epoch_loss / len(X)
        print(f'Epoch {epoch+1:3d}: m={m:.4f}, b={b:.4f}, Loss={avg_loss:.6f}')

# Create loss surface
m_vals = np.linspace(-0.5, 3.0, 200)
b_vals = np.linspace(-1.0, 2.5, 200)
M, B = np.meshgrid(m_vals, b_vals, indexing="ij")

preds = M[..., None] * X + B[..., None]
J_grid = 0.5 * np.mean((preds - Y)**2, axis=-1)

m_history = [h[0] for h in history]
b_history = [h[1] for h in history]

# Static plot first
plt.figure(figsize=(12, 6))

cont = plt.contour(M, B, J_grid, colors='black', levels=30, alpha=0.4, linewidths=0.5)
plt.contourf(M, B, J_grid, cmap='viridis', alpha=0.7, levels=50)
plt.clabel(cont, inline=True, fontsize=8, fmt='%.3f')

plt.plot(m_history, b_history, 'r-', alpha=0.7, linewidth=2, label='SGD Path')

plt.plot(m_history[0], b_history[0], 'go', markersize=8, label='Start (0,0)')
plt.plot(m_history[-1], b_history[-1], 'bo', markersize=8, label=f'End ({m:.3f},{b:.3f})')

plt.plot(2, 1, 'w*', markersize=12, markeredgecolor='black',
         markeredgewidth=2, label='True Optimum (2,1)')

plt.colorbar(label='Loss')
plt.title('Stochastic Gradient Descent - Loss Surface and Optimization Path')
plt.xlabel('m (slope)')
plt.ylabel('b (intercept)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.xlim(m_vals[0], m_vals[-1])
plt.ylim(b_vals[0], b_vals[-1])

plt.tight_layout()
plt.show()

print("\nFinal Results:")
print(f'Learned parameters: y = {m:.4f}x + {b:.4f}')
print(f'Target parameters:  y = 2.0000x + 1.0000')
print(f'Error in m: {abs(m - 2.0):.4f}')
print(f'Error in b: {abs(b - 1.0):.4f}')
print(f'Total parameter updates: {len(history)}')

# Animation function
def create_animated_plots():
    """Create animated contour plot and line fitting visualization"""
    print("\nCreating animated visualization...")
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Line fitting animation
    ax1.scatter(X, Y, color='red', s=50, alpha=0.8, label='Data points', zorder=3)
    line_current, = ax1.plot([], [], 'blue', linewidth=2, label='Current fit')
    line_true, = ax1.plot(X, 2*X + 1, 'green', linestyle='--', alpha=0.7, 
                         linewidth=2, label='True line: y=2x+1')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Animated Line Fitting (SGD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, 10.5)
    ax1.set_ylim(0, 22)
    
    # Right plot: Contour plot with path animation
    cont = ax2.contour(M, B, J_grid, colors='black', levels=20, alpha=0.6)
    ax2.contourf(M, B, J_grid, cmap='viridis', alpha=0.6)
    ax2.clabel(cont, inline=True, fontsize=8)
    
    # Initialize animated elements for contour plot
    path_line, = ax2.plot([], [], 'red', linewidth=2, alpha=0.8, label='SGD Path')
    current_point = ax2.scatter([], [], color='yellow', s=100, marker='o', 
                              edgecolors='black', linewidth=2, label='Current position', zorder=5)
    start_point = ax2.scatter(m_history[0], b_history[0], color='green', s=80, 
                            marker='s', edgecolors='black', label='Start', zorder=5)
    true_point = ax2.scatter(2, 1, color='white', s=120, marker='*', 
                           edgecolors='black', linewidth=2, label='True Optimum', zorder=5)
    
    ax2.set_xlabel('m (slope)')
    ax2.set_ylabel('b (intercept)')
    ax2.set_title('Animated SGD on Cost Contours')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(m_vals[0], m_vals[-1])
    ax2.set_ylim(b_vals[0], b_vals[-1])
    
    # Animation function
    def animate(frame):
        if frame < len(history):
            # Get current parameters
            current_m, current_b = history[frame]
            
            # Update line fit on left plot
            Y_current = current_m * X + current_b
            line_current.set_data(X, Y_current)
            
            # Update path on contour plot (show path up to current frame)
            # Show every 5th point to make it less cluttered for SGD
            show_every = max(1, len(history) // 200)  # Show ~200 points max
            path_m = [h[0] for h in history[:frame+1:show_every]]
            path_b = [h[1] for h in history[:frame+1:show_every]]
            path_line.set_data(path_m, path_b)
            
            # Update current position on contour plot
            current_point.set_offsets([[current_m, current_b]])
            
            # Update title with current iteration info
            cost = 0.5 * np.mean((current_m * X + current_b - Y)**2)
            epoch_num = frame // len(X) + 1  # Approximate epoch number
            fig.suptitle(f'Update {frame}: Epoch ~{epoch_num}, m={current_m:.3f}, b={current_b:.3f}, Cost={cost:.3f}', 
                        fontsize=12)
        
        return line_current, path_line, current_point
    
    # Create and run animation
    print("Starting animation...")
    # Sample frames for faster animation (every 10th update for SGD)
    sample_frames = min(500, len(history))  # Limit to 500 frames max
    frame_step = max(1, len(history) // sample_frames)
    
    anim = FuncAnimation(fig, animate, frames=range(0, len(history), frame_step), 
                        interval=100, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    initial_cost = 0.5 * np.mean((0*X + 0 - Y)**2)
    final_cost = 0.5 * np.mean((m*X + b - Y)**2)
    print(f"Animation complete! Converged from cost {initial_cost:.3f} to {final_cost:.6f}")
    
    return anim

# Run the animation
print("\n" + "="*50)
print("CREATING ANIMATION")
print("="*50)
animation = create_animated_plots()