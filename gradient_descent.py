import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

X = np.linspace(1, 10, 20)
Y = 2*X + 1

m = 0.0
b = 0.0
alpha = 0.02
epochs = 250
history = []

def gradient_descent(m, b):
    for epoch in range(epochs):
        pred = m*X + b
        dm = np.mean((pred - Y) * X)
        db = np.mean(pred - Y)
        
        m = m - alpha * dm
        b = b - alpha * db
        history.append((m, b))
    
    return m, b

def main():
    m_vals = np.linspace(0, 4, 200)
    b_vals = np.linspace(-2, 4, 200)
    M, B = np.meshgrid(m_vals, b_vals, indexing='ij')
    preds = M[..., None] * X + B[..., None]
    J_grid = 0.5 * np.mean((preds - Y)**2, axis=-1)
    
    final_m, final_b = gradient_descent(m, b)
    
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    
    ax1.scatter(X, Y, color='red', label='Data points') 
    Y_pred = final_m * X + final_b
    ax1.plot(X, Y_pred, color='blue', label=f'Fitted line: y = {final_m:.2f}x + {final_b:.2f}')
    
    ax1.set_xlabel('X') 
    ax1.set_ylabel('Y') 
    ax1.legend()
    ax1.set_title('Linear Regression with Gradient Descent')  
    ax1.grid(True, alpha=0.3)
    

    cont = ax2.contour(M, B, J_grid, colors='black', levels=20)
    ax2.contourf(M, B, J_grid, cmap='plasma', alpha=0.6)  
    ax2.clabel(cont, inline=True, fontsize=8)  
    
   
    path_m = [point[0] for point in history]
    path_b = [point[1] for point in history]
    ax2.plot(path_m, path_b, 'white', linewidth=3, alpha=0.9, label='GD Path')
    ax2.plot(path_m[0], path_b[0], 'go', markersize=8, label='Start (0,0)')
    ax2.plot(path_m[-1], path_b[-1], 'ro', markersize=8, label=f'End ({final_m:.2f},{final_b:.2f})')
    ax2.plot(2, 1, 'r*', markersize=12, label='True Optimum (2,1)')
    
    ax2.set_xlabel('m (slope)')  
    ax2.set_ylabel('b (intercept)') 
    ax2.legend()
    ax2.set_title('Cost Function with Gradient Descent Path') 
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Final parameters: m = {final_m:.4f}, b = {final_b:.4f}")
    print(f"True parameters: m = 2.0000, b = 1.0000")
    print(f"Number of iterations: {len(history)}")
    print(f"Final cost: {0.5 * np.mean((final_m * X + final_b - Y)**2):.6f}")
    
  
    create_animated_plots()

def create_animated_plots():
    """Create animated contour plot and line fitting visualization"""
    print("\nCreating animated visualization...")
    
  
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
   
    m_vals = np.linspace(1, 4, 200)
    b_vals = np.linspace(-2, 4, 200)
    M, B = np.meshgrid(m_vals, b_vals, indexing='ij')
    preds = M[..., None] * X + B[..., None]
    J_grid = 0.5 * np.mean((preds - Y)**2, axis=-1)
    

    ax1.scatter(X, Y, color='red', s=50, alpha=0.8, label='Data points', zorder=3)
    line_current, = ax1.plot([], [], 'blue', linewidth=2, label='Current fit')
    line_true, = ax1.plot(X, 2*X + 1, 'green', linestyle='--', alpha=0.7, 
                         linewidth=2, label='True line: y=2x+1')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Animated Line Fitting')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.5, 10.5)
    ax1.set_ylim(0, 22)
    
   
    cont = ax2.contour(M, B, J_grid, colors='black', levels=20, alpha=0.6)
    ax2.contourf(M, B, J_grid, cmap='plasma', alpha=0.6)
    ax2.clabel(cont, inline=True, fontsize=8)
    
    # Initialize animated elements for contour plot
    path_line, = ax2.plot([], [], 'white', linewidth=3, alpha=0.9, label='GD Path')
    current_point = ax2.scatter([], [], color='red', s=100, marker='o', 
                              edgecolors='black', linewidth=2, label='Current position', zorder=5)
    start_point = ax2.scatter(history[0][0], history[0][1], color='green', s=80, 
                            marker='s', edgecolors='black', label='Start', zorder=5)
    true_point = ax2.scatter(2, 1, color='cyan', s=120, marker='*', 
                           edgecolors='navy', linewidth=2, label='True Optimum', zorder=5)
    
    ax2.set_xlabel('m (slope)')
    ax2.set_ylabel('b (intercept)')
    ax2.set_title('Animated Gradient Descent on Cost Contours')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Animation function
    def animate(frame):
        if frame < len(history):
            # Get current parameters
            current_m, current_b = history[frame]
            
            # Update line fit on left plot
            Y_current = current_m * X + current_b
            line_current.set_data(X, Y_current)
            
            # Update path on contour plot (show path up to current frame)
            path_m = [h[0] for h in history[:frame+1]]
            path_b = [h[1] for h in history[:frame+1]]
            path_line.set_data(path_m, path_b)
            
            # Update current position on contour plot
            current_point.set_offsets([[current_m, current_b]])
            
            # Update title with current iteration info
            cost = 0.5 * np.mean((current_m * X + current_b - Y)**2)
            fig.suptitle(f'Iteration {frame}: m={current_m:.3f}, b={current_b:.3f}, Cost={cost:.3f}', 
                        fontsize=14)
        
        return line_current, path_line, current_point
    
    # Create and run animation
    print("Starting animation...")
    anim = FuncAnimation(fig, animate, frames=len(history)+20, 
                        interval=150, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Animation complete! Converged from cost {0.5 * np.mean((0*X + 0 - Y)**2):.3f} to final cost")

if __name__ == "__main__":
    main()