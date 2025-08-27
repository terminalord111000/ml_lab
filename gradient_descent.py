import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import time


def compute_cost(m, b, X, Y):
    pred = m * X + b
    return 0.5 * np.mean((pred - Y) ** 2)


def gradient_descent(X, Y, alpha=0.01, iterations=200, tol=1e-12):
    m, b = 0.0, 0.0
    n = len(X)
    history = []

    for iteration in range(iterations):
        preds = m * X + b
        dm = (1 / n) * np.sum((preds - Y) * X)
        db = (1 / n) * np.sum((preds - Y))
        m -= alpha * dm
        b -= alpha * db
        cost = compute_cost(m, b, X, Y)
        history.append((m, b, cost))

        if iteration > 0 and abs(history[-1][2] - history[-2][2]) < tol:
            break

    return history


def main():
    np.random.seed(42)
    X = np.linspace(1, 10, 20)
    Y = 2 * X + 1 + np.random.randn(20)

    m_vals = np.linspace(0, 4, 200)
    b_vals = np.linspace(-2, 4, 200)
    M, B = np.meshgrid(m_vals, b_vals, indexing="ij")
    preds = M[..., None] * X + B[..., None]
    J_grid = 0.5 * np.mean((preds - Y) ** 2, axis=-1)

    history = gradient_descent(X, Y, alpha=0.01, iterations=200)
    path_m = [h[0] for h in history]
    path_b = [h[1] for h in history]
    path_J = [h[2] for h in history]

    print(
        f"Finished: iterations={len(history)}, "
        f"m={path_m[-1]:.6f}, b={path_b[-1]:.6f}, J={path_J[-1]:.8f}"
    )

    fig = plt.figure(figsize=(12, 5))

    # 1. 2D contour plot
    ax1 = fig.add_subplot(1, 2, 1)
    contours = ax1.contour(M, B, J_grid, levels=20, linewidths=0.6, colors="black")
    ax1.contourf(M, B, J_grid, levels=20, cmap="plasma", alpha=0.7)
    ax1.plot(path_m, path_b, color="lime", linewidth=2, label="Gradient descent path")
    ax1.scatter(path_m[0], path_b[0], color="red", s=80, marker="s", edgecolors="black", label="Start")
    ax1.scatter(path_m[-1], path_b[-1], color="gold", s=120, marker="*", edgecolors="black", label="End")
    ax1.scatter(2, 1, color="cyan", s=150, marker="^", edgecolors="navy", linewidth=2, label="True (2,1)")
    ax1.set_xlabel("m (slope)")
    ax1.set_ylabel("b (intercept)")
    ax1.set_title("2D Contour")
    ax1.legend()

    # 2. 3D surface
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(M, B, J_grid, cmap='plasma', alpha=0.6, linewidth=0, antialiased=True)
    ax2.plot(path_m, path_b, path_J, color='lime', linewidth=3)
    ax2.scatter(path_m[0], path_b[0], path_J[0], color='red', s=80, marker='s')
    ax2.scatter(path_m[-1], path_b[-1], path_J[-1], color='gold', s=120, marker='*')
    ax2.set_xlabel("m (slope)")
    ax2.set_ylabel("b (intercept)")
    ax2.set_zlabel("Cost J")
    ax2.set_title("3D Surface")

    # Animation setup
    print("\nCreating animated 3D visualization...")
    fig_anim = plt.figure(figsize=(12, 8))
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    surface = ax_anim.plot_surface(M, B, J_grid, cmap='plasma', alpha=0.4, linewidth=0, antialiased=True)
    

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()