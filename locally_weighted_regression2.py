import numpy as np
import matplotlib.pyplot as plt

# Training data
X = np.array([1, 2, 3])
Y = np.array([1, 2, 2])

# Hyperparameters
alpha = 0.05
epochs = 300
tau = 0.5  # smaller tau = more local fit

def gradient_descent(X, Y, x_query, tau, alpha, epochs):
    # initialize params
    m, b = 0.0, 0.0
    
    # compute weights for this query
    w = np.exp(-(X - x_query)**2 / (2 * tau * tau))
    
    # gradient descent
    for _ in range(epochs):
        pred = m * X + b
        error = pred - Y
        
        dm = np.sum(w * error * X) / np.sum(w)
        db = np.sum(w * error) / np.sum(w)
        
        m -= alpha * dm
        b -= alpha * db
    
    return m, b

# Generate smooth curve
x_query_points = np.linspace(min(X)-1, max(X)+1, 100)
y_preds = []

for x_q in x_query_points:
    m_final, b_final = gradient_descent(X, Y, x_q, tau, alpha, epochs)
    y_preds.append(m_final * x_q + b_final)

# Plot
plt.scatter(X, Y, color="blue", label="Training points")
plt.plot(x_query_points, y_preds, color="red", label=f"LWR (tau={tau})")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Locally Weighted Regression (Gradient Descent)")
plt.legend()
plt.show()
