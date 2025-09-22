import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([1,2,3])
y_train = np.array([1,2,2])

def locally_weighted_regression(x_query, X, y, tau=1.0):
    # weights relative to query point
    w = np.exp(-(X - x_query)**2 / (2 * tau * tau))
    W = np.diag(w)

    # construct design matrix [1, x]
    X_design = np.c_[np.ones(len(X)), X]   # shape (n,2)

    # weighted normal equation
    theta = np.linalg.pinv(X_design.T @ W @ X_design) @ (X_design.T @ W @ y)

    # predict at query point
    x_vec = np.array([1, x_query])
    y_pred = x_vec @ theta
    return y_pred

# generate curve
x_query_points = np.linspace(0, 4, 100)
y_preds = [locally_weighted_regression(x, X_train, y_train, tau=0.5) 
           for x in x_query_points]

# plot
plt.scatter(X_train, y_train, color="red", label="Training data")
plt.plot(x_query_points, y_preds, color="blue", label="LWR prediction (tau=0.5)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Locally Weighted Regression (no design matrix)")
plt.legend()
plt.show()
