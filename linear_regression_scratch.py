"""
Task 3.1 — Linear Regression from Scratch using Gradient Descent

We implement Simple Linear Regression without using sklearn.
The model learns the equation:

    y = w*x + b

Where:
    w = weight (slope)
    b = bias (intercept)

We train the model using Gradient Descent by minimizing Mean Squared Error (MSE).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Create plots folder if not exists
if not os.path.exists("plots"):
    os.makedirs("plots")

# -------------------------------------------------
# Step 1: Create Synthetic Dataset
# -------------------------------------------------
"""
We generate data using the true equation:

    y = 2x + 1 + noise

Noise is added to simulate real-world imperfect data.
"""

np.random.seed(42)

X = np.linspace(0, 10, 100)      # 100 values between 0 and 10
noise = np.random.randn(100)    # random noise
y = 2 * X + 1 + noise           # true relationship

# Convert to column vectors
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

# -------------------------------------------------
# Step 2: Linear Regression Class
# -------------------------------------------------

class LinearRegression:
    """
    This class implements Linear Regression using Gradient Descent.

    Model equation:
        y_pred = w*x + b
    """

    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr              # learning rate
        self.epochs = epochs     # number of training iterations

    def fit(self, X, y):
        """
        Train the model using Gradient Descent.

        Mathematical idea:

        We minimize Mean Squared Error (MSE):

            MSE = (1/m) * Σ (y - y_pred)^2

        Using Gradient Descent update rules:

            w = w - α * dMSE/dw
            b = b - α * dMSE/db
        """

        self.m, self.n = X.shape   # number of samples and features
        self.W = np.zeros((self.n, 1))   # initialize weight with 0
        self.b = 0                        # initialize bias with 0
        self.costs = []                  # store loss values

        for i in range(self.epochs):

            # -----------------------------
            # Forward Propagation
            # -----------------------------
            """
            Prediction formula:

                y_pred = w*x + b
            """
            y_pred = np.dot(X, self.W) + self.b

            # -----------------------------
            # Cost Function (MSE)
            # -----------------------------
            """
            Mean Squared Error:

                MSE = (1/m) * Σ (y - y_pred)^2

            This measures how far predictions are from actual values.
            """
            cost = np.mean((y - y_pred) ** 2)
            self.costs.append(cost)

            # -----------------------------
            # Gradient Calculation
            # -----------------------------
            """
            Partial derivatives of MSE:

            dMSE/dw = (-2/m) * Σ x * (y - y_pred)
            dMSE/db = (-2/m) * Σ (y - y_pred)
            """

            dW = (-2/self.m) * np.dot(X.T, (y - y_pred))
            db = (-2/self.m) * np.sum(y - y_pred)

            # -----------------------------
            # Parameter Update
            # -----------------------------
            """
            Gradient Descent update:

                w = w - α * dMSE/dw
                b = b - α * dMSE/db
            """

            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        """
        Make predictions using learned parameters.
        """
        return np.dot(X, self.W) + self.b

    def r2_score(self, y, y_pred):
        """
        R² Score formula:

            R² = 1 - (SS_res / SS_total)

        where:
            SS_res = Σ(y - y_pred)²
            SS_total = Σ(y - mean(y))²

        R² tells how well the model fits the data.
        """
        ss_total = np.sum((y - np.mean(y))**2)
        ss_res = np.sum((y - y_pred)**2)
        return 1 - (ss_res / ss_total)

# -------------------------------------------------
# Step 3: Train Model
# -------------------------------------------------

model = LinearRegression(lr=0.01, epochs=1000)
model.fit(X, y)

y_pred = model.predict(X)
r2 = model.r2_score(y, y_pred)

print("Learned Weight (w):", model.W[0][0])
print("Learned Bias (b):", model.b)
print("R² Score:", r2)

# -------------------------------------------------
# Step 4: Plot Regression Line
# -------------------------------------------------

plt.scatter(X, y, label="Actual Data")
plt.plot(X, y_pred, color='red', label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression from Scratch")
plt.legend()
plt.savefig("plots/linear_regression.png")
plt.show()

# -------------------------------------------------
# Step 5: Plot Cost Convergence
# -------------------------------------------------

plt.plot(model.costs)
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function Convergence")
plt.savefig("plots/cost_convergence.png")
plt.show()
