# Ploynomial Regression & Overfitting

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Create dataset
np.random.seed(0)
X = np.linspace(-3, 3, 100).reshape(-1,1)
y = X**2 + np.random.randn(100,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

degrees = [1,2,3,5,10]

plt.scatter(X, y, color='black')

for d in degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_poly_train, y_train)

    y_pred = model.predict(poly.transform(X))

    train_error = mean_squared_error(y_train, model.predict(X_poly_train))
    test_error = mean_squared_error(y_test, model.predict(X_poly_test))

    print(f"Degree {d} -> Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")

    plt.plot(X, y_pred, label=f"Degree {d}")

plt.legend()
plt.title("Polynomial Regression & Overfitting")
plt.savefig("plots/polynomial_models.png")
plt.show()
