import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# 1. Generate Data
np.random.seed(42)

X = np.sort(6 * np.random.rand(100, 1) + 4)

y = np.sin(X).ravel() + np.random.normal(
    0, 0.2, X.shape[0]
)


# 2. Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# 3. Create degree-15 polynomial features

poly_degree = 15

poly = PolynomialFeatures(
    degree=poly_degree
)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


# 4. Scale polynomial features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_poly
)

X_test_scaled = scaler.transform(
    X_test_poly
)


# 5. Define regularization parameters
lambdas = np.logspace(-4, 4, 200)

train_errors = []
test_errors = []


# 6. Train Ridge Regression for each lambda
for lam in lambdas:

    ridge = Ridge(alpha=lam)

    ridge.fit(
        X_train_scaled,
        y_train
    )

    # Training prediction
    y_train_pred = ridge.predict(
        X_train_scaled
    )

    # Testing prediction
    y_test_pred = ridge.predict(
        X_test_scaled
    )

    # Calculate Mean Squared Error
    train_mse = mean_squared_error(
        y_train,
        y_train_pred
    )

    test_mse = mean_squared_error(
        y_test,
        y_test_pred
    )

    train_errors.append(train_mse)
    test_errors.append(test_mse)


# 7. Find best lambda
best_index = np.argmin(test_errors)

best_lambda = lambdas[best_index]
best_test_error = test_errors[best_index]

print("\nBest Regularization Parameter:")
print("Alpha =", best_lambda)

print("\nMinimum Testing MSE:")
print(best_test_error)


# 8. Plot error curves
plt.figure(figsize=(10, 6))

plt.plot(
    lambdas,
    train_errors,
    label="Training Error",
    linewidth=2
)

plt.plot(
    lambdas,
    test_errors,
    label="Testing Error",
    linewidth=2,
    linestyle="--"
)

plt.xscale("log")

plt.xlabel(
    "Regularization Parameter (Lambda / Alpha)",
    fontsize=12
)

plt.ylabel(
    "Mean Squared Error",
    fontsize=12
)

plt.title(
    "Ridge Regression Regularization - Degree 15",
    fontsize=14
)

plt.legend(fontsize=12)

plt.grid(
    True,
    which="both",
    linestyle="--"
)

plt.show()