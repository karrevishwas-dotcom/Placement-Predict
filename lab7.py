# ============================================================
# LAB 7: LOGISTIC REGRESSION
# Binary and Multiclass Classification
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)


# ============================================================
# PART A: BINARY CLASSIFICATION
# ============================================================

print("==========================================")
print("PART A: BINARY CLASSIFICATION")
print("==========================================")

# Generate binary classification dataset
X_binary, y_binary = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_classes=2,
    random_state=42
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_binary,
    y_binary,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Create Logistic Regression model
binary_model = LogisticRegression()

# Train model
binary_model.fit(X_train, y_train)

# Predict test data
y_pred = binary_model.predict(X_test)

# Classification report
print("\n--- Binary Classification Report ---")
print(classification_report(y_test, y_pred))

# Accuracy
binary_accuracy = accuracy_score(y_test, y_pred)

print("Binary Accuracy:", binary_accuracy)


# ============================================================
# BINARY CLASSIFICATION DECISION BOUNDARY
# ============================================================

# Create grid for decision boundary
x_min, x_max = X_binary[:, 0].min() - 1, X_binary[:, 0].max() + 1
y_min, y_max = X_binary[:, 1].min() - 1, X_binary[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Predict classes for grid points
grid_predictions = binary_model.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

grid_predictions = grid_predictions.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    grid_predictions,
    alpha=0.3
)

plt.scatter(
    X_binary[:, 0],
    X_binary[:, 1],
    c=y_binary,
    edgecolor="k"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression - Binary Classification")

plt.show()


# ============================================================
# PART B: MULTICLASS CLASSIFICATION
# ============================================================

print("\n==========================================")
print("PART B: MULTICLASS CLASSIFICATION")
print("==========================================")

# Generate multiclass dataset
X_multi, y_multi = make_blobs(
    n_samples=1500,
    centers=3,
    n_features=2,
    cluster_std=1.5,
    random_state=42
)

# Split data
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi,
    y_multi,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train_multi))
print("Testing samples:", len(X_test_multi))


# ============================================================
# MULTINOMIAL LOGISTIC REGRESSION
# ============================================================

# Logistic Regression for multiclass classification
multinomial_model = LogisticRegression(
    solver="lbfgs"
)

# Train model
multinomial_model.fit(
    X_train_multi,
    y_train_multi
)

# Predict
multinomial_pred = multinomial_model.predict(
    X_test_multi
)

# Accuracy
multinomial_accuracy = accuracy_score(
    y_test_multi,
    multinomial_pred
)

print("\n--- Multinomial Logistic Regression ---")
print("Accuracy:", multinomial_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test_multi,
        multinomial_pred
    )
)


# ============================================================
# ONE-VS-REST (OvR) LOGISTIC REGRESSION
# ============================================================

# In scikit-learn 1.9.1, use OneVsRestClassifier
# instead of multi_class="ovr"

ovr_model = OneVsRestClassifier(
    LogisticRegression(
        solver="lbfgs"
    )
)

# Train OvR model
ovr_model.fit(
    X_train_multi,
    y_train_multi
)

# Predict
ovr_pred = ovr_model.predict(
    X_test_multi
)

# Accuracy
ovr_accuracy = accuracy_score(
    y_test_multi,
    ovr_pred
)

print("\n--- One-vs-Rest (OvR) Logistic Regression ---")
print("Accuracy:", ovr_accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test_multi,
        ovr_pred
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test_multi,
    ovr_pred
)

print("\n--- Confusion Matrix ---")
print(cm)

# Plot confusion matrix
plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - Multiclass Logistic Regression")

plt.show()


# ============================================================
# MULTICLASS DECISION BOUNDARY
# ============================================================

# Create grid
x_min, x_max = X_multi[:, 0].min() - 1, X_multi[:, 0].max() + 1
y_min, y_max = X_multi[:, 1].min() - 1, X_multi[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Predict grid points
multi_grid_predictions = ovr_model.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

multi_grid_predictions = multi_grid_predictions.reshape(
    xx.shape
)

# Plot multiclass decision boundary
plt.figure(figsize=(8, 6))

plt.contourf(
    xx,
    yy,
    multi_grid_predictions,
    alpha=0.3
)

plt.scatter(
    X_multi[:, 0],
    X_multi[:, 1],
    c=y_multi,
    edgecolor="k"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression - Multiclass Classification")

plt.show()


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n==========================================")
print("FINAL RESULTS")
print("==========================================")

print("Binary Accuracy:", binary_accuracy)

print(
    "Multinomial Accuracy:",
    multinomial_accuracy
)

print(
    "One-vs-Rest Accuracy:",
    ovr_accuracy
)

print("\nLab 7 completed successfully.")