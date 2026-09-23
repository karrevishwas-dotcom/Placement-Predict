import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

# Import validated ingestion function
from src.data.ingest import load_and_validate_data


def compute_cost(X, y, w):
    """
    Computes Mean Squared Error (MSE) cost function
    divided by 2.
    """
    m = len(y)

    predictions = np.dot(X, w)
    errors = predictions - y

    cost = (1 / (2 * m)) * np.sum(errors ** 2)

    return cost


def gradient_descent(X, y, w, alpha, num_iters):
    """
    Implements Gradient Descent optimization from scratch.
    """

    m = len(y)
    cost_history = []

    for i in range(num_iters):

        # Calculate predictions
        predictions = np.dot(X, w)

        # Calculate errors
        errors = predictions - y

        # Calculate gradient
        gradient = (1 / m) * np.dot(X.T, errors)

        # Update weights
        w = w - alpha * gradient

        # Calculate and store cost
        cost = compute_cost(X, y, w)
        cost_history.append(cost)

    return w, cost_history


def run_gradient_descent_experiment():

    print("\n" + "=" * 60)
    print("LAB 5 - LINEAR REGRESSION USING GRADIENT DESCENT")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    print("\nLoading dataset...")
    print("Data path:", DATA_PATH)

    df = load_and_validate_data(DATA_PATH)

    print("Dataset loaded successfully.")
    print("Number of rows:", len(df))

    # ---------------------------------------------------------
    # 2. SELECT FEATURE AND TARGET
    # ---------------------------------------------------------

    feature_cols = ["cgpa"]
    target_col = "salary_package_lpa"

    print("\nFeature:", feature_cols)
    print("Target:", target_col)

    # Remove missing values
    df_clean = df.dropna(
        subset=feature_cols + [target_col]
    ).copy()

    print("Rows after removing missing values:", len(df_clean))

    # Convert to NumPy arrays
    X_raw = df_clean[feature_cols].values

    y_raw = df_clean[target_col].values.reshape(-1, 1)

    # ---------------------------------------------------------
    # 3. TRAIN-TEST SPLIT
    # ---------------------------------------------------------

    print("\nSplitting dataset into 80% training and 20% testing...")

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=0.20,
        random_state=42
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ---------------------------------------------------------
    # 4. FEATURE SCALING
    # ---------------------------------------------------------

    print("\nScaling features and target...")

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_x.fit_transform(X_train)

    X_test_scaled = scaler_x.transform(X_test)

    y_train_scaled = scaler_y.fit_transform(y_train)

    print("Scaling completed.")

    # ---------------------------------------------------------
    # 5. ADD INTERCEPT / BIAS COLUMN
    # ---------------------------------------------------------

    X_train_design = np.hstack([
        np.ones((X_train_scaled.shape[0], 1)),
        X_train_scaled
    ])

    print("\nDesign matrix created.")
    print("Design matrix shape:", X_train_design.shape)

    # ---------------------------------------------------------
    # 6. LEARNING RATE EXPERIMENT
    # ---------------------------------------------------------

    learning_rates = [0.001, 0.01, 0.1, 0.5]

    num_iterations = 1000

    print("\nLearning rates:", learning_rates)
    print("Iterations:", num_iterations)

    # Create output directory
    os.makedirs("reports/figures", exist_ok=True)

    plt.figure(figsize=(10, 6))

    results = {}

    # ---------------------------------------------------------
    # 7. RUN GRADIENT DESCENT
    # ---------------------------------------------------------

    for alpha in learning_rates:

        print(f"\nRunning Gradient Descent with alpha = {alpha}")

        # Initialize weights with zeros
        w_init = np.zeros(
            (X_train_design.shape[1], 1)
        )

        # Run Gradient Descent
        w_opt, cost_history = gradient_descent(
            X_train_design,
            y_train_scaled,
            w_init,
            alpha,
            num_iterations
        )

        # Store results
        results[alpha] = {
            "weights": w_opt,
            "history": cost_history
        }

        # Print final cost
        print(
            f"Final cost for alpha {alpha}: "
            f"{cost_history[-1]:.6f}"
        )

        # Plot cost history
        plt.plot(
            cost_history,
            label=f"Alpha (α) = {alpha}"
        )

    # ---------------------------------------------------------
    # 8. SAVE COST GRAPH
    # ---------------------------------------------------------

    plt.xlabel("Iterations")
    plt.ylabel("Cost Function E(w) - MSE")
    plt.title(
        "Effect of Different Learning Rates "
        "on Gradient Descent Convergence"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    graph_path = (
        "reports/figures/"
        "gd_learning_rates_comparison.png"
    )

    plt.savefig(graph_path)
    plt.close()

    print("\nCost graph saved to:")
    print(graph_path)

    # ---------------------------------------------------------
    # 9. SELECT BEST LEARNING RATE
    # ---------------------------------------------------------

    best_alpha = 0.1

    final_w = results[best_alpha]["weights"]

    print("\n" + "=" * 60)
    print(
        f"CUSTOM GRADIENT DESCENT PARAMETERS "
        f"(alpha = {best_alpha})"
    )
    print("=" * 60)

    print(
        f"Intercept (w0): "
        f"{final_w[0, 0]:.4f}"
    )

    print(
        f"Coefficient (w1): "
        f"{final_w[1, 0]:.4f}"
    )

    # ---------------------------------------------------------
    # 10. SCIKIT-LEARN COMPARISON
    # ---------------------------------------------------------

    print("\nTraining Scikit-learn Linear Regression...")

    sklearn_model = SklearnLinearRegression()

    sklearn_model.fit(
        X_train_scaled,
        y_train_scaled
    )

    print("\n" + "=" * 60)
    print("SCIKIT-LEARN COMPARISON")
    print("=" * 60)

    print(
        f"Scikit-learn Intercept: "
        f"{sklearn_model.intercept_[0]:.4f}"
    )

    print(
        f"Scikit-learn Coefficient: "
        f"{sklearn_model.coef_[0, 0]:.4f}"
    )

    # ---------------------------------------------------------
    # 11. FINAL MESSAGE
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("LAB 5 LINEAR REGRESSION COMPLETE")
    print("=" * 60)

    print("\nGenerated file:")
    print(
        "reports/figures/"
        "gd_learning_rates_comparison.png"
    )


# -------------------------------------------------------------
# MAIN PROGRAM
# -------------------------------------------------------------

if __name__ == "__main__":
    run_gradient_descent_experiment()