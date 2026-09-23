import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import data ingestion function
from src.data.ingest import load_and_validate_data


def train_linear_regression_ls():

    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    print("Loading dataset...")

    df = load_and_validate_data(DATA_PATH)

    print(f"Dataset shape: {df.shape}")


    # ============================================================
    # 2. REMOVE INVALID / MISSING VALUES
    # ============================================================

    df_clean = df.dropna(
        subset=[
            "cgpa",
            "communication_skill_score",
            "salary_package_lpa"
        ]
    ).copy()

    print(
        f"Clean dataset shape: {df_clean.shape}"
    )


    # ============================================================
    # 3. DEFINE INPUT FEATURES AND TARGET
    # ============================================================

    # Input features
    feature_cols = [
        "cgpa",
        "communication_skill_score"
    ]

    # Output / target
    target_col = "salary_package_lpa"


    # Extract X and y

    X_raw = df_clean[feature_cols].values

    y = df_clean[target_col].values.reshape(-1, 1)

    # Number of data points

    N = X_raw.shape[0]

    print("\n============================================================")
    print("LINEAR REGRESSION")
    print("============================================================")

    print(
        f"Loaded {N} data points"
    )

    print(
        f"Input dimension L = {X_raw.shape[1]}"
    )

    print(
        f"Output dimension M = {y.shape[1]}"
    )


    # ============================================================
    # 4. CREATE DESIGN MATRIX
    # ============================================================

    # Add a column of 1s for the intercept/bias

    X_design = np.hstack(
        [
            np.ones((N, 1)),
            X_raw
        ]
    )

    print("\nDesign Matrix Shape:")
    print(X_design.shape)


    # ============================================================
    # 5. STANDARD LEAST SQUARES
    # ============================================================

    # Normal Equation:
    #
    # w = (X^T X)^-1 X^T y

    print("\nCalculating Normal Equation...")

    # Calculate X^T X

    XT_X = np.dot(
        X_design.T,
        X_design
    )


    # Calculate inverse of X^T X

    try:

        XT_X_inv = np.linalg.inv(
            XT_X
        )

        print(
            "Matrix inverse calculated successfully."
        )

    except np.linalg.LinAlgError:

        print(
            "Matrix is singular."
        )

        print(
            "Using pseudo-inverse instead."
        )

        XT_X_inv = np.linalg.pinv(
            XT_X
        )


    # Calculate X^T y

    XT_y = np.dot(
        X_design.T,
        y
    )


    # Calculate optimal weights

    w_optimal = np.dot(
        XT_X_inv,
        XT_y
    )


    # ============================================================
    # 6. DISPLAY OPTIMAL PARAMETERS
    # ============================================================

    print("\n============================================================")
    print("OPTIMAL MODEL PARAMETERS")
    print("============================================================")

    print(
        f"Intercept (w0): "
        f"{w_optimal[0, 0]:.4f}"
    )

    print(
        f"Coefficient for CGPA (w1): "
        f"{w_optimal[1, 0]:.4f}"
    )

    print(
        f"Coefficient for Communication Skill Score (w2): "
        f"{w_optimal[2, 0]:.4f}"
    )


    # ============================================================
    # 7. REGRESSION EQUATION
    # ============================================================

    w0 = w_optimal[0, 0]

    w1 = w_optimal[1, 0]

    w2 = w_optimal[2, 0]


    print("\n============================================================")
    print("REGRESSION EQUATION")
    print("============================================================")

    print(
        f"Salary Package = "
        f"{w0:.4f} "
        f"+ ({w1:.4f} × CGPA) "
        f"+ ({w2:.4f} × Communication Skill Score)"
    )


    # ============================================================
    # 8. MAKE PREDICTIONS
    # ============================================================

    y_pred = np.dot(
        X_design,
        w_optimal
    )


    # ============================================================
    # 9. CALCULATE SUM OF SQUARED ERROR
    # ============================================================

    E_w = 0.5 * np.sum(
        (y_pred - y) ** 2
    )


    print("\n============================================================")
    print("ERROR")
    print("============================================================")

    print(
        f"Minimized Error (E_w): "
        f"{E_w:.4f}"
    )


    # ============================================================
    # 10. CALCULATE MSE AND RMSE
    # ============================================================

    mse = np.mean(
        (y_pred - y) ** 2
    )

    rmse = np.sqrt(
        mse
    )


    print(
        f"Mean Squared Error (MSE): "
        f"{mse:.4f}"
    )

    print(
        f"Root Mean Squared Error (RMSE): "
        f"{rmse:.4f}"
    )


    # ============================================================
    # 11. CREATE OUTPUT DIRECTORY
    # ============================================================

    output_directory = "reports/figures"

    os.makedirs(
        output_directory,
        exist_ok=True
    )


    # ============================================================
    # 12. CREATE 3D VISUALIZATION
    # ============================================================

    print("\nCreating 3D regression plane...")

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )


    # ------------------------------------------------------------
    # Actual data points
    # ------------------------------------------------------------

    ax.scatter(
        X_raw[:, 0],
        X_raw[:, 1],
        y.ravel(),
        alpha=0.6,
        label="Actual Data Points"
    )


    # ============================================================
    # 13. CREATE MESH GRID
    # ============================================================

    x1_surf = np.linspace(
        X_raw[:, 0].min(),
        X_raw[:, 0].max(),
        20
    )

    x2_surf = np.linspace(
        X_raw[:, 1].min(),
        X_raw[:, 1].max(),
        20
    )


    x1_mesh, x2_mesh = np.meshgrid(
        x1_surf,
        x2_surf
    )


    # ============================================================
    # 14. CALCULATE REGRESSION PLANE
    # ============================================================

    # Equation:
    #
    # y = w0 + w1*x1 + w2*x2

    y_mesh = (
        w0
        + w1 * x1_mesh
        + w2 * x2_mesh
    )


    # ============================================================
    # 15. PLOT REGRESSION PLANE
    # ============================================================

    ax.plot_surface(
        x1_mesh,
        x2_mesh,
        y_mesh,
        alpha=0.3,
        edgecolor="none"
    )


    # ============================================================
    # 16. LABEL AXES
    # ============================================================

    ax.set_xlabel(
        "CGPA (Feature 1)"
    )

    ax.set_ylabel(
        "Communication Skill Score (Feature 2)"
    )

    ax.set_zlabel(
        "Salary Package LPA (Target)"
    )

    ax.set_title(
        "Linear Regression via Standard Least Squares"
    )


    # ============================================================
    # 17. SAVE GRAPH
    # ============================================================

    plt.tight_layout()

    output_path = (
        "reports/figures/"
        "linear_regression_3d_plane.png"
    )

    plt.savefig(
        output_path,
        dpi=300
    )

    plt.close()


    # ============================================================
    # 18. FINAL MESSAGE
    # ============================================================

    print("\n============================================================")
    print("LAB 4 LINEAR REGRESSION COMPLETE")
    print("============================================================")

    print(
        f"3D regression plot saved to:"
    )

    print(
        output_path
    )


# ================================================================
# MAIN PROGRAM
# ================================================================

if __name__ == "__main__":

    train_linear_regression_ls()