import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import the validated ingestion function from ingest.py
from ingest import load_and_validate_data


def perform_eda():

    # ============================================================
    # 1. LOAD DATASET
    # ============================================================

    DATA_PATH = os.path.join(
        "src",
        "data",
        "raw_placement_data.csv"
    )

    print("\n" + "=" * 60)
    print("LOADING PLACEMENT DATASET")
    print("=" * 60)

    try:
        df = load_and_validate_data(DATA_PATH)
    except Exception as e:
        print("ERROR: Unable to load the dataset.")
        print("Reason:", e)
        return

    print("Dataset loaded successfully!")


    # ============================================================
    # 2. DATASET DIMENSIONS
    # ============================================================

    print("\n" + "=" * 60)
    print("--- 1. DATASET DIMENSIONS ---")
    print("=" * 60)

    print("Total Rows (Samples):", df.shape[0])
    print("Total Columns (Features):", df.shape[1])


    # ============================================================
    # 3. FEATURE NAMES AND DATA TYPES
    # ============================================================

    print("\n" + "=" * 60)
    print("--- 2. FEATURE NAMES & DATA TYPES ---")
    print("=" * 60)

    print(df.dtypes)


    # ============================================================
    # 4. MISSING VALUES
    # ============================================================

    print("\n" + "=" * 60)
    print("--- 3. MISSING VALUES & DUPLICATES ---")
    print("=" * 60)

    missing_vals = df.isnull().sum()

    if missing_vals.sum() > 0:
        print("\nMissing Values per Column:")
        print(missing_vals[missing_vals > 0])
    else:
        print("No missing values found.")


    # ============================================================
    # 5. DUPLICATE RECORDS
    # ============================================================

    duplicates = df.duplicated().sum()

    print("\nDuplicate Records Count:", duplicates)


    # ============================================================
    # 6. SUMMARY STATISTICS
    # ============================================================

    print("\n" + "=" * 60)
    print("--- 4. SUMMARY STATISTICS (NUMERICAL FEATURES) ---")
    print("=" * 60)

    print(df.describe())


    # ============================================================
    # 7. CLASS IMBALANCE ANALYSIS
    # ============================================================

    print("\n" + "=" * 60)
    print("--- 5. CLASS IMBALANCE ANALYSIS ---")
    print("=" * 60)

    if "placement_status" in df.columns:

        class_counts = df["placement_status"].value_counts()

        class_percentages = (
            df["placement_status"]
            .value_counts(normalize=True)
            * 100
        )

        print("\nPlacement Status Counts:")
        print(class_counts)

        print("\nPlacement Status Percentages:")
        print(class_percentages)

    else:
        print(
            "Target column 'placement_status' "
            "not found for class imbalance analysis."
        )


    # ============================================================
    # 8. CREATE FIGURES DIRECTORY
    # ============================================================

    print("\n" + "=" * 60)
    print("--- 6. GENERATING VISUALIZATIONS ---")
    print("=" * 60)

    sns.set_theme(style="whitegrid")

    os.makedirs("reports/figures", exist_ok=True)


    # ============================================================
    # 9. CORRELATION MATRIX HEATMAP
    # ============================================================

    try:

        numerical_df = df.select_dtypes(include=[np.number])

        if numerical_df.shape[1] >= 2:

            plt.figure(figsize=(10, 8))

            corr_matrix = numerical_df.corr()

            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                linewidths=0.5
            )

            plt.title("Feature Correlation Matrix Heatmap")

            plt.tight_layout()

            plt.savefig(
                "reports/figures/correlation_heatmap.png"
            )

            plt.close()

            print(
                "Saved correlation heatmap to "
                "reports/figures/correlation_heatmap.png"
            )

        else:
            print(
                "Not enough numerical columns "
                "to generate correlation heatmap."
            )

    except Exception as e:

        print("Error generating correlation heatmap:", e)


    # ============================================================
    # 10. SCATTER PLOT: CGPA VS SALARY
    # ============================================================

    try:

        if (
            "cgpa" in df.columns
            and "salary_package_lpa" in df.columns
        ):

            plt.figure(figsize=(8, 6))

            if "placement_status" in df.columns:

                sns.scatterplot(
                    data=df,
                    x="cgpa",
                    y="salary_package_lpa",
                    hue="placement_status",
                    alpha=0.7
                )

            else:

                sns.scatterplot(
                    data=df,
                    x="cgpa",
                    y="salary_package_lpa",
                    alpha=0.7
                )

            plt.title(
                "CGPA vs Salary Package "
                "(Colored by Placement Status)"
            )

            plt.tight_layout()

            plt.savefig(
                "reports/figures/scatter_cgpa_salary.png"
            )

            plt.close()

            print(
                "Saved scatter plot to "
                "reports/figures/scatter_cgpa_salary.png"
            )

        else:

            print(
                "CGPA or salary_package_lpa column "
                "not found. Scatter plot skipped."
            )

    except Exception as e:

        print("Error generating scatter plot:", e)


    # ============================================================
    # 11. PAIR PLOT
    # ============================================================

    try:

        pairplot_cols = [
            "cgpa",
            "backlogs",
            "communication_skills",
            "internships",
            "salary_package_lpa"
        ]

        valid_pair_cols = [
            col
            for col in pairplot_cols
            if col in df.columns
        ]

        if len(valid_pair_cols) > 1:

            pp = sns.pairplot(
                df[valid_pair_cols],
                diag_kind="kde",
                corner=True
            )

            pp.fig.suptitle(
                "Pairwise Relationships of Key Numerical Features",
                y=1.02
            )

            pp.savefig(
                "reports/figures/pairplot_features.png"
            )

            plt.close()

            print(
                "Saved pair plot to "
                "reports/figures/pairplot_features.png"
            )

        else:

            print(
                "Not enough valid columns "
                "for pair plot."
            )

    except Exception as e:

        print(
            "Skipping pair plot due to error:",
            e
        )


    # ============================================================
    # 12. OUTLIER DETECTION USING BOXPLOT
    # ============================================================

    try:

        if numerical_df.shape[1] > 0:

            plt.figure(figsize=(12, 6))

            sns.boxplot(
                data=numerical_df,
                orient="h"
            )

            plt.title(
                "Outlier Identification via Boxplots "
                "(Numerical Features)"
            )

            plt.tight_layout()

            plt.savefig(
                "reports/figures/outliers_boxplot.png"
            )

            plt.close()

            print(
                "Saved outlier boxplot to "
                "reports/figures/outliers_boxplot.png"
            )

        else:

            print(
                "No numerical columns available "
                "for outlier detection."
            )

    except Exception as e:

        print(
            "Error generating outlier boxplot:",
            e
        )


    # ============================================================
    # 13. COMPLETION MESSAGE
    # ============================================================

    print("\n" + "=" * 60)
    print("EDA EXECUTION COMPLETE")
    print("=" * 60)

    print(
        "All generated visualizations are stored in:"
    )

    print("reports/figures/")


# ================================================================
# MAIN PROGRAM
# ================================================================

if __name__ == "__main__":
    perform_eda()