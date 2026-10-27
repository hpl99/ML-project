"""Practical 1: Exploratory Data Analysis & Preprocessing Pipeline.

Generates summary statistics, missing-value diagnostic heatmaps, Pearson correlation matrices,
and constructs the scikit-learn preprocessing ColumnTransformer fitted on TRAIN ONLY (ADR 0005).
"""
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ml.config import (
    REPORTS_DIR, REQUIRED_FEATURES, CATEGORICAL_FEATURES,
    CONTINUOUS_TERRAIN_FEATURES
)

logger = logging.getLogger(__name__)


def generate_eda_report(df: pd.DataFrame) -> Tuple[Dict[str, Any], Path]:
    """Compute summary stats and generate side-by-side missingness heatmap & correlation matrix."""
    logger.info("Computing Practical 1 EDA statistics...")

    num_features = [f for f in REQUIRED_FEATURES if f in df.columns]
    cat_features = [f for f in CATEGORICAL_FEATURES if f in df.columns]

    stats = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "missing_counts": df[num_features].isna().sum().to_dict(),
        "summary": df[num_features].describe().round(3).to_dict()
    }

    # Generate Side-by-Side Figure: Missingness Heatmap + Correlation Matrix
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor("#ffffff")

    # 1. Missingness Heatmap
    missing_mat = df[num_features].isna().values.T.astype(int)
    im1 = ax1.imshow(missing_mat, aspect="auto", cmap="Blues", interpolation="nearest")
    ax1.set_yticks(range(len(num_features)))
    ax1.set_yticklabels(num_features, fontsize=9)
    ax1.set_xlabel("Sample Index", fontsize=10, fontweight="bold")
    ax1.set_title("Missing Value Diagnostic Pattern (Practical 1)", fontsize=12, fontweight="bold", pad=12)
    plt.colorbar(im1, ax=ax1, fraction=0.03, pad=0.04, label="Missing Flag (1=Missing)")

    # 2. Pearson Correlation Matrix
    corr_mat = df[num_features].corr(method="pearson").values
    im2 = ax2.imshow(corr_mat, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax2.set_xticks(range(len(num_features)))
    ax2.set_yticks(range(len(num_features)))
    ax2.set_xticklabels(num_features, rotation=90, fontsize=8)
    ax2.set_yticklabels(num_features, fontsize=8)
    ax2.set_title("Feature Correlation Matrix (Pearson r)", fontsize=12, fontweight="bold", pad=12)
    plt.colorbar(im2, ax=ax2, fraction=0.03, pad=0.04, label="Correlation (r)")

    plt.tight_layout()
    fig_path = REPORTS_DIR / "eda_missing_and_corr.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved Practical 1 diagnostic figure to {fig_path}")

    return stats, fig_path


def build_preprocessor(numeric_features: list = None, categorical_features: list = None) -> ColumnTransformer:
    """Construct scikit-learn Pipeline and ColumnTransformer fitted strictly on TRAIN ONLY.

    Applies:
    - Median Imputation for numeric features
    - StandardScaler for continuous features
    - OneHotEncoder with handle_unknown='ignore' for categorical factors
    """
    if numeric_features is None:
        numeric_features = REQUIRED_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Other Region")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )
    return preprocessor


if __name__ == "__main__":
    from ml.build_table import run_build_table
    df_data = run_build_table(limit=200)
    stats, p = generate_eda_report(df_data)
    print("EDA completed successfully.")
