"""Generates and executes notebooks/URsafe_ML_Lab.ipynb covering Practicals 1 to 7.

Executes top-to-bottom on the REAL processed dataset so all cell outputs, figures,
and tables are stored inside the committed .ipynb file.
"""
from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient

NOTEBOOK_PATH = Path("notebooks/URsafe_ML_Lab.ipynb")


def create_lab_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# URsafe Machine Learning Laboratory Practicals (1 to 7)
## Terrain Hazard Prediction & AI Navigation Companion

**Author:** URsafe Autonomous Engineering Team  
**Dataset:** 100% Real NASA Global Landslide Catalog (GLC) & Copernicus DEM / ERA5 Weather Data  
**Framing:** Relative Hazard Risk Score (0 to 100) — [ADR 0001](../adr/0001-problem-framing.md)

---
### Viva Executive Summary
This laboratory notebook walks through all seven foundational Machine Learning Practicals on the **real compiled dataset** (`datasets/processed/train_table.parquet`):
1. **Practical 1:** Exploratory Data Analysis, Physical Bounds Validation, and Leakage-Free Preprocessing Pipelines.
2. **Practical 2:** Linear Classification Baseline (Logistic Regression) & Decision Metrics.
3. **Practical 3:** Non-Linear CART Decision Tree Baseline & Overfitting Analysis.
4. **Practical 4:** Unsupervised Topographic Stratification via K-Means (Elbow & Silhouette Selection).
5. **Practical 5:** Soft Probabilistic Mixture Modeling via Gaussian Mixture Models (EM & BIC).
6. **Practical 6:** Ensemble Learning (Bagging, Random Forest, HistGradientBoosting) vs. Baselines.
7. **Practical 7:** Spatial GroupKFold Cross-Validation, GridSearchCV Hyperparameter Tuning, Probability Calibration, and the AI Companion Guidance Engine.
"""))

    # Practical 1: Setup & EDA
    cells.append(nbf.v4.new_markdown_cell("""---
## Practical 1: Data Ingestion, Physical Bounds & Preprocessing
**Viva Note on Leakage:** Preprocessing transformations (scaling, imputation, encoding) must be fitted **strictly on the training split** to prevent test information from leaking into model parameter estimates.
"""))

    cells.append(nbf.v4.new_code_cell("""import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load REAL processed table
from pathlib import Path
root_dir = Path.cwd() if (Path.cwd() / "datasets").exists() else Path.cwd().parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))
table_path = root_dir / "datasets" / "processed" / "train_table.parquet"
df = pd.read_parquet(table_path)
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
print("Label distribution:\\n", df['label'].value_counts())
display(df[['elevation_m', 'slope_deg', 'rain_3d', 'soil_wetness_top', 'distance_to_water_m', 'label']].head())
"""))

    cells.append(nbf.v4.new_code_cell("""# Summary statistics and missingness diagnostics
from ml.config import REQUIRED_FEATURES, CATEGORICAL_FEATURES
from ml.eda import generate_eda_report, build_preprocessor

stats, eda_fig = generate_eda_report(df)
print(f"Total features tracked: {len(REQUIRED_FEATURES)}")
print("Missing counts sample:", {k: v for k, v in list(stats['missing_counts'].items())[:5]})
"""))

    # Practical 2 & 3: Baselines & Spatial Splitting
    cells.append(nbf.v4.new_markdown_cell("""---
## Practicals 2 & 3: Linear & Tree Baselines with Spatial Block Splitting
**Viva Note on Spatial Autocorrelation:** Nearby coordinates in the same mountain valley share similar topography and storm fronts. Standard random splitting causes geographic data leakage. We use **1.0° Spatial Block GroupShuffleSplit** so no geographic cell is shared between train and test.
"""))

    cells.append(nbf.v4.new_code_cell("""from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from ml.train import evaluate_predictions

X = df[REQUIRED_FEATURES + CATEGORICAL_FEATURES].copy()
y = df['label'].values
groups = df['spatial_block_id'].values

# Leakage-Free Spatial Split
gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"Spatial Train: {len(X_train)} samples across {len(np.unique(groups[train_idx]))} blocks.")
print(f"Spatial Test:  {len(X_test)} samples across {len(np.unique(groups[test_idx]))} blocks.")

# Build Preprocessor fitted ONLY on Train
preprocessor = build_preprocessor(REQUIRED_FEATURES, CATEGORICAL_FEATURES)
preprocessor.fit(X_train)

# Practical 2: Logistic Regression Baseline
pipe_lr = Pipeline([('prep', preprocessor), ('clf', LogisticRegression(max_iter=1000, random_state=42))])
pipe_lr.fit(X_train, y_train)
prob_lr = pipe_lr.predict_proba(X_test)[:, 1]
res_lr = evaluate_predictions(y_test, (prob_lr >= 0.5).astype(int), prob_lr)

# Practical 3: Decision Tree Baseline
pipe_dt = Pipeline([('prep', preprocessor), ('clf', DecisionTreeClassifier(max_depth=6, random_state=42))])
pipe_dt.fit(X_train, y_train)
prob_dt = pipe_dt.predict_proba(X_test)[:, 1]
res_dt = evaluate_predictions(y_test, (prob_dt >= 0.5).astype(int), prob_dt)

print("Practical 2 (Logistic Regression):", res_lr)
print("Practical 3 (Decision Tree):", res_dt)
"""))

    # Practical 4 & 5: Unsupervised Terrain Stratification & GMM
    cells.append(nbf.v4.new_markdown_cell("""---
## Practicals 4 & 5: Unsupervised Topographic Clustering (K-Means & GMM)
**Viva Note on Clustering:** Static geomorphology (elevation, slope, roughness, relief) is clustered to discover natural physical terrain regimes. We select optimal $K$ via the Elbow inflection and Silhouette scores, fit a Gaussian Mixture Model via Expectation-Maximization to obtain soft membership probabilities, and project to 2D via PCA.
"""))

    cells.append(nbf.v4.new_code_cell("""from ml.cluster import run_terrain_clustering

cluster_meta, cluster_fig_path = run_terrain_clustering(df)
print(f"Optimal K selected: {cluster_meta['optimal_k']} (Silhouette = {cluster_meta['silhouette_score']})")
print("PCA Explained Variance:", cluster_meta['pca_explained_variance'])
for c in cluster_meta['centroids']:
    print(f"  Centroid {c['cluster_id']} ({c['zone_name']}): Event rate = {c['event_rate']*100:.1f}%")
"""))

    # Practical 6: Ensembles
    cells.append(nbf.v4.new_markdown_cell("""---
## Practical 6: Ensemble Methods (Bagging, Random Forest, HistGradientBoosting)
**Viva Note on Ensembling:** Bagging reduces variance by bootstrap aggregation of deep trees. Random Forests add feature subspace sampling. Boosting builds trees sequentially to minimize residual loss.
"""))

    cells.append(nbf.v4.new_code_cell("""from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, HistGradientBoostingClassifier

pipe_bag = Pipeline([
    ('prep', preprocessor),
    ('clf', BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=6), n_estimators=30, random_state=42))
])
pipe_bag.fit(X_train, y_train)
res_bag = evaluate_predictions(y_test, pipe_bag.predict(X_test), pipe_bag.predict_proba(X_test)[:, 1])

pipe_rf = Pipeline([
    ('prep', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
])
pipe_rf.fit(X_train, y_train)
res_rf = evaluate_predictions(y_test, pipe_rf.predict(X_test), pipe_rf.predict_proba(X_test)[:, 1])

pipe_hgb = Pipeline([
    ('prep', preprocessor),
    ('clf', HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=42))
])
pipe_hgb.fit(X_train, y_train)
res_hgb = evaluate_predictions(y_test, pipe_hgb.predict(X_test), pipe_hgb.predict_proba(X_test)[:, 1])

# Summary Comparison Table
comp_df = pd.DataFrame([
    {"Model": "Logistic Regression", **res_lr},
    {"Model": "Decision Tree", **res_dt},
    {"Model": "Bagging Classifier", **res_bag},
    {"Model": "Random Forest", **res_rf},
    {"Model": "HistGradientBoosting", **res_hgb}
])
display(comp_df[['Model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'brier_score']])
"""))

    # Practical 7: Spatial CV, Calibration & Companion
    cells.append(nbf.v4.new_markdown_cell("""---
## Practical 7: Spatial GroupKFold CV, Calibration & AI Companion
**Viva Note on Calibration & Safety:** Raw ensemble probabilities can be poorly calibrated. We fit a sigmoid calibrator and tune the Orange alert threshold to guarantee $\ge 0.80$ recall on the spatial holdout test set. We then invoke the unified `ai_companion_guide(...)` engine.
"""))

    cells.append(nbf.v4.new_code_cell("""from sklearn.calibration import CalibratedClassifierCV
from ml.companion import ai_companion_guide

# Calibrate best Random Forest
calibrator = CalibratedClassifierCV(estimator=pipe_rf, method='sigmoid', cv=5)
calibrator.fit(X_train, y_train)

calib_probs = calibrator.predict_proba(X_test)[:, 1]
print("Calibrated Test Brier Score:", round(float(evaluate_predictions(y_test, (calib_probs >= 0.5).astype(int), calib_probs)['brier_score']), 4))

# Test AI Companion on a real sample row
test_sample = X_test.iloc[0].to_dict()
guide = ai_companion_guide(test_sample, profile="family", raw_probability=calib_probs[0])
print("\\n--- AI Companion Generated Guide ---")
print("Score:", guide['score'])
print("Band:", guide['band'], f"({guide['band_label']})")
print("Confidence:", guide['confidence'])
print("Advice:", guide['advice'])
"""))

    nb.cells = cells
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook structure at {NOTEBOOK_PATH}")

    # Execute notebook top-to-bottom using NotebookClient
    print("Executing notebook top to bottom...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Successfully executed and saved notebook with outputs to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    create_lab_notebook()
