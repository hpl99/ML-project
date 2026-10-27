# ADR 0007: Model Architecture Selection, Clustering, and Tuning

- **Status**: Accepted
- **Date**: 2026-10-27
- **Author**: URsafe Autonomous Engineering Team

## Context
Practicals 1 to 7 require systematic comparison across model families: linear baselines (Logistic Regression), single tree baselines (Decision Tree), ensembling (Bagging, Random Forest, and HistGradientBoosting), terrain unsupervised clustering (K-Means and Gaussian Mixture Models), and hyperparameter tuning via cross-validation.

## Decision
1. **Unsupervised Terrain Stratification (Practicals 4 & 5)**:
   - Cluster static morphometric variables (`elevation_m`, `slope_deg`, `roughness`, `relief_m`) using K-Means.
   - Select $K \in [3, 8]$ via the Elbow Method (inertia inflection) and Silhouette score analysis.
   - Fit Gaussian Mixture Models ($K$ components) with `covariance_type='diag'` (compared to `full` via BIC) to output soft probabilistic membership across terrain zones.
   - Name zones based on empirical landslide concentration: `Low Susceptibility Basin`, `Moderate Slopes`, `High Hazard Mountain Terrain`.
   - Perform PCA dimensionality reduction to 2D for interactive visualization.
2. **Supervised Classification Suite (Practicals 2, 3, 6 & 7)**:
   - Logistic Regression with standard scaling and L2 regularization.
   - Decision Tree (baseline CART).
   - Bagging Classifier over decision trees.
   - Random Forest Classifier (GridSearchCV tuning over `n_estimators`, `max_depth`, `min_samples_split`).
   - Histogram-based Gradient Boosting Classifier (`HistGradientBoostingClassifier`).
3. **Primary Evaluation Metrics**:
   - Area Under ROC Curve (ROC-AUC), Area Under Precision-Recall Curve (PR-AUC), Recall at target operational threshold, F1-Score, and Brier Score.
4. **Final Model Selection**:
   - The tuned ensemble achieving the highest spatial-split PR-AUC and ROC-AUC is selected as the primary production model and saved to `models/final_model.joblib`.

## Consequences
- Unsupervised clusters provide domain context to the user, while tuned ensembles ensure robust non-linear hazard prediction.
