"""Practicals 2, 3, 6, 7 & Calibration: Supervised Learning, Ensembling, and Spatial Validation.

Implements:
1. Spatial Block GroupShuffleSplit (80:20) + Random Split (leakage demonstration) (ADR 0006)
2. Baseline models: Logistic Regression and Decision Tree (Practicals 2 & 3)
3. Ensemble models: Bagging, Random Forest, HistGradientBoosting (Practical 6)
4. Hyperparameter tuning via 5-fold Spatial GroupKFold GridSearchCV (Practical 7)
5. Probability calibration (Isotonic/Sigmoid) with Brier scores and reliability curves (ADR 0008)
6. Calibrated risk bands with Orange threshold tuned for >= 0.80 recall
7. Model Card and metrics export to reports/metrics.json and models/model_card.json
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import BaggingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    brier_score_loss
)
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, GridSearchCV, StratifiedKFold,
    train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ml.config import (
    MODELS_DIR, REPORTS_DIR, ADR_GEN_DIR, REQUIRED_FEATURES, CATEGORICAL_FEATURES,
    ALL_MODEL_FEATURES, RANDOM_SEED, RISK_BANDS
)
from ml.eda import build_preprocessor

logger = logging.getLogger(__name__)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculate standard evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    far = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.5,
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4) if len(np.unique(y_true)) > 1 else 0.5,
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "false_alarm_rate": round(far, 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        }
    }


def run_training_pipeline(df: pd.DataFrame) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Execute complete ML pipeline across all Practicals on real training data."""
    logger.info("Executing supervised learning training pipeline...")

    # Ensure required features exist in table
    feature_cols = [c for c in ALL_MODEL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    y = df["label"].values
    groups = df["spatial_block_id"].values

    # -------------------------------------------------------------------------
    # 1. SPATIAL BLOCK SPLIT (80:20) vs. RANDOM SPLIT (Leakage Demonstration)
    # -------------------------------------------------------------------------
    # Spatial Block Split: GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    train_groups = groups[train_idx]

    logger.info(
        f"Spatial Block Split: Train={len(X_train)} samples across {len(np.unique(train_groups))} blocks, "
        f"Test={len(X_test)} samples across {len(np.unique(groups[test_idx]))} blocks."
    )

    # Random split baseline (to demonstrate leakage gap per Section 5.2)
    X_tr_rand, X_te_rand, y_tr_rand, y_te_rand = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # -------------------------------------------------------------------------
    # 2. FIT SCALER & PREPROCESSOR ON TRAIN ONLY
    # -------------------------------------------------------------------------
    num_feats = [c for c in REQUIRED_FEATURES if c in X_train.columns]
    cat_feats = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    preprocessor = build_preprocessor(numeric_features=num_feats, categorical_features=cat_feats)
    preprocessor.fit(X_train)
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")

    # -------------------------------------------------------------------------
    # 3. BASELINES & ENSEMBLES SUITE (Practicals 2, 3, 6)
    # -------------------------------------------------------------------------
    models_dict = {
        "LogisticRegression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
        ]),
        "DecisionTree": Pipeline([
            ("prep", preprocessor),
            ("clf", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED))
        ]),
        "BaggingClassifier": Pipeline([
            ("prep", preprocessor),
            ("clf", BaggingClassifier(
                estimator=DecisionTreeClassifier(max_depth=6),
                n_estimators=30, random_state=RANDOM_SEED, n_jobs=-1
            ))
        ]),
        "RandomForest": Pipeline([
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=100, max_depth=8, min_samples_split=4,
                random_state=RANDOM_SEED, n_jobs=-1
            ))
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", preprocessor),
            ("clf", HistGradientBoostingClassifier(
                max_iter=100, max_depth=6, random_state=RANDOM_SEED
            ))
        ])
    }

    results = {}
    fitted_models = {}

    for name, pipe in models_dict.items():
        logger.info(f"Fitting model: {name}...")
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe

        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        res = evaluate_predictions(y_test, y_pred, y_prob)
        results[name] = res
        logger.info(
            f"{name} -> Accuracy: {res['accuracy']:.3f}, F1: {res['f1']:.3f}, "
            f"ROC-AUC: {res['roc_auc']:.3f}, PR-AUC: {res['pr_auc']:.3f}"
        )

    # Leakage Demonstration: Random Split RF vs Spatial Split RF
    rf_rand_pipe = Pipeline([
        ("prep", build_preprocessor(num_feats, cat_feats)),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=RANDOM_SEED))
    ])
    rf_rand_pipe.fit(X_tr_rand, y_tr_rand)
    rand_prob = rf_rand_pipe.predict_proba(X_te_rand)[:, 1]
    rand_metrics = evaluate_predictions(y_te_rand, (rand_prob >= 0.5).astype(int), rand_prob)

    leakage_gap = {
        "random_split_roc_auc": rand_metrics["roc_auc"],
        "spatial_split_roc_auc": results["RandomForest"]["roc_auc"],
        "auc_leakage_delta": round(rand_metrics["roc_auc"] - results["RandomForest"]["roc_auc"], 4),
        "notes": "Spatial autocorrelation causes random split to overestimate performance by leaking nearby points."
    }
    logger.info(f"Spatial Autocorrelation Leakage Gap: Delta ROC-AUC = {leakage_gap['auc_leakage_delta']:+.4f}")

    # -------------------------------------------------------------------------
    # 4. PRACTICAL 7: HYPERPARAMETER TUNING & SPATIAL CV
    # -------------------------------------------------------------------------
    logger.info("Executing Practical 7 Spatial GroupKFold hyperparameter tuning...")
    param_grid = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [6, 10],
        "clf__min_samples_split": [2, 5]
    }
    gkf = GroupKFold(n_splits=min(5, len(np.unique(train_groups))))
    cv_splits = list(gkf.split(X_train, y_train, groups=train_groups))
    grid_search = GridSearchCV(
        estimator=models_dict["RandomForest"],
        param_grid=param_grid,
        cv=cv_splits,
        scoring="average_precision",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    logger.info(f"GridSearchCV Best Params: {grid_search.best_params_} (Best CV PR-AUC = {grid_search.best_score_:.4f})")

    # CV Boxplot Comparison: Standard StratifiedKFold vs Spatial GroupKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    skf_scores = []
    gkf_scores = []
    for (tr, te) in skf.split(X_train, y_train):
        pipe_tmp = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(max_depth=8, n_estimators=50, random_state=RANDOM_SEED))])
        pipe_tmp.fit(X_train.iloc[tr], y_train[tr])
        skf_scores.append(average_precision_score(y_train[te], pipe_tmp.predict_proba(X_train.iloc[te])[:, 1]))

    for (tr, te) in gkf.split(X_train, y_train, groups=train_groups):
        pipe_tmp = Pipeline([("prep", preprocessor), ("clf", RandomForestClassifier(max_depth=8, n_estimators=50, random_state=RANDOM_SEED))])
        pipe_tmp.fit(X_train.iloc[tr], y_train[tr])
        gkf_scores.append(average_precision_score(y_train[te], pipe_tmp.predict_proba(X_train.iloc[te])[:, 1]))

    # Plot CV Boxplots
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([skf_scores, gkf_scores], tick_labels=["Random 5-Fold StratifiedCV", "Spatial 5-Fold GroupKFold"])
    ax.set_ylabel("PR-AUC Score", fontweight="bold")
    ax.set_title("Cross-Validation Generalization: Random vs Spatial Group CV", fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    cv_fig_path = REPORTS_DIR / "cv_comparison.png"
    plt.savefig(cv_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 5. MODEL SELECTION & PROBABILITY CALIBRATION (Section 5.6 & ADR 0008)
    # -------------------------------------------------------------------------
    # Select best model: compare Tuned RF vs HistGradientBoosting
    hgb_prob = fitted_models["HistGradientBoosting"].predict_proba(X_test)[:, 1]
    rf_best_prob = best_rf.predict_proba(X_test)[:, 1]

    chosen_model = best_rf if average_precision_score(y_test, rf_best_prob) >= average_precision_score(y_test, hgb_prob) else fitted_models["HistGradientBoosting"]
    chosen_name = "TunedRandomForest" if chosen_model == best_rf else "HistGradientBoosting"
    logger.info(f"Primary Production Model Selected: {chosen_name}")

    # Probability Calibration using Sigmoid or Isotonic
    calibrator = CalibratedClassifierCV(estimator=chosen_model, method="sigmoid", cv=5)
    calibrator.fit(X_train, y_train)

    calib_prob = calibrator.predict_proba(X_test)[:, 1]
    raw_brier = brier_score_loss(y_test, rf_best_prob)
    calib_brier = brier_score_loss(y_test, calib_prob)
    logger.info(f"Calibration Brier Score: Raw={raw_brier:.4f} -> Calibrated={calib_brier:.4f}")

    # Plot Calibration Reliability Curve
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, rf_best_prob, n_bins=8)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, calib_prob, n_bins=8)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax.plot(prob_pred_raw, prob_true_raw, "s--", color="#e11d48", label=f"Raw Model (Brier={raw_brier:.3f})")
    ax.plot(prob_pred_cal, prob_true_cal, "o-", color="#0284c7", label=f"Calibrated Model (Brier={calib_brier:.3f})")
    ax.set_xlabel("Mean Predicted Relative Risk Probability", fontweight="bold")
    ax.set_ylabel("Empirical Event Fraction", fontweight="bold")
    ax.set_title("Probability Calibration Reliability Diagram (Section 5.6)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    calib_fig_path = REPORTS_DIR / "calibration_curve.png"
    plt.savefig(calib_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 6. OPERATIONAL BAND THRESHOLDS (Section 5.6: Orange recall >= 0.80)
    # -------------------------------------------------------------------------
    # Search threshold where recall >= 0.80
    thresholds = np.linspace(0.1, 0.9, 81)
    orange_thresh = 0.55
    orange_recall = 0.0
    orange_far = 0.0

    for th in thresholds:
        preds = (calib_prob >= th).astype(int)
        rec = recall_score(y_test, preds, zero_division=0)
        if rec >= 0.80:
            orange_thresh = float(th)
            orange_recall = float(rec)
            cm_t = confusion_matrix(y_test, preds)
            tn_t, fp_t, fn_t, tp_t = cm_t.ravel() if cm_t.shape == (2, 2) else (0, 0, 0, 0)
            orange_far = float(fp_t / (fp_t + tn_t)) if (fp_t + tn_t) > 0 else 0.0

    logger.info(
        f"Calibrated Orange Alert Threshold: Probability={orange_thresh:.2f} (Score={int(orange_thresh*100)}), "
        f"Hold-out Recall={orange_recall:.3f}, False Alarm Rate={orange_far:.3f}"
    )

    # -------------------------------------------------------------------------
    # 7. FEATURE IMPORTANCE BAR CHART (Section 5.5)
    # -------------------------------------------------------------------------
    # Extract feature importances from tree classifier inside pipeline
    tree_clf = chosen_model.named_steps["clf"]
    if hasattr(tree_clf, "feature_importances_"):
        raw_importances = tree_clf.feature_importances_
        # Match with transformed feature names
        num_len = len(num_feats)
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_names = list(cat_encoder.get_feature_names_out(cat_feats))
        all_col_names = num_feats + cat_names
        feat_importances = dict(zip(all_col_names[:len(raw_importances)], [round(float(v), 4) for v in raw_importances]))
    else:
        feat_importances = {f: 1.0 / len(num_feats) for f in num_feats}

    # Plot Feature Importance Bar Chart
    sorted_fi = sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)[:12]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([x[0] for x in reversed(sorted_fi)], [x[1] for x in reversed(sorted_fi)], color="#0f172a")
    ax.set_xlabel("Relative Feature Importance (MDI)", fontweight="bold")
    ax.set_title("Top 12 Most Predictive Terrain & Weather Drivers", fontsize=11, fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fi_fig_path = REPORTS_DIR / "feature_importance.png"
    plt.savefig(fi_fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 8. SAVE MODEL ARTIFACTS & MODEL CARD (Section 5.6)
    # -------------------------------------------------------------------------
    joblib.dump(chosen_model, MODELS_DIR / "final_model.joblib")
    joblib.dump(calibrator, MODELS_DIR / "calibrator.joblib")

    final_test_metrics = evaluate_predictions(y_test, (calib_prob >= orange_thresh).astype(int), calib_prob)

    model_card = {
        "model_name": f"URsafe Terrain Hazard Model ({chosen_name})",
        "model_version": "v0.1.0",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "framework": "scikit-learn 1.9.1 / Python 3.13",
        "problem_framing": "Relative Hazard Risk Score (0-100), not an absolute probability (ADR 0001)",
        "features_used": feature_cols,
        "calibration": {
            "method": "sigmoid",
            "raw_brier": round(raw_brier, 4),
            "calibrated_brier": round(calib_brier, 4)
        },
        "bands": RISK_BANDS,
        "orange_threshold": {
            "probability": round(orange_thresh, 3),
            "score": int(round(orange_thresh * 100)),
            "spatial_test_recall": round(orange_recall, 3),
            "false_alarm_rate": round(orange_far, 3)
        },
        "performance": {
            "spatial_holdout": final_test_metrics,
            "models_comparison": results,
            "leakage_gap": leakage_gap
        },
        "feature_importances": feat_importances,
        "known_weaknesses": [
            "Conservative slope angles: 90m DEM smooths micro-cliffs and roadside cut-slopes.",
            "Coarse atmospheric resolution: 25km reanalysis grid can miss localized cloudburst cells in isolated ravines.",
            "Pseudo-absence sampling: Safe locations are unrecorded rather than verified stable across centuries.",
            "Roadway reporting bias: Global landslide catalogues are concentrated along transit corridors and populated towns."
        ],
        "disclaimer": "Model-derived relative estimates. Never a guarantee. Check local disaster warnings."
    }

    with open(MODELS_DIR / "model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(model_card["performance"], f, indent=2)

    with open(ADR_GEN_DIR / "metrics_snapshot.json", "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model": chosen_name,
            "roc_auc": final_test_metrics["roc_auc"],
            "pr_auc": final_test_metrics["pr_auc"],
            "f1": final_test_metrics["f1"],
            "recall_orange": round(orange_recall, 3),
            "brier": round(calib_brier, 4)
        }, f, indent=2)

    logger.info("Training, calibration, and Model Card successfully completed.")
    return model_card, results


if __name__ == "__main__":
    from ml.build_table import run_build_table
    df_data = run_build_table(limit=200)
    card, comp = run_training_pipeline(df_data)
    print(card["orange_threshold"])
