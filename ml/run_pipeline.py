"""URsafe Master Autonomous ML Pipeline Orchestrator.

Executes end-to-end data ingestion, negative sampling, feature engineering,
exploratory data analysis, terrain clustering, ensemble training, spatial CV,
probability calibration, explainability setup, and ADR/manifest synchronization.

Supports:
    python -m ml.run_pipeline --limit 200    # Fast smoke test
    python -m ml.run_pipeline                # Full production run
"""
import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from ml.build_table import run_build_table
from ml.cluster import run_terrain_clustering
from ml.config import (
    ADR_DIR, ADR_GEN_DIR, REPORTS_DIR, MODELS_DIR, DATASETS_DIR,
    ALL_MODEL_FEATURES, REQUIRED_FEATURES, CATEGORICAL_FEATURES
)
from ml.eda import generate_eda_report
from ml.train import run_training_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("URsafePipeline")


def get_git_commit() -> str:
    """Retrieve current short git commit hash."""
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "local_dev"


def update_feature_manifest(df: pd.DataFrame):
    """Write machine-readable adr/generated/feature_manifest.json."""
    manifest = []
    for f in REQUIRED_FEATURES:
        if f in df.columns:
            missing_rate = round(float(df[f].isna().mean()), 4)
            corr_label = round(float(df[[f, "label"]].corr().iloc[0, 1]), 4) if "label" in df.columns else 0.0
            manifest.append({
                "feature": f,
                "type": "numeric",
                "missing_rate": missing_rate,
                "correlation_with_label": corr_label,
                "selected": True,
                "reason": "Meets physics checks and non-collinearity thresholds (<0.95)"
            })
    for f in CATEGORICAL_FEATURES:
        if f in df.columns:
            manifest.append({
                "feature": f,
                "type": "categorical",
                "missing_rate": 0.0,
                "correlation_with_label": None,
                "selected": True,
                "reason": "Geoclimatic separation"
            })

    out_path = ADR_GEN_DIR / "feature_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"features": manifest, "generated_at": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    logger.info(f"Updated feature manifest at {out_path}")


def append_to_run_log(command_str: str, pos_count: int, neg_count: int, final_model_name: str, roc_auc: float, pr_auc: float, warnings_notes: str):
    """Append row to adr/RUN_LOG.md table per Section 8."""
    run_log_path = ADR_DIR / "RUN_LOG.md"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    commit_hash = get_git_commit()
    total_rows = pos_count + neg_count
    feat_count = len(ALL_MODEL_FEATURES)

    row_line = (
        f"| {timestamp} | `{commit_hash}` | `{command_str}` | {pos_count} | {neg_count} | {total_rows} | "
        f"{feat_count} features | {final_model_name} | {roc_auc:.4f} | {pr_auc:.4f} | {warnings_notes} |\n"
    )

    with open(run_log_path, "a", encoding="utf-8") as f:
        f.write(row_line)
    logger.info(f"Appended run record to {run_log_path}")


def main():
    parser = argparse.ArgumentParser(description="URsafe Machine Learning Pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit for fast smoke testing (e.g. 200)")
    parser.add_argument("--refresh-data", action="store_true", help="Force re-download of raw datasets")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 80)
    logger.info(f"STARTING URSAFE ML PIPELINE (limit={args.limit}, refresh={args.refresh_data})")
    logger.info("=" * 80)

    # 1. Build Features and Training Table
    logger.info(">>> PHASE 1 & 2: Data Ingestion & Table Assembly...")
    df_table = run_build_table(limit=args.limit, refresh=args.refresh_data)

    pos_count = int((df_table["label"] == 1).sum())
    neg_count = int((df_table["label"] == 0).sum())
    logger.info(f"Dataset compiled: {len(df_table)} rows ({pos_count} landslides, {neg_count} pseudo-absences).")

    # 2. Exploratory Data Analysis (Practical 1)
    logger.info(">>> PHASE 3: Exploratory Data Analysis & Diagnostic Plots...")
    eda_stats, eda_fig = generate_eda_report(df_table)
    update_feature_manifest(df_table)

    # 3. Unsupervised Terrain Stratification (Practicals 4 & 5)
    logger.info(">>> PHASE 4: Terrain Morphology Clustering & GMM Fitting...")
    cluster_meta, cluster_fig = run_terrain_clustering(df_table)

    # 4. Supervised Training, CV, Tuning & Calibration (Practicals 2, 3, 6, 7 & 5.6)
    logger.info(">>> PHASE 5: Model Training, Ensembling, Calibration & Model Card...")
    model_card, models_comp = run_training_pipeline(df_table)

    # 5. Append to RUN_LOG.md
    elapsed_sec = round(time.time() - start_time, 1)
    cmd = f"python -m ml.run_pipeline{' --limit ' + str(args.limit) if args.limit else ''}"
    best_model = model_card["model_name"]
    holdout = model_card["performance"]["spatial_holdout"]

    notes = f"Completed in {elapsed_sec}s. Optimal K={cluster_meta['optimal_k']}. Orange band threshold={model_card['orange_threshold']['score']}."
    append_to_run_log(
        command_str=cmd,
        pos_count=pos_count,
        neg_count=neg_count,
        final_model_name=best_model,
        roc_auc=holdout["roc_auc"],
        pr_auc=holdout["pr_auc"],
        warnings_notes=notes
    )

    logger.info("=" * 80)
    logger.info(f"URSAFE PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_sec}s")
    logger.info(f"Holdout Spatial Metrics: ROC-AUC={holdout['roc_auc']:.4f}, PR-AUC={holdout['pr_auc']:.4f}, Recall={holdout['recall']:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
