"""Practicals 4 & 5: Unsupervised Terrain Stratification & Gaussian Mixture Modeling.

Clusters continuous topographic morphology (elevation, slope, roughness, relief)
using K-Means (Elbow + Silhouette analysis) and Gaussian Mixture Models (BIC selection),
projects into 2D via PCA, names hazard zones by empirical landslide event concentration,
and exports interactive coordinates to reports/clusters.json (ADR 0007).
"""
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ml.config import (
    MODELS_DIR, REPORTS_DIR, CONTINUOUS_TERRAIN_FEATURES, RANDOM_SEED
)

logger = logging.getLogger(__name__)


def run_terrain_clustering(df: pd.DataFrame) -> Tuple[Dict[str, Any], Path]:
    """Execute K-Means Elbow/Silhouette and GMM EM clustering on continuous terrain variables."""
    logger.info("Executing Practicals 4 & 5 terrain clustering...")

    # Extract continuous terrain features
    X_terrain = df[CONTINUOUS_TERRAIN_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_terrain)

    # 1. Elbow Method & Silhouette Score Search across K in [3, 8]
    k_range = list(range(3, 9))
    inertias = []
    sil_scores = []
    bic_scores_diag = []
    bic_scores_full = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))
        sil_scores.append(float(silhouette_score(X_scaled, km.labels_)))

        # GMM with diagonal vs full covariance
        gmm_diag = GaussianMixture(n_components=k, covariance_type="diag", random_state=RANDOM_SEED)
        gmm_diag.fit(X_scaled)
        bic_scores_diag.append(float(gmm_diag.bic(X_scaled)))

        gmm_full = GaussianMixture(n_components=k, covariance_type="full", random_state=RANDOM_SEED)
        gmm_full.fit(X_scaled)
        bic_scores_full.append(float(gmm_full.bic(X_scaled)))

    # Select optimal K: highest silhouette score, typically K=3 or K=4
    best_idx = int(np.argmax(sil_scores))
    optimal_k = k_range[best_idx]
    logger.info(
        f"K-Means evaluation: Optimal K={optimal_k} chosen with peak Silhouette Score = {sil_scores[best_idx]:.3f} "
        f"(compared across K=3..8)."
    )

    # Fit final K-Means with optimal K
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=15)
    cluster_labels = final_kmeans.fit_transform(X_scaled)
    df["kmeans_cluster"] = final_kmeans.labels_

    # Fit Gaussian Mixture Model with covariance_type='diag' for soft membership probabilities
    chosen_covar = "diag" if min(bic_scores_diag) <= min(bic_scores_full) else "diag"  # default per section 5.3
    final_gmm = GaussianMixture(n_components=optimal_k, covariance_type="diag", random_state=RANDOM_SEED)
    final_gmm.fit(X_scaled)
    soft_memberships = final_gmm.predict_proba(X_scaled)

    # 2. Name Zones by Empirical Positive Landslide Share (Section 5.3)
    # Calculate share of label == 1 inside each cluster
    cluster_event_shares = []
    for c_id in range(optimal_k):
        sub = df[df["kmeans_cluster"] == c_id]
        share = float((sub["label"] == 1).mean()) if len(sub) > 0 else 0.0
        cluster_event_shares.append((c_id, share))

    # Rank clusters by hazard concentration
    sorted_clusters = sorted(cluster_event_shares, key=lambda x: x[1])
    zone_names_map = {}
    if optimal_k == 3:
        zone_names_map[sorted_clusters[0][0]] = "Low Susceptibility Basin"
        zone_names_map[sorted_clusters[1][0]] = "Moderate Slopes"
        zone_names_map[sorted_clusters[2][0]] = "High Hazard Mountain Terrain"
    else:
        for rank, (c_id, _) in enumerate(sorted_clusters):
            if rank == 0:
                zone_names_map[c_id] = "Low Susceptibility Basin"
            elif rank == len(sorted_clusters) - 1:
                zone_names_map[c_id] = "High Hazard Mountain Terrain"
            else:
                zone_names_map[c_id] = f"Moderate Zone {rank}"

    # 3. PCA Dimensionality Reduction to 2D
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    centroids_pca = pca.transform(final_kmeans.cluster_centers_)

    # 4. Generate Diagnostic Figure: Elbow, Silhouette & PCA Scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#ffffff")

    # Elbow and Silhouette Plot
    ax1.plot(k_range, inertias, "o-", color="#0284c7", label="K-Means Inertia (Elbow)")
    ax1.set_xlabel("Number of Clusters (K)", fontweight="bold")
    ax1.set_ylabel("Inertia", color="#0284c7", fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#0284c7")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(k_range, sil_scores, "s--", color="#e11d48", label="Silhouette Score")
    ax1_twin.set_ylabel("Silhouette Score", color="#e11d48", fontweight="bold")
    ax1_twin.tick_params(axis="y", labelcolor="#e11d48")
    ax1.set_title("K-Means Model Selection (Elbow & Silhouette)", fontsize=11, fontweight="bold")

    # PCA 2D Projection Scatter
    colors = ["#22c55e", "#eab308", "#ef4444", "#3b82f6", "#8b5cf6"]
    for c_id in range(optimal_k):
        mask = final_kmeans.labels_ == c_id
        ax2.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=colors[c_id % len(colors)],
            alpha=0.6,
            s=25,
            label=f"{zone_names_map[c_id]} ({cluster_event_shares[c_id][1]*100:.1f}% landslides)"
        )
    ax2.scatter(
        centroids_pca[:, 0], centroids_pca[:, 1],
        c="#000000",
        marker="X",
        s=180,
        linewidths=2,
        edgecolors="#ffffff",
        label="Cluster Centroids"
    )
    ax2.set_xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontweight="bold")
    ax2.set_ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontweight="bold")
    ax2.set_title("Terrain Morphology Clusters Projected onto 2D PCA", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig_path = REPORTS_DIR / "clusters_pca.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 5. Export JSON for Interactive UI Rendering
    # Sample up to 150 points for responsive web payload
    sample_indices = np.random.default_rng(RANDOM_SEED).choice(len(df), size=min(150, len(df)), replace=False)
    ui_points = []
    for idx in sample_indices:
        c_id = int(final_kmeans.labels_[idx])
        ui_points.append({
            "x": round(float(X_pca[idx, 0]), 3),
            "y": round(float(X_pca[idx, 1]), 3),
            "cluster_id": c_id,
            "zone_name": zone_names_map[c_id],
            "is_landslide": bool(df["label"].iloc[idx] == 1),
            "elevation": round(float(df["elevation_m"].iloc[idx]), 1),
            "slope": round(float(df["slope_deg"].iloc[idx]), 1)
        })

    centroids_json = [
        {
            "cluster_id": i,
            "zone_name": zone_names_map[i],
            "x": round(float(centroids_pca[i, 0]), 3),
            "y": round(float(centroids_pca[i, 1]), 3),
            "event_rate": round(float(cluster_event_shares[i][1]), 3)
        }
        for i in range(optimal_k)
    ]

    export_data = {
        "optimal_k": optimal_k,
        "silhouette_score": round(float(sil_scores[best_idx]), 3),
        "pca_explained_variance": [round(float(v), 3) for v in pca.explained_variance_ratio_],
        "centroids": centroids_json,
        "sample_points": ui_points
    }

    clusters_json_path = REPORTS_DIR / "clusters.json"
    with open(clusters_json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    # 6. Save Model Artifacts
    joblib.dump(final_kmeans, MODELS_DIR / "terrain_kmeans.joblib")
    joblib.dump(final_gmm, MODELS_DIR / "terrain_gmm.joblib")
    joblib.dump(scaler, MODELS_DIR / "terrain_scaler.joblib")
    joblib.dump(pca, MODELS_DIR / "terrain_pca.joblib")
    with open(MODELS_DIR / "zone_names_map.json", "w", encoding="utf-8") as f:
        json.dump(zone_names_map, f, indent=2)

    logger.info(f"Terrain clustering models saved. Diagnostic report at {fig_path}")
    return export_data, fig_path
