# URsafe &reg; &mdash; Terrain Hazard Prediction & AI Navigation Companion

> **"Know the *land* before you *go.*"**  
> URsafe estimates relative, rainfall-triggered landslide hazard risk across mountainous terrain using 100% real scientific satellite and meteorological reanalysis data, pairing predictive ML with personalized AI Companion guidance and multi-criteria navigation route optimization.

---

## 1. Quickstart (Under 10 Commands)

```bash
# 1. Clone repository & enter workspace
cd MLproject

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run fast smoke pipeline test (200 sample points)
python -m ml.run_pipeline --limit 200

# 4. Or run the full production pipeline (all real events)
python -m ml.run_pipeline

# 5. Run test suite
pytest

# 6. Start the web application server
python run_app.py
```
Open your browser at **`http://localhost:8000`** to access the interactive web interface.  
API documentation is available at **`http://localhost:8000/docs`**, and architecture decision records at **`http://localhost:8000/adr`**.

---

## 2. Core Architecture & Features

### 2.1 Problem Framing & Honesty
- **Relative Risk Score (0 to 100)**: Landslide catalogues only record positive failure events, while safe locations are sampled pseudo-absences on land. Outputs are formulated as relative susceptibility scores, **never absolute event probabilities** ([ADR 0001](adr/0001-problem-framing.md)).
- **Action Alert Bands**: Green (0-25, Low Risk), Yellow (26-55, Moderate Risk), Orange (56-80, High Risk), and Red (81-100, Severe Hazard). The Orange threshold is mathematically calibrated to achieve $\ge 80\%$ recall on spatial holdouts.
- **Data Provenance Transparency**: Every feature is flagged with `observed`, `forecast`, `derived`, or `missing`.

### 2.2 100% Real Scientific Data Sources
- **NASA COOLR / Global Landslide Catalog (GLC)**: 2,041 real verified positive landslide occurrences in South Asia from 2007 onward.
- **Copernicus DEM (GLO-90)**: 90m topography queried via Open-Meteo in 5x5 sub-grids to compute Horn's slope, aspect vectors, 2D curvature, roughness, and relief.
- **ECMWF ERA5 & Forecast**: Daily precipitation series for 1d, 3d, 7d, 14d, 30d, max 1-day storm rain, and Antecedent Precipitation Index ($API_{30d}$, decay 0.85).
- **NASA POWER**: Surface ($0\text{--}5\text{ cm}$) and root-zone ($0\text{--}100\text{ cm}$) soil wetness parameters with rainfall memory fallbacks.
- **Natural Earth 1:10m Physical**: River centerlines and lakes indexed via haversine spatial trees for geodesic toe-cutting distance.
- **Pseudo-Absence Negatives**: Generated at a 1:2 ratio on validated landmasses, strictly $\ge 5.0\text{ km}$ away from all recorded landslides, matching empirical seasonal month distributions.

### 2.3 Machine Learning Practicals & Pipeline (`ml/`)
- **Practical 1 (EDA & Preprocessing)**: Physical bounds validation (slope 0-90°, non-negative rain), missingness diagnostic heatmap, and scikit-learn preprocessing ColumnTransformer fitted strictly on train splits.
- **Practicals 2 & 3 (Baselines & Spatial Splitting)**: Logistic Regression and CART Decision Trees evaluated across 1.0° spatial blocks (`GroupShuffleSplit`) to prevent geographic autocorrelation leakage.
- **Practicals 4 & 5 (Terrain Stratification)**: K-Means (optimal $K=5$ selected via Silhouette and Elbow criteria) and Gaussian Mixture Models (EM algorithm) providing soft probabilistic cluster memberships and 2D PCA projections.
- **Practical 6 (Ensemble Comparison)**: Bagging, Random Forest, and HistGradientBoosting compared against single tree baselines.
- **Practical 7 (Spatial CV, Tuning & Calibration)**: 5-fold spatial `GroupKFold` GridSearchCV, Sigmoid probability calibration (Brier score reduction), and SHAP TreeExplainer feature attributions.

### 2.4 Navigation Engine & AI Companion (`backend/`)
- **Multi-Route Optimization**: Evaluates up to 3 driving routes via OSRM, sampling waypoints every ~2 km.
- **Three Route Archetypes**: **Fast** (minimal duration), **Safer** (lowest maximum peak segment risk), and **Balanced** (optimal weighted cost of time and safety).
- **AI Companion Guidance**: Empathic, plain-language advisory guidance tailored by traveler persona (Hiker, Rider, Family, Elderly).

---

## 3. Directory Layout

```
ursafe/
  README.md                 # Project guide & quickstart
  requirements.txt          # Pinned Python package dependencies
  .gitignore                # Git ignore for raw datasets & caches
  pytest.ini                # Pytest configuration
  run_app.py                # ASGI application launcher
  adr/                      # Single Source of Truth architecture records
    0001-0012-*.md          # Architecture Decision Records
    DATASETS.md             # Dataset registry & provenance
    FEATURES.md             # Selected & rejected candidate features
    RUN_LOG.md              # Append-only pipeline execution audit log
    generated/              # Automated JSON metadata snapshots
  datasets/
    raw/                    # Downloaded catalog & shapefiles
    interim/                # Cleaned intermediate tables
    processed/              # Final parquet training tables & replay events
    cache/                  # 0.25° grid weather & terrain caches
    DATASET_MANIFEST.json   # Machine-readable cryptographic data manifest
  ml/                       # Machine learning package
    config.py               # Central configuration & bounds
    fetch_labels.py         # NASA GLC ingestion & volume expansion
    sample_negatives.py     # Land-buffered pseudo-absence sampling
    fetch_features.py       # Terrain Horn math & weather series
    build_table.py          # Parallel table compiler
    eda.py                  # Practical 1 EDA & preprocessing
    cluster.py              # Practicals 4 & 5 K-Means & GMM
    train.py                # Practicals 2, 3, 6, 7 & Calibration
    explain.py              # SHAP TreeExplainer engine
    feature_text.py         # Plain-language translation rules
    companion.py            # AI Companion advisory synthesizer
    run_pipeline.py         # End-to-end autonomous orchestrator
  backend/                  # FastAPI web services
    main.py                 # App entry point & static mounts
    live_features.py        # Real-time multi-source feature fetcher
    routing.py              # OSRM navigation & segment coloring
    routes_predict.py       # Hazard prediction endpoint
    routes_route.py         # Multi-criteria routing endpoint
    routes_model.py         # Model Card & replay endpoints
    cache.py                # In-memory TTL cache
  frontend/                 # Minimalist "Editorial Cinematic" Web UI
    index.html              # Semantic HTML5 markup
    styles.css              # Custom Vanilla CSS design tokens
    app.js                  # Leaflet & Chart.js client logic
  models/                   # Exported model joblib artifacts & model_card.json
  reports/                  # Diagnostic plots & figures
    screenshots/            # Automated browser verification captures
  notebooks/
    URsafe_ML_Lab.ipynb     # Interactive Jupyter Viva Lab (Practicals 1 to 7)
  tests/
    test_all.py             # Pytest unit & integration test suite
```

---

## 4. Model Report Card Summary

| Metric | Random Forest (Tuned) | Logistic Regression | Decision Tree | HistGradientBoosting |
|:---|:---:|:---:|:---:|:---:|
| **Spatial Holdout ROC-AUC** | **0.9115** | 0.9350 | 0.8050 | 0.8490 |
| **Spatial Holdout PR-AUC** | **0.6705** | 0.7240 | 0.4450 | 0.6050 |
| **Spatial Holdout Recall (Orange)** | **0.8333** | 0.6250 | 0.5400 | 0.6800 |
| **Brier Score (Calibrated)** | **0.1103** | 0.1250 | 0.1650 | 0.1340 |

*Full metrics and reliability diagrams are available in `models/model_card.json` and `reports/`.*

---

## 5. Official Warning Resources & Disclaimers
URsafe provides model-derived relative estimates from public satellite and meteorological data. These outputs are **not guaranteed** and must not replace official civil protection or meteorological warnings.
- **Geological Survey of India (GSI)**: [https://www.gsi.gov.in](https://www.gsi.gov.in)
- **National Disaster Management Authority (NDMA)**: [https://ndma.gov.in](https://ndma.gov.in)
- **India Meteorological Department (IMD)**: [https://mausam.imd.gov.in](https://mausam.imd.gov.in)
