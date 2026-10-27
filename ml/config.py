"""URsafe ML Pipeline Configuration.

Defines spatial bounds, volume requirements, filepaths, feature definitions,
and hyperparameter grids for reproducible autonomous execution.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
RAW_DATA_DIR = DATASETS_DIR / "raw"
INTERIM_DATA_DIR = DATASETS_DIR / "interim"
PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
CACHE_DIR = DATASETS_DIR / "cache"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
SCREENSHOTS_DIR = REPORTS_DIR / "screenshots"
ADR_DIR = BASE_DIR / "adr"
ADR_GEN_DIR = ADR_DIR / "generated"

# Ensure all directories exist
for p in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR,
          MODELS_DIR, REPORTS_DIR, SCREENSHOTS_DIR, ADR_DIR, ADR_GEN_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Random Seed for strict reproducibility
RANDOM_SEED = 42

# Spatial Regions (lon_min, lon_max, lat_min, lat_max)
REGIONS = {
    "south_asia": {
        "name": "South Asia",
        "lon_min": 60.0, "lon_max": 100.0,
        "lat_min": 5.0,  "lat_max": 38.0
    },
    "pan_asia": {
        "name": "Pan-Asia",
        "lon_min": 60.0, "lon_max": 150.0,
        "lat_min": -10.0, "lat_max": 55.0
    },
    "global": {
        "name": "Global",
        "lon_min": -180.0, "lon_max": 180.0,
        "lat_min": -90.0,  "lat_max": 90.0
    }
}

# Volume Constraints (Section 3.1)
MIN_POSITIVES = 2000
MAX_POSITIVES = 6000
NEGATIVE_RATIO = 2  # 2 negatives per positive sample
MIN_DISTANCE_TO_EVENT_KM = 5.0  # Buffer distance for pseudo-absences

# Spatial Autocorrelation Control
SPATIAL_BLOCK_SIZE_DEG = 1.0  # 1.0 degree block grid for GroupShuffleSplit

# Network & Ingestion Config
USER_AGENT = "URsafe-TerrainHazard/1.0 (academic-safety-research; contact: engineering@ursafe.org)"
REQUEST_TIMEOUT = 25  # seconds
MAX_RETRIES = 3
GRID_CELL_CACHE_DEG = 0.25  # Grouping for atmospheric cache

# Traveler Profile Multipliers (Section 5.8)
PROFILE_MULTIPLIERS = {
    "hiker": 1.0,
    "rider": 1.0,
    "family": 1.20,
    "elderly": 1.25
}

# Hazard Risk Bands
RISK_BANDS = {
    "Green": {"min": 0, "max": 25, "label": "Low Risk", "action": "Standard terrain safety; routine caution."},
    "Yellow": {"min": 26, "max": 55, "label": "Moderate Risk", "action": "Rolling terrain or light precipitation; stay alert."},
    "Orange": {"min": 56, "max": 80, "label": "High Risk", "action": "Elevated vulnerability; consider alternative routes."},
    "Red": {"min": 81, "max": 100, "label": "Severe Hazard", "action": "Extreme hazard; avoid travel until storm clears."}
}

# Feature Columns
CONTINUOUS_TERRAIN_FEATURES = [
    "elevation_m",
    "slope_deg",
    "roughness",
    "relief_m"
]

REQUIRED_FEATURES = [
    "elevation_m",
    "slope_deg",
    "aspect_sin",
    "aspect_cos",
    "curvature",
    "roughness",
    "relief_m",
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "rain_14d",
    "rain_30d",
    "rain_max_day_30d",
    "api_30d",
    "soil_wetness_top",
    "soil_wetness_root_7d",
    "distance_to_water_m",
    "log_dist_to_water",
    "month_sin",
    "month_cos"
]

CATEGORICAL_FEATURES = [
    "region_group"
]

ALL_MODEL_FEATURES = REQUIRED_FEATURES + CATEGORICAL_FEATURES
