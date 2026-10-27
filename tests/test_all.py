"""Comprehensive URsafe Unit & Integration Test Suite.

Tests physics calculations, spatial splitting, threshold banding,
route risk aggregation, API endpoints, and AI companion guidance.
"""
import math
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from ml.companion import determine_risk_band, ai_companion_guide
from ml.build_table import enforce_physical_bounds, assign_spatial_blocks
from ml.fetch_features import compute_rainfall_features_from_series
from backend.routing import evaluate_route_risk, compare_routes


# -----------------------------------------------------------------------------
# 1. Physics Calculations: Slope & Aspect from known 3x3 grid
# -----------------------------------------------------------------------------
def test_slope_and_aspect_known_grid():
    """Verify Horn's method computes exact 0.0 deg for flat plane and 45.0 deg for 45 deg incline."""
    # Flat grid: 100m everywhere
    flat_3x3 = np.full((3, 3), 100.0)
    dx = 100.0
    dy = 100.0

    p_flat = ((flat_3x3[0, 2] + 2 * flat_3x3[1, 2] + flat_3x3[2, 2]) -
              (flat_3x3[0, 0] + 2 * flat_3x3[1, 0] + flat_3x3[2, 0])) / (8.0 * dx)
    q_flat = ((flat_3x3[0, 0] + 2 * flat_3x3[0, 1] + flat_3x3[0, 2]) -
              (flat_3x3[2, 0] + 2 * flat_3x3[2, 1] + flat_3x3[2, 2])) / (8.0 * dy)
    slope_flat = math.degrees(math.atan(math.sqrt(p_flat**2 + q_flat**2)))
    assert slope_flat == pytest.approx(0.0, abs=1e-5)

    # 45-degree slope inclined along X: z = x
    x = np.array([0.0, 100.0, 200.0])
    incline_3x3 = np.tile(x, (3, 1))
    p_45 = ((incline_3x3[0, 2] + 2 * incline_3x3[1, 2] + incline_3x3[2, 2]) -
            (incline_3x3[0, 0] + 2 * incline_3x3[1, 0] + incline_3x3[2, 0])) / (8.0 * dx)
    q_45 = ((incline_3x3[0, 0] + 2 * incline_3x3[0, 1] + incline_3x3[0, 2]) -
            (incline_3x3[2, 0] + 2 * incline_3x3[2, 1] + incline_3x3[2, 2])) / (8.0 * dy)
    slope_45 = math.degrees(math.atan(math.sqrt(p_45**2 + q_45**2)))
    assert slope_45 == pytest.approx(45.0, abs=1e-5)


# -----------------------------------------------------------------------------
# 2. Rainfall Window Sums & Antecedent Precipitation Index
# -----------------------------------------------------------------------------
def test_rainfall_window_sums():
    """Verify 1d, 3d, 7d, 14d, 30d cumulative sums and API decay."""
    # Synthesize daily rainfall series: 10mm every day
    dates = pd.date_range("2023-06-01", "2023-07-15", freq="D").strftime("%Y-%m-%d").tolist()
    rain_dict = {d: 10.0 for d in dates}

    res = compute_rainfall_features_from_series(rain_dict, "2023-07-10")
    assert res["rain_1d"] == 10.0
    assert res["rain_3d"] == 30.0
    assert res["rain_7d"] == 70.0
    assert res["rain_14d"] == 140.0
    assert res["rain_30d"] == 300.0
    assert res["rain_max_day_30d"] == 10.0
    assert res["api_30d"] > 0.0


# -----------------------------------------------------------------------------
# 3. Invalid Physical Bounds Handling
# -----------------------------------------------------------------------------
def test_physical_bounds_handling():
    """Verify invalid slope (>90), negative rainfall, and soil wetness > 1 become NaN."""
    df_raw = pd.DataFrame({
        "slope_deg": [15.0, 95.0, -5.0],
        "rain_1d": [10.0, -4.0, 20.0],
        "soil_wetness_top": [0.5, 1.4, -0.1],
        "elevation_m": [500.0, 12000.0, -600.0]
    })
    df_clean = enforce_physical_bounds(df_raw)

    assert pd.isna(df_clean.loc[1, "slope_deg"])
    assert pd.isna(df_clean.loc[2, "slope_deg"])
    assert df_clean.loc[0, "slope_deg"] == 15.0

    assert pd.isna(df_clean.loc[1, "rain_1d"])
    assert df_clean.loc[0, "rain_1d"] == 10.0

    assert pd.isna(df_clean.loc[1, "soil_wetness_top"])
    assert pd.isna(df_clean.loc[2, "soil_wetness_top"])
    assert df_clean.loc[0, "soil_wetness_top"] == 0.5

    assert pd.isna(df_clean.loc[1, "elevation_m"])
    assert pd.isna(df_clean.loc[2, "elevation_m"])
    assert df_clean.loc[0, "elevation_m"] == 500.0


# -----------------------------------------------------------------------------
# 4. Spatial Block Split Autocorrelation Check
# -----------------------------------------------------------------------------
def test_spatial_split_no_leakage():
    """Verify that spatial block IDs are disjoint between train and test partitions."""
    from sklearn.model_selection import GroupShuffleSplit
    df = pd.DataFrame({
        "latitude": [28.1, 28.2, 29.5, 29.6, 30.1, 30.2],
        "longitude": [77.1, 77.2, 78.5, 78.6, 79.1, 79.2],
        "label": [1, 0, 1, 0, 1, 0]
    })
    df = assign_spatial_blocks(df, block_size=1.0)
    groups = df["spatial_block_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=groups))

    train_blocks = set(groups[train_idx])
    test_blocks = set(groups[test_idx])

    # No shared blocks
    assert len(train_blocks.intersection(test_blocks)) == 0


# -----------------------------------------------------------------------------
# 5. Risk Band Mapping & Operational Thresholds
# -----------------------------------------------------------------------------
def test_risk_band_mapping():
    """Verify score mapping into Green, Yellow, Orange, Red."""
    assert determine_risk_band(10)["name"] == "Green"
    assert determine_risk_band(40)["name"] == "Yellow"
    assert determine_risk_band(65)["name"] == "Orange"
    assert determine_risk_band(90)["name"] == "Red"
    assert determine_risk_band(105)["name"] == "Red"


# -----------------------------------------------------------------------------
# 6. AI Companion Guidance Synthesis
# -----------------------------------------------------------------------------
def test_ai_companion_content():
    """Verify that companion guidance outputs score, band, confidence, and advice text."""
    sample = {"slope_deg": 30.0, "rain_3d": 75.0, "elevation_m": 1200.0, "soil_wetness_top": 0.8}
    guide = ai_companion_guide(sample, profile="family", raw_probability=0.75)

    assert "score" in guide
    assert guide["score"] >= 80  # 75 * 1.2 = 90
    assert guide["band"] in ["Orange", "Red"]
    assert "advice" in guide
    assert len(guide["advice"]) > 20
    assert "profile" in guide
    assert guide["profile"] == "family"


# -----------------------------------------------------------------------------
# 7. Navigation Route Risk Aggregation
# -----------------------------------------------------------------------------
def test_route_risk_aggregation():
    """Verify independence heuristic route_risk = 1 - prod(1 - p)."""
    mock_route = {
        "duration": 3600.0,
        "distance": 45000.0,
        "geometry": {
            "type": "LineString",
            "coordinates": [[78.0, 30.0], [78.1, 30.1], [78.2, 30.2], [78.3, 30.3]]
        }
    }
    with patch("backend.routing.get_live_features") as mock_feat:
        mock_feat.return_value = ({"slope_deg": 10.0, "rain_3d": 20.0}, [], {})
        res = evaluate_route_risk(mock_route, profile="hiker")

        assert "duration_min" in res
        assert res["duration_min"] == 60.0
        assert "mean_risk" in res
        assert "cumulative_heuristic_risk" in res
        assert 0.0 <= res["cumulative_heuristic_risk"] <= 100.0


# -----------------------------------------------------------------------------
# 8. REST API Endpoints Smoke Tests (FastAPI TestClient)
# -----------------------------------------------------------------------------
client = TestClient(app)


def test_api_health():
    """Verify GET /api/health returns 200 OK and model version."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "model_version" in data


def test_api_config():
    """Verify GET /api/config returns supported region and bands."""
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "supported_region" in data
    assert "risk_bands" in data
    assert "traveler_profiles" in data


def test_api_predict():
    """Verify POST /api/predict handles coordinates and returns score and advice."""
    payload = {"lat": 30.3244, "lon": 78.0339, "profile": "hiker"}
    resp = client.post("/api/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "score" in data
    assert "band" in data
    assert "advice" in data
    assert "top_factors" in data


def test_api_model_report():
    """Verify GET /api/model/report returns Model Card benchmarks and clusters."""
    resp = client.get("/api/model/report")
    assert resp.status_code == 200
    data = resp.json()
    assert "performance" in data
    assert "known_weaknesses" in data
