"""Prediction Route Endpoint: /api/predict.

Accepts geographical coordinates, traveler persona, and optional historical date.
Returns relative hazard risk score, alert band, confidence, terrain zone with soft memberships,
30-day observed + 3-day forecast rainfall, top SHAP drivers, and plain-language AI Companion advice.
"""
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.live_features import get_live_features, get_realtime_location_status, fetch_current_weather, reverse_geocode
from ml.companion import ai_companion_guide
from ml.config import MODELS_DIR, CONTINUOUS_TERRAIN_FEATURES, RISK_BANDS
from ml.explain import compute_sample_attributions

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Prediction"])

# Lazy-loaded model singletons
_CALIBRATOR = None
_GMM = None
_SCALER = None
_ZONE_MAP = None


def get_models():
    """Load or retrieve cached model artifacts."""
    global _CALIBRATOR, _GMM, _SCALER, _ZONE_MAP
    if _CALIBRATOR is None and (MODELS_DIR / "calibrator.joblib").exists():
        _CALIBRATOR = joblib.load(MODELS_DIR / "calibrator.joblib")
    if _GMM is None and (MODELS_DIR / "terrain_gmm.joblib").exists():
        _GMM = joblib.load(MODELS_DIR / "terrain_gmm.joblib")
    if _SCALER is None and (MODELS_DIR / "terrain_scaler.joblib").exists():
        _SCALER = joblib.load(MODELS_DIR / "terrain_scaler.joblib")
    if _ZONE_MAP is None and (MODELS_DIR / "zone_names_map.json").exists():
        import json
        with open(MODELS_DIR / "zone_names_map.json") as f:
            _ZONE_MAP = {int(k): v for k, v in json.load(f).items()}
    return _CALIBRATOR, _GMM, _SCALER, _ZONE_MAP


class PredictRequest(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    profile: str = Field("hiker", description="Traveler persona: hiker | rider | family | elderly")
    date: Optional[str] = Field(None, description="Optional ISO date YYYY-MM-DD (defaults to current date)")


@router.get("/status")
def get_location_status_telemetry(lat: float, lon: float) -> Dict[str, Any]:
    """Retrieve instant real-time weather and terrain status without running full ML pipeline."""
    try:
        return get_realtime_location_status(lat, lon)
    except Exception as e:
        logger.exception("Status telemetry endpoint failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
def predict_hazard(req: PredictRequest) -> Dict[str, Any]:
    """Execute real-time terrain hazard prediction and companion guidance."""
    try:
        # 1. Fetch multi-source features, provenance, real-time weather & location
        features, rain_chart, provenance = get_live_features(req.lat, req.lon, req.date)
        status_telemetry = get_realtime_location_status(req.lat, req.lon)
        live_weather = status_telemetry["weather"]
        location_name = status_telemetry["location_name"]
        hydrology = status_telemetry["hydrology"]
        terrain_stability = status_telemetry["terrain"]

        # 2. Predict calibrated relative risk
        calibrator, gmm, scaler, zone_map = get_models()
        df_row = pd.DataFrame([features])

        if calibrator is not None:
            try:
                prob = float(calibrator.predict_proba(df_row)[:, 1][0])
            except Exception as e:
                logger.warning(f"Calibrator prediction error: {e}")
                prob = 0.35
        else:
            # Domain heuristic fallback if pipeline hasn't completed
            slope = features.get("slope_deg", 5.0)
            rain3d = features.get("rain_3d", 10.0)
            prob = min(max((slope * 0.015) + (rain3d * 0.005), 0.05), 0.95)

        # 3. Soft Terrain Zone Memberships (Gaussian Mixture Model)
        zone_info = {
            "name": "Moderate Slopes",
            "cluster_id": 1,
            "description": "Mid-elevation hilly terrain with moderate gradient.",
            "memberships": {"Low": 0.20, "Moderate": 0.65, "High": 0.15}
        }
        if gmm is not None and scaler is not None and zone_map is not None:
            try:
                terrain_vec = [[features.get(f, 0.0) for f in CONTINUOUS_TERRAIN_FEATURES]]
                vec_scaled = scaler.transform(terrain_vec)
                c_id = int(gmm.predict(vec_scaled)[0])
                soft_probs = gmm.predict_proba(vec_scaled)[0]
                memberships = {zone_map.get(i, f"Zone {i}"): round(float(p), 3) for i, p in enumerate(soft_probs)}
                zone_info = {
                    "name": zone_map.get(c_id, "Moderate Slopes"),
                    "cluster_id": c_id,
                    "description": f"Classified into terrain cluster {c_id} ({zone_map.get(c_id)}).",
                    "memberships": memberships
                }
            except Exception as e:
                logger.warning(f"GMM zone inference error: {e}")

        # 4. SHAP Feature Contributions
        shap_dict = compute_sample_attributions(df_row)

        # 5. Universal AI Companion Guidance with real-time weather & location
        companion_payload = ai_companion_guide(
            terrain_sample=features,
            profile=req.profile,
            raw_probability=prob,
            shap_contributions=shap_dict,
            terrain_zone=zone_info,
            live_weather=live_weather,
            location_name=location_name
        )

        return {
            "status": "success",
            "model_version": "v0.1.0",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "coordinates": {"lat": round(req.lat, 5), "lon": round(req.lon, 5)},
            "location_name": location_name,
            "date": req.date or datetime.utcnow().strftime("%Y-%m-%d"),
            "current_weather": live_weather,
            "hydrology_status": hydrology,
            "terrain_stability": terrain_stability,
            "features": features,
            "feature_provenance": provenance,
            "rainfall_chart": rain_chart,
            **companion_payload
        }
    except Exception as e:
        logger.exception("Prediction endpoint encountered failure")
        raise HTTPException(status_code=500, detail=str(e))