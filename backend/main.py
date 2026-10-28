"""FastAPI Application Entry Point.

Serves the REST API endpoints and mounts static assets for the frontend
and architectural decision records (/adr).
"""
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

# Ensure project root is available in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.cache import cache
from backend.routes_model import router as model_router
from backend.routes_predict import router as predict_router
from backend.routes_route import router as route_router
from ml.config import (
    BASE_DIR, REGIONS, RISK_BANDS, PROFILE_MULTIPLIERS,
    REQUIRED_FEATURES, CATEGORICAL_FEATURES, USER_AGENT
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("URsafeBackend")

app = FastAPI(
    title="URsafe API",
    description="Terrain Hazard Prediction and AI Navigation Companion Backend",
    version="v0.1.0"
)

# CORS enabled for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Sub-Routers
app.include_router(predict_router)
app.include_router(route_router)
app.include_router(model_router)


# -----------------------------------------------------------------------------
# System & Configuration Endpoints
# -----------------------------------------------------------------------------
@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Return system status, active model version, and data freshness."""
    model_card_path = BASE_DIR / "models" / "model_card.json"
    trained_at = None
    if model_card_path.exists():
        try:
            import json
            with open(model_card_path, "r") as f:
                trained_at = json.load(f).get("trained_at")
        except Exception:
            pass

    return {
        "status": "healthy",
        "service": "URsafe Hazard Prediction Engine",
        "model_version": "v0.1.0",
        "model_trained_at": trained_at or "Training in progress / Cached",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "upstream_data_freshness": "Real-time satellite & meteorological reanalysis"
    }


@app.get("/api/config")
def get_system_config() -> Dict[str, Any]:
    """Return supported region bounds, risk bands, persona multipliers, and disclaimers."""
    return {
        "app_name": "URsafe",
        "tagline": "Know the land before you go.",
        "supported_region": REGIONS["south_asia"],
        "all_regions": REGIONS,
        "risk_bands": RISK_BANDS,
        "traveler_profiles": PROFILE_MULTIPLIERS,
        "features": {
            "required": REQUIRED_FEATURES,
            "categorical": CATEGORICAL_FEATURES
        },
        "disclaimer": "Outputs are model-derived relative estimates from public satellite and meteorological data. Not a guarantee. Check local government warnings.",
        "official_resources": [
            {"agency": "Geological Survey of India (GSI)", "url": "https://www.gsi.gov.in"},
            {"agency": "National Disaster Management Authority (NDMA)", "url": "https://ndma.gov.in"},
            {"agency": "India Meteorological Department (IMD)", "url": "https://mausam.imd.gov.in"}
        ]
    }


@app.get("/api/geocode")
def geocode_place(q: str = Query(..., min_length=2, description="Place or city name")) -> List[Dict[str, Any]]:
    """Proxy to Open-Meteo Geocoding API with caching."""
    cache_key = f"geocode_{q.lower().strip()}"
    cached = cache.get(cache_key)
    if cached:
        return cached

    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": q, "count": 5, "language": "en", "format": "json"}
    headers = {"User-Agent": USER_AGENT}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            results = r.json().get("results", [])
            places = [
                {
                    "name": item.get("name"),
                    "latitude": item.get("latitude"),
                    "longitude": item.get("longitude"),
                    "country": item.get("country", ""),
                    "admin1": item.get("admin1", "")
                }
                for item in results
            ]
            cache.set(cache_key, places, ttl_sec=86400)  # 24h cache
            return places
    except Exception as e:
        logger.warning(f"Geocoding error for '{q}': {e}")

    return []


@app.get("/api/reverse-geocode")
def reverse_geocode_coords(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180)) -> Dict[str, Any]:
    """Reverse geocode coordinates to human readable place name."""
    from backend.live_features import reverse_geocode
    place = reverse_geocode(lat, lon)
    return {"place_name": place, "lat": lat, "lon": lon}



# -----------------------------------------------------------------------------
# Static Mounts (ADR, Reports, Frontend)
# -----------------------------------------------------------------------------
from fastapi.responses import RedirectResponse

adr_dir = BASE_DIR / "adr"
if adr_dir.exists():
    @app.get("/adr")
    def redirect_adr():
        return RedirectResponse(url="/adr/")
    app.mount("/adr", StaticFiles(directory=str(adr_dir), html=True), name="adr")

reports_dir = BASE_DIR / "reports"
if reports_dir.exists():
    app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")

frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print("  URsafe: Starting Server via backend.main")
    print("  Web Interface:      http://127.0.0.1:8000")
    print("  API Documentation:  http://127.0.0.1:8000/docs")
    print("=" * 70)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)

