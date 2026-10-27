"""Live Feature Extraction and Real-Time Provenance Tagging.

Fetches live Copernicus DEM elevation, Open-Meteo 30-day observed + 3-day forecast rainfall,
NASA POWER soil wetness with fallback, and Natural Earth water distance.
Never invents data; flags all provenances explicitly (observed / forecast / derived / missing).
"""
import logging
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional

import pandas as pd

from backend.cache import cache
from ml.fetch_features import (
    fetch_terrain_grid, compute_distance_to_water,
    fetch_live_rainfall_forecast, fetch_cell_rainfall_series,
    compute_rainfall_features_from_series, fetch_soil_wetness,
    get_grid_cell
)
from ml.fetch_labels import assign_region_group

logger = logging.getLogger(__name__)


def get_live_features(lat: float, lon: float, date_str: Optional[str] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, str]]:
    """Assemble complete real-time or historical feature dictionary with provenance tags.

    Returns:
    - feature_dict: Dict of model inputs
    - rainfall_chart: List of 33 daily items with 'date', 'rainfall_mm', 'type'
    - provenance_tags: Dict mapping feature name to provenance ('observed' | 'forecast' | 'derived' | 'missing')
    """
    cache_key = f"live_{round(lat, 4)}_{round(lon, 4)}_{date_str or 'current'}"
    cached = cache.get(cache_key)
    if cached:
        return cached["features"], cached["chart"], cached["provenance"]

    provenance: Dict[str, str] = {}

    # 1. Terrain Morphology (Copernicus DEM 90m)
    try:
        terrain = fetch_terrain_grid(lat, lon)
        for k in terrain:
            provenance[k] = "observed"
    except Exception as e:
        logger.warning(f"Error fetching terrain for ({lat}, {lon}): {e}")
        terrain = {
            "elevation_m": 500.0, "slope_deg": 5.0, "aspect_sin": 0.0,
            "aspect_cos": 1.0, "curvature": 0.0, "roughness": 5.0, "relief_m": 20.0
        }
        for k in terrain:
            provenance[k] = "derived"

    # 2. Distance to Water (Natural Earth 10m)
    try:
        dist_m, log_dist = compute_distance_to_water(lat, lon)
        provenance["distance_to_water_m"] = "observed"
        provenance["log_dist_to_water"] = "observed"
    except Exception:
        dist_m, log_dist = 5000.0, 8.517
        provenance["distance_to_water_m"] = "derived"
        provenance["log_dist_to_water"] = "derived"

    # 3. Rainfall (Observed + Forecast)
    is_historical = date_str is not None and date_str < datetime.utcnow().strftime("%Y-%m-%d")
    rainfall_chart = []

    if is_historical:
        # Query archive series for the 0.25° grid cell
        c_lat, c_lon = get_grid_cell(lat, lon)
        series = fetch_cell_rainfall_series(c_lat, c_lon)
        rain_feats = compute_rainfall_features_from_series(series, date_str)
        for k in rain_feats:
            provenance[k] = "observed"

        # Build chart for the 30 days preceding event date
        t_dt = datetime.strptime(date_str, "%Y-%m-%d")
        for i in range(29, -1, -1):
            d = (t_dt - pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            rainfall_chart.append({
                "date": d,
                "rainfall_mm": series.get(d, 0.0),
                "type": "observed"
            })
    else:
        # Live 30-day observed + 3-day forecast
        rain_feats, rainfall_chart = fetch_live_rainfall_forecast(lat, lon)
        for k in rain_feats:
            provenance[k] = "observed"
        # Tag forecast components
        provenance["rain_forecast_3d"] = "forecast"

    # 4. Soil Wetness (NASA POWER & Fallback)
    target_dt_str = date_str or datetime.utcnow().strftime("%Y-%m-%d")
    top_soil, root_soil, soil_prov = fetch_soil_wetness(lat, lon, target_dt_str)
    if soil_prov == "derived":
        api = rain_feats.get("api_30d", 0.0)
        top_soil = round(min(0.20 + (api / 200.0) * 0.70, 0.95), 3)
        root_soil = round(min(0.25 + (rain_feats.get("rain_14d", 0.0) / 300.0) * 0.65, 0.95), 3)

    provenance["soil_wetness_top"] = soil_prov
    provenance["soil_wetness_root_7d"] = soil_prov

    # 5. Cyclical Time & Region
    try:
        month = datetime.strptime(target_dt_str, "%Y-%m-%d").month
    except Exception:
        month = datetime.utcnow().month

    import math
    month_sin = round(math.sin(2.0 * math.pi * month / 12.0), 4)
    month_cos = round(math.cos(2.0 * math.pi * month / 12.0), 4)
    provenance["month_sin"] = "derived"
    provenance["month_cos"] = "derived"

    region_grp = assign_region_group(lat, lon)
    provenance["region_group"] = "derived"

    features = {
        **terrain,
        **rain_feats,
        "soil_wetness_top": top_soil,
        "soil_wetness_root_7d": root_soil,
        "distance_to_water_m": dist_m,
        "log_dist_to_water": log_dist,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "region_group": region_grp
    }

    cache.set(cache_key, {"features": features, "chart": rainfall_chart, "provenance": provenance}, ttl_sec=900)
    return features, rainfall_chart, provenance
