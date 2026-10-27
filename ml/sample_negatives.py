"""Pseudo-Absence (Negative) Sampling Module.

Synthesizes representative negative samples on validated continental landmasses,
enforcing a minimum 5.0 km exclusion buffer from all known positive landslide
events and matching empirical seasonal month distributions (ADR 0003).
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from sklearn.neighbors import BallTree

from ml.config import (
    RAW_DATA_DIR, INTERIM_DATA_DIR, DATASETS_DIR, REGIONS,
    NEGATIVE_RATIO, MIN_DISTANCE_TO_EVENT_KM, RANDOM_SEED,
    USER_AGENT, REQUEST_TIMEOUT
)
from ml.fetch_labels import compute_sha256, assign_region_group

logger = logging.getLogger(__name__)

NATURAL_EARTH_LAND_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_land.geojson"
)


def fetch_land_polygon(dest_path: Path) -> Path:
    """Download Natural Earth 50m land polygons if not cached."""
    if dest_path.exists() and dest_path.stat().st_size > 100_000:
        logger.info(f"Using cached land polygons at: {dest_path}")
        return dest_path

    logger.info(f"Downloading Natural Earth land polygons from: {NATURAL_EARTH_LAND_URL}")
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(NATURAL_EARTH_LAND_URL, headers=headers, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 200 and len(resp.content) > 100_000:
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"Saved land polygons ({len(resp.content)} bytes) to {dest_path}")
        return dest_path
    raise RuntimeError("Failed to fetch Natural Earth land polygons.")


def load_regional_land_geometry(geojson_path: Path, region_bounds: dict):
    """Load and merge land geometries intersecting the target bounding box."""
    with open(geojson_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    min_lon, max_lon = region_bounds["lon_min"], region_bounds["lon_max"]
    min_lat, max_lat = region_bounds["lat_min"], region_bounds["lat_max"]

    matching_geoms = []
    for feat in data.get("features", []):
        geom = shape(feat["geometry"])
        bounds = geom.bounds  # minx, miny, maxx, maxy
        # Rough bbox intersection test
        if not (bounds[2] < min_lon or bounds[0] > max_lon or bounds[3] < min_lat or bounds[1] > max_lat):
            matching_geoms.append(geom)

    logger.info(f"Found {len(matching_geoms)} land polygon features intersecting region.")
    if not matching_geoms:
        # Fallback to union of all
        return unary_union([shape(feat["geometry"]) for feat in data.get("features", [])])
    return unary_union(matching_geoms)


def sample_pseudo_absences(df_positives: pd.DataFrame, ratio: int = NEGATIVE_RATIO) -> pd.DataFrame:
    """Generate pseudo-absences at random land locations >= 5km from positives."""
    n_negatives = len(df_positives) * ratio
    logger.info(f"Sampling {n_negatives} pseudo-absences ({ratio}:1 negative-to-positive ratio).")

    land_path = RAW_DATA_DIR / "ne_land.geojson"
    fetch_land_polygon(land_path)

    # Determine region bounding box from positive events
    min_lon, max_lon = float(df_positives["longitude"].min()), float(df_positives["longitude"].max())
    min_lat, max_lat = float(df_positives["latitude"].min()), float(df_positives["latitude"].max())

    # Add 0.5 degree buffer
    min_lon = max(min_lon - 0.5, -180.0)
    max_lon = min(max_lon + 0.5, 180.0)
    min_lat = max(min_lat - 0.5, -90.0)
    max_lat = min(max_lat + 0.5, 90.0)

    bounds = {"lon_min": min_lon, "lon_max": max_lon, "lat_min": min_lat, "lat_max": max_lat}
    land_geom = load_regional_land_geometry(land_path, bounds)

    # Build BallTree of positive event coordinates in radians for fast geodesic queries
    # Earth radius ~6371 km
    pos_coords_rad = np.radians(df_positives[["latitude", "longitude"]].values)
    pos_tree = BallTree(pos_coords_rad, metric="haversine")

    buffer_rad = MIN_DISTANCE_TO_EVENT_KM / 6371.0  # 5 km in radians

    # Extract empirical dates from positives to match seasonality
    pos_dates = df_positives["event_date"].dropna().values
    rng = np.random.default_rng(RANDOM_SEED)

    negatives = []
    attempts = 0
    max_attempts = n_negatives * 50

    while len(negatives) < n_negatives and attempts < max_attempts:
        batch_size = min(max((n_negatives - len(negatives)) * 3, 100), 5000)
        rand_lats = rng.uniform(min_lat, max_lat, size=batch_size)
        rand_lons = rng.uniform(min_lon, max_lon, size=batch_size)
        attempts += batch_size

        # Test distance buffer against all known positives
        cand_rad = np.radians(np.column_stack([rand_lats, rand_lons]))
        # Query distance to nearest positive
        distances, _ = pos_tree.query(cand_rad, k=1)
        valid_dist_mask = distances.flatten() >= buffer_rad

        for lat, lon in zip(rand_lats[valid_dist_mask], rand_lons[valid_dist_mask]):
            if len(negatives) >= n_negatives:
                break
            # Test if point lies inside continental land polygon
            pt = Point(lon, lat)
            if land_geom.contains(pt):
                # Pick a random date from positive events to mirror seasonality
                sample_dt = rng.choice(pos_dates)
                negatives.append({
                    "event_id": f"pseudo_neg_{len(negatives)+1:06d}",
                    "latitude": round(lat, 5),
                    "longitude": round(lon, 5),
                    "event_date": sample_dt,
                    "date_str": pd.to_datetime(sample_dt).strftime("%Y-%m-%d"),
                    "region_group": assign_region_group(lat, lon),
                    "label": 0
                })

    if len(negatives) < n_negatives:
        logger.warning(f"Could only generate {len(negatives)} / {n_negatives} negatives after {attempts} attempts.")

    df_negatives = pd.DataFrame(negatives)
    logger.info(f"Generated {len(df_negatives)} valid pseudo-absence negative points.")
    return df_negatives


def get_negative_labels(df_positives: pd.DataFrame, limit: int = None, refresh: bool = False) -> pd.DataFrame:
    """Main orchestrator for pseudo-absence generation."""
    interim_path = INTERIM_DATA_DIR / f"negatives_{limit or 'full'}.parquet"
    if not refresh and interim_path.exists():
        logger.info(f"Loading cached interim negatives from: {interim_path}")
        return pd.read_parquet(interim_path)

    df_neg = sample_pseudo_absences(df_positives, ratio=NEGATIVE_RATIO)
    df_neg.to_parquet(interim_path, index=False)
    df_neg.to_csv(INTERIM_DATA_DIR / f"negatives_{limit or 'full'}.csv", index=False)

    # Update manifest for Natural Earth land
    manifest_path = DATASETS_DIR / "DATASET_MANIFEST.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            land_path = RAW_DATA_DIR / "ne_land.geojson"
            entry = {
                "dataset_name": "Natural Earth 50m Land Polygons",
                "primary_url": NATURAL_EARTH_LAND_URL,
                "license": "Public Domain",
                "download_date": datetime.now().strftime("%Y-%m-%d"),
                "rows": len(df_neg),
                "sha256": compute_sha256(land_path) if land_path.exists() else None,
                "raw_file": "ne_land.geojson"
            }
            sources = [s for s in data.get("sources", []) if s.get("dataset_name") != entry["dataset_name"]]
            sources.append(entry)
            data["sources"] = sources
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update manifest with land data: {e}")

    return df_neg


if __name__ == "__main__":
    from ml.fetch_labels import get_positive_labels
    pos = get_positive_labels(limit=200)
    neg = get_negative_labels(pos, limit=200)
    print(neg.head())
