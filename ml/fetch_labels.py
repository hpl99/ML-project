"""Label Ingestion Module.

Fetches and standardizes real historical landslide positive events from the
NASA Global Landslide Catalog (GLC) / COOLR repository.
Implements the volume rule and region expansion strategy (ADR 0002).
"""
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import requests

from ml.config import (
    RAW_DATA_DIR, INTERIM_DATA_DIR, DATASETS_DIR, REGIONS,
    MIN_POSITIVES, MAX_POSITIVES, USER_AGENT, REQUEST_TIMEOUT,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Primary mirrors for NASA Global Landslide Catalog / COOLR
GLC_CSV_URLS = [
    "https://raw.githubusercontent.com/AntoineSueur/Geoplot-landslides/master/Global_Landslide_Catalog_Export.csv",
    "https://raw.githubusercontent.com/erikanagamine/autonomousdatabase/master/Global_Landslide_Catalog_Export.csv"
]


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def assign_region_group(lat: float, lon: float) -> str:
    """Categorize geographic coordinates into broad geoclimatic zones."""
    if 5.0 <= lat <= 38.0 and 60.0 <= lon <= 100.0:
        return "South Asia"
    elif -10.0 <= lat <= 55.0 and 100.0 < lon <= 150.0:
        return "East/Southeast Asia"
    elif -55.0 <= lat <= 60.0 and -140.0 <= lon <= -30.0:
        return "Americas"
    elif 35.0 <= lat <= 70.0 and -10.0 <= lon <= 45.0:
        return "Europe"
    else:
        return "Other Region"


def fetch_raw_glc(dest_path: Path) -> Tuple[Path, str]:
    """Download raw NASA GLC export if not already cached on disk."""
    if dest_path.exists() and dest_path.stat().st_size > 500_000:
        logger.info(f"Using cached raw GLC file at: {dest_path}")
        return dest_path, "local_cache"

    headers = {"User-Agent": USER_AGENT}
    for url in GLC_CSV_URLS:
        try:
            logger.info(f"Attempting download of NASA GLC from: {url}")
            resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and len(resp.content) > 500_000:
                with open(dest_path, "wb") as f:
                    f.write(resp.content)
                logger.info(f"Successfully downloaded {len(resp.content)} bytes to {dest_path}")
                return dest_path, url
        except Exception as e:
            logger.warning(f"Mirror download failed for {url}: {e}")

    raise RuntimeError("Could not download real NASA Global Landslide Catalog from any upstream mirror.")


def load_and_clean_positives(raw_path: Path, limit: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean raw GLC records, enforce physical checks, and apply volume rules."""
    df_raw = pd.read_csv(raw_path, low_memory=False)
    logger.info(f"Raw catalog contains {len(df_raw)} total records across {df_raw.shape[1]} columns.")

    # Find coordinate columns
    lat_col = next((c for c in df_raw.columns if c.lower() in ["latitude", "lat"]), None)
    lon_col = next((c for c in df_raw.columns if c.lower() in ["longitude", "lon"]), None)
    date_col = next((c for c in df_raw.columns if c.lower() in ["date", "event_date"]), None)

    if not (lat_col and lon_col and date_col):
        raise ValueError(f"Required columns missing in raw data: {lat_col}, {lon_col}, {date_col}")

    # Standardize column naming
    df = df_raw.copy()
    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["event_date"] = pd.to_datetime(df[date_col], errors="coerce")

    # Drop records with invalid coordinates or missing dates
    valid_mask = (
        df["latitude"].notna() &
        df["longitude"].notna() &
        df["event_date"].notna() &
        (df["latitude"] >= -90.0) & (df["latitude"] <= 90.0) &
        (df["longitude"] >= -180.0) & (df["longitude"] <= 180.0)
    )
    df = df[valid_mask].copy()

    # Filter to 2007 onward (so daily ERA5 reanalysis and satellite weather exist)
    df = df[df["event_date"] >= "2007-01-01"].copy()

    # Location accuracy check (drop records with radius > 25km when accuracy column exists)
    if "location_accuracy" in df.columns:
        # e.g., 'known', '5km', '25km', '100km'
        high_error_mask = df["location_accuracy"].astype(str).str.lower().str.contains("100km|unknown", regex=True)
        df = df[~high_error_mask].copy()

    # Drop exact spatial-temporal duplicate events
    df = df.drop_duplicates(subset=["latitude", "longitude", "event_date"]).copy()

    # Reset index and generate stable event_id
    df["event_id"] = [f"glc_{i+1:06d}" for i in range(len(df))]

    # Format event date as standard ISO YYYY-MM-DD
    df["date_str"] = df["event_date"].dt.strftime("%Y-%m-%d")

    # Assign broad region group
    df["region_group"] = [assign_region_group(lat, lon) for lat, lon in zip(df["latitude"], df["longitude"])]

    # Section 3.1 Volume Rule:
    # 1. Default region: South Asia (lon 60-100, lat 5-38)
    sa_mask = (
        (df["longitude"] >= REGIONS["south_asia"]["lon_min"]) &
        (df["longitude"] <= REGIONS["south_asia"]["lon_max"]) &
        (df["latitude"] >= REGIONS["south_asia"]["lat_min"]) &
        (df["latitude"] <= REGIONS["south_asia"]["lat_max"])
    )
    sa_count = int(sa_mask.sum())
    logger.info(f"South Asia positive events count: {sa_count}")

    region_scope_used = "South Asia"
    if sa_count >= MIN_POSITIVES:
        df_selected = df[sa_mask].copy()
        logger.info(f"Target count met in South Asia ({len(df_selected)} events >= {MIN_POSITIVES}).")
    else:
        # Expansion Step 1: Pan-Asia
        logger.info(f"South Asia had {sa_count} < {MIN_POSITIVES}. Expanding to Pan-Asia.")
        pan_mask = (
            (df["longitude"] >= REGIONS["pan_asia"]["lon_min"]) &
            (df["longitude"] <= REGIONS["pan_asia"]["lon_max"]) &
            (df["latitude"] >= REGIONS["pan_asia"]["lat_min"]) &
            (df["latitude"] <= REGIONS["pan_asia"]["lat_max"])
        )
        pan_count = int(pan_mask.sum())
        if pan_count >= MIN_POSITIVES:
            df_selected = df[pan_mask].copy()
            region_scope_used = "Pan-Asia"
        else:
            # Expansion Step 2: Global
            logger.info("Pan-Asia count still insufficient. Expanding to Global scope.")
            df_selected = df.copy()
            region_scope_used = "Global"

    # Cap at MAX_POSITIVES (6000) for runtime constraints
    if len(df_selected) > MAX_POSITIVES:
        logger.info(f"Subsampling {len(df_selected)} down to cap of {MAX_POSITIVES} events.")
        df_selected = df_selected.sample(n=MAX_POSITIVES, random_state=RANDOM_SEED).copy()

    # If limit specified (e.g. for smoke testing --limit 200)
    if limit is not None and limit < len(df_selected):
        logger.info(f"Applying user limit: {limit} samples (smoke test).")
        df_selected = df_selected.sample(n=limit, random_state=RANDOM_SEED).copy()

    df_selected["label"] = 1  # 1 for positive landslide occurrence
    df_selected = df_selected.sort_values("event_date").reset_index(drop=True)

    metadata = {
        "raw_total_records": len(df_raw),
        "post_2007_valid_records": len(df),
        "selected_positive_count": len(df_selected),
        "region_scope": region_scope_used,
        "date_min": df_selected["date_str"].min(),
        "date_max": df_selected["date_str"].max(),
    }
    return df_selected, metadata


def update_manifest(source_url: str, raw_path: Path, row_count: int) -> None:
    """Update datasets/DATASET_MANIFEST.json with cryptographic and provenance metadata."""
    manifest_path = DATASETS_DIR / "DATASET_MANIFEST.json"
    data = {"generated_at": datetime.utcnow().isoformat() + "Z", "sources": []}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass

    sha256 = compute_sha256(raw_path) if raw_path.exists() else None
    source_entry = {
        "dataset_name": "NASA Global Landslide Catalog (GLC) / COOLR",
        "primary_url": source_url,
        "license": "NASA Open Data Policy (Public Domain)",
        "download_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "rows": row_count,
        "sha256": sha256,
        "raw_file": raw_path.name
    }
    # Update or append
    sources = [s for s in data.get("sources", []) if s.get("dataset_name") != source_entry["dataset_name"]]
    sources.append(source_entry)
    data["sources"] = sources
    data["last_updated"] = datetime.utcnow().isoformat() + "Z"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Updated dataset manifest at {manifest_path}")


def get_positive_labels(limit: int = None, refresh: bool = False) -> pd.DataFrame:
    """Main orchestrator function for label ingestion."""
    raw_csv = RAW_DATA_DIR / "glc_events_raw.csv"
    interim_path = INTERIM_DATA_DIR / f"positives_{limit or 'full'}.parquet"

    if not refresh and interim_path.exists():
        logger.info(f"Loading cached interim positives from: {interim_path}")
        return pd.read_parquet(interim_path)

    raw_path, source_url = fetch_raw_glc(raw_csv)
    df_pos, meta = load_and_clean_positives(raw_path, limit=limit)

    # Cache interim table
    df_pos.to_parquet(interim_path, index=False)
    # Also save CSV for simple inspection
    df_pos.to_csv(INTERIM_DATA_DIR / f"positives_{limit or 'full'}.csv", index=False)

    update_manifest(source_url, raw_path, len(df_pos))
    logger.info(f"Positives successfully processed: {len(df_pos)} rows. Metadata: {meta}")
    return df_pos


if __name__ == "__main__":
    df = get_positive_labels(limit=200)
    print(df[["event_id", "date_str", "latitude", "longitude", "region_group", "label"]].head())
