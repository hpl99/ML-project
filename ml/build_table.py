"""Training Table Assembly and Physical Validation Module.

Combines real positive landslide events and pseudo-absences, ingests multi-source
terrain, meteorological, and hydrological features using 0.25° grid caching, applies
physical bounds validation, computes spatial block indices, and writes processed artifacts.
"""
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd

from ml.config import (
    PROCESSED_DATA_DIR, ADR_GEN_DIR, REQUIRED_FEATURES,
    CATEGORICAL_FEATURES, SPATIAL_BLOCK_SIZE_DEG, RANDOM_SEED
)
from ml.fetch_features import (
    get_grid_cell, fetch_cell_rainfall_series, extract_sample_features
)
from ml.fetch_labels import get_positive_labels
from ml.sample_negatives import get_negative_labels

logger = logging.getLogger(__name__)


def enforce_physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Flag physically invalid measurements as NaN per Section 5.1 & ADR 0005."""
    df_clean = df.copy()

    # Slope physically bounded between 0 and 90 degrees
    if "slope_deg" in df_clean.columns:
        invalid_slope = (df_clean["slope_deg"] < 0.0) | (df_clean["slope_deg"] > 90.0)
        df_clean.loc[invalid_slope, "slope_deg"] = np.nan

    # Precipitation non-negative
    rain_cols = [c for c in df_clean.columns if c.startswith("rain_") or c == "api_30d"]
    for rc in rain_cols:
        df_clean.loc[df_clean[rc] < 0.0, rc] = np.nan

    # Soil wetness between 0 and 1
    for sc in ["soil_wetness_top", "soil_wetness_root_7d"]:
        if sc in df_clean.columns:
            invalid_soil = (df_clean[sc] < 0.0) | (df_clean[sc] > 1.0)
            df_clean.loc[invalid_soil, sc] = np.nan

    # Terrestrial elevation bounds (-450m Dead Sea to 8848m Everest)
    if "elevation_m" in df_clean.columns:
        invalid_elev = (df_clean["elevation_m"] < -450.0) | (df_clean["elevation_m"] > 9000.0)
        df_clean.loc[invalid_elev, "elevation_m"] = np.nan

    return df_clean


def assign_spatial_blocks(df: pd.DataFrame, block_size: float = SPATIAL_BLOCK_SIZE_DEG) -> pd.DataFrame:
    """Assign spatial block ID to each coordinate for leakage-free GroupShuffleSplit (ADR 0006)."""
    df = df.copy()
    b_lats = np.floor(df["latitude"] / block_size).astype(int)
    b_lons = np.floor(df["longitude"] / block_size).astype(int)
    df["spatial_block_id"] = [f"block_{lat}_{lon}" for lat, lon in zip(b_lats, b_lons)]
    return df


def _process_single_cell(cell_info: Tuple[Tuple[float, float], list]) -> Dict[int, dict]:
    """Process all samples in a single cell using targeted date windows."""
    (c_lat, c_lon), rows = cell_info
    dts = [datetime.strptime(r["date_str"], "%Y-%m-%d") for _, r in rows]
    min_dt = min(dts) - timedelta(days=35)
    max_dt = max(dts)

    rain_series = fetch_cell_rainfall_series(
        cell_lat=c_lat,
        cell_lon=c_lon,
        start_date_str=min_dt.strftime("%Y-%m-%d"),
        end_date_str=max_dt.strftime("%Y-%m-%d")
    )

    cell_results = {}
    for idx, row in rows:
        feat_dict = extract_sample_features(
            lat=row["latitude"],
            lon=row["longitude"],
            date_str=row["date_str"],
            rain_series=rain_series
        )
        full_row = row.to_dict()
        full_row.update(feat_dict)
        cell_results[idx] = full_row
    return cell_results


def build_feature_table(df_samples: pd.DataFrame) -> pd.DataFrame:
    """Extract features efficiently by grouping samples into 0.25 degree cells with concurrency."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import timedelta

    logger.info(f"Extracting features for {len(df_samples)} total samples across geography...")

    # Group samples by 0.25 degree cell
    cell_groups = defaultdict(list)
    for idx, row in df_samples.iterrows():
        c_lat, c_lon = get_grid_cell(row["latitude"], row["longitude"])
        cell_groups[(c_lat, c_lon)].append((idx, row))

    logger.info(f"Unique 0.25° grid cells to process: {len(cell_groups)}")

    extracted_rows = {}
    cell_items = list(cell_groups.items())

    # Process using ThreadPoolExecutor
    workers = min(8, len(cell_items))
    completed_cells = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_single_cell, item): item for item in cell_items}
        for future in as_completed(futures):
            try:
                res = future.result()
                extracted_rows.update(res)
            except Exception as e:
                logger.warning(f"Error processing cell: {e}")
            completed_cells += 1
            if completed_cells % 50 == 0 or completed_cells == len(cell_items):
                logger.info(f"Progress: processed {completed_cells}/{len(cell_items)} cells.")

    # Reassemble DataFrame preserving index
    records = [extracted_rows.get(i) for i in df_samples.index if i in extracted_rows]
    df_features = pd.DataFrame(records)

    # Physical bounds check
    df_features = enforce_physical_bounds(df_features)

    # Spatial block assignment
    df_features = assign_spatial_blocks(df_features)

    return df_features


def save_held_out_replay_events(df_positives: pd.DataFrame, n_events: int = 12) -> Path:
    """Save diverse real historical landslide events for interactive UI replay."""
    # Select events across varied dates and geographies
    replay_sample = df_positives.sample(n=min(n_events, len(df_positives)), random_state=RANDOM_SEED).copy()
    replay_list = []
    for _, row in replay_sample.iterrows():
        replay_list.append({
            "event_id": row["event_id"],
            "date": row["date_str"],
            "latitude": round(float(row["latitude"]), 4),
            "longitude": round(float(row["longitude"]), 4),
            "region_group": str(row.get("region_group", "South Asia")),
            "description": f"Real Landslide Event on {row['date_str']} ({row.get('region_group', 'South Asia')})"
        })

    out_path = PROCESSED_DATA_DIR / "held_out_events.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(replay_list, f, indent=2)
    logger.info(f"Saved {len(replay_list)} held-out replay events to {out_path}")
    return out_path


def run_build_table(limit: int = None, refresh: bool = False) -> pd.DataFrame:
    """Main orchestrator for dataset merging, feature calculation, and artifact creation."""
    table_parquet = PROCESSED_DATA_DIR / "train_table.parquet"
    table_csv = PROCESSED_DATA_DIR / "train_table.csv"

    if not refresh and table_parquet.exists() and (limit is None or len(pd.read_parquet(table_parquet)) >= limit):
        logger.info(f"Loading cached processed table from: {table_parquet}")
        return pd.read_parquet(table_parquet)

    # 1. Positives
    df_pos = get_positive_labels(limit=limit, refresh=refresh)

    # 2. Negatives
    df_neg = get_negative_labels(df_pos, limit=limit, refresh=refresh)

    # Save replay events from positives
    save_held_out_replay_events(df_pos)

    # Combine into single sample table
    df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
    df_combined = df_combined.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    logger.info(f"Combined sample set: {len(df_combined)} rows ({len(df_pos)} pos, {len(df_neg)} neg).")

    # 3. Build features
    df_table = build_feature_table(df_combined)

    # Save final artifacts
    df_table.to_parquet(table_parquet, index=False)
    df_table.to_csv(table_csv, index=False)
    logger.info(f"Final training table saved to {table_parquet} ({len(df_table)} rows, {df_table.shape[1]} cols).")

    # Write summary stats to adr/generated/dataset_stats.json
    stats = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_rows": len(df_table),
        "positives_count": int((df_table["label"] == 1).sum()),
        "negatives_count": int((df_table["label"] == 0).sum()),
        "class_ratio": f"1:{round(int((df_table['label'] == 0).sum()) / max(int((df_table['label'] == 1).sum()), 1), 1)}",
        "date_min": df_table["date_str"].min(),
        "date_max": df_table["date_str"].max(),
        "spatial_blocks_count": int(df_table["spatial_block_id"].nunique()),
        "features_count": len(REQUIRED_FEATURES) + len(CATEGORICAL_FEATURES),
        "columns": df_table.columns.tolist()
    }
    with open(ADR_GEN_DIR / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return df_table


if __name__ == "__main__":
    df = run_build_table(limit=200)
    print(df.head())
