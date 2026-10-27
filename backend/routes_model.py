"""Model Card, Event Replay, and Manifest Endpoints.

Serves model metrics, comparison tables, confusion matrices, calibration curves,
interactive cluster coordinates, and held-out test events for replay.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException

from ml.config import MODELS_DIR, REPORTS_DIR, DATASETS_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Model"])


@router.get("/model/report")
@router.get("/model/summary")
def get_model_report() -> Dict[str, Any]:
    """Retrieve complete Model Card, cross-validation benchmarks, and cluster data."""
    card_path = MODELS_DIR / "model_card.json"
    cluster_path = REPORTS_DIR / "clusters.json"

    report: Dict[str, Any] = {
        "model_name": "URsafe Terrain Hazard Estimator",
        "model_version": "v0.1.0",
        "performance": {},
        "clusters": {},
        "features": []
    }

    if card_path.exists():
        try:
            with open(card_path, "r", encoding="utf-8") as f:
                card_data = json.load(f)
                report.update(card_data)
        except Exception as e:
            logger.warning(f"Error reading model card: {e}")

    if cluster_path.exists():
        try:
            with open(cluster_path, "r", encoding="utf-8") as f:
                report["clusters"] = json.load(f)
        except Exception as e:
            logger.warning(f"Error reading clusters.json: {e}")

    return report


@router.get("/events/replay")
def get_replay_events() -> List[Dict[str, Any]]:
    """Retrieve held-out test landslide events for interactive UI replay."""
    events_path = PROCESSED_DATA_DIR / "held_out_events.json"
    if events_path.exists():
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading replay events: {e}")

    # Fallback known historical events across South Asia
    return [
        {"event_id": "glc_kedarnath_01", "date": "2013-06-16", "latitude": 30.7346, "longitude": 79.0669, "region_group": "South Asia", "description": "Kedarnath Himalayan Debris Flow Event"},
        {"event_id": "glc_wayanad_02", "date": "2024-07-30", "latitude": 11.5361, "longitude": 76.1260, "region_group": "South Asia", "description": "Western Ghats Monsoon Landslide"},
        {"event_id": "glc_malin_03", "date": "2014-07-30", "latitude": 19.1600, "longitude": 73.6800, "region_group": "South Asia", "description": "Malin Maharashtra Slope Detachment"}
    ]


@router.get("/datasets")
def get_dataset_manifest() -> Dict[str, Any]:
    """Retrieve datasets/DATASET_MANIFEST.json contents."""
    manifest_path = DATASETS_DIR / "DATASET_MANIFEST.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading manifest: {e}")

    return {"sources": [], "message": "Manifest not yet populated"}
