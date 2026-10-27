"""Navigation Route Comparison Endpoint: /api/route.

Compares up to 3 alternative driving routes via OSRM, evaluates terrain hazard risk
along sampled waypoints, and returns color-coded segments for Fast, Balanced, and Safer paths (ADR 0010).
"""
import logging
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.routing import compare_routes

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Routing"])


class RouteRequest(BaseModel):
    start: List[float] = Field(..., min_length=2, max_length=2, description="[latitude, longitude] of origin")
    end: List[float] = Field(..., min_length=2, max_length=2, description="[latitude, longitude] of destination")
    profile: str = Field("hiker", description="Traveler persona: hiker | rider | family | elderly")


@router.post("/route")
def calculate_hazard_routes(req: RouteRequest) -> Dict[str, Any]:
    """Calculate and compare hazard-aware navigation corridors."""
    try:
        start_lat, start_lon = req.start[0], req.start[1]
        end_lat, end_lon = req.end[0], req.end[1]

        results = compare_routes(
            start_lat=start_lat,
            start_lon=start_lon,
            end_lat=end_lat,
            end_lon=end_lon,
            profile=req.profile
        )
        return results
    except Exception as e:
        logger.exception("Route comparison endpoint encountered failure")
        raise HTTPException(status_code=500, detail=str(e))
