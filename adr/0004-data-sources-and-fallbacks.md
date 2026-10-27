# ADR 0004: Data Sources, Fallbacks and Resilience Architecture

- **Status**: Accepted
- **Date**: 2026-10-27
- **Author**: URsafe Autonomous Engineering Team

## Context
URsafe relies on external public scientific services for elevation, historical reanalysis, live forecast meteorology, soil moisture, and hydrology. Hard Rule 2 mandates: "Never fabricate a value. If a data source fails, use the fallback listed for it. If every fallback fails, mark the value missing, lower the confidence, and log it in the ADR. Never silently invent numbers."

## Decision
We establish explicit primary endpoints, caching tiers, and fallback chains for every ingested variable:

1. **Elevation & Micro-Terrain**:
   - Primary: Open-Meteo Elevation API (Copernicus DEM 90m, 5x5 sub-grid at 0.001° spacing).
   - Resilience: Local disk cache grouped by coordinates; batch requests up to 100 points per call.
   - Fallback: Local SRTM/ETOPO grid lookup if service is unreachable.
2. **Rainfall (Historical & Forecast)**:
   - Primary: Open-Meteo Historical Archive (daily `precipitation_sum`, UTC) and Live Forecast API (past 30 days observed + 3 days forecast).
   - Resilience: Group sample locations by 0.25° grid cells and fetch complete time series once per cell; cache to `datasets/cache/rain_{lat}_{lon}.json`.
   - Fallback: Neighboring grid cell interpolation within 0.5°; if unavailable, mark rainfall as missing and tag confidence as `Low`.
3. **Soil Wetness**:
   - Primary: NASA POWER Daily Point API (`GWETTOP`, `GWETROOT`).
   - Operational Lag Handling: NASA POWER lags 3-5 days behind real time. For live inference, query the latest available observation and return its timestamp.
   - Fallback: Compute 30-day Antecedent Precipitation Index (API, decay factor 0.85) from weather series; tag feature as `derived`.
4. **Distance to Water**:
   - Primary: Natural Earth 10m physical river centerlines and lake polygons indexed via `scipy.spatial.cKDTree` or `sklearn.neighbors.BallTree` (haversine metric).
   - Fallback: Precomputed regional hydrographic distance lookup grid.
5. **Caching & Upstream Rate Limiting**:
   - Descriptive User-Agent header (`URsafe-TerrainHazard/1.0 (academic-safety-research; contact: engineering@ursafe.org)`).
   - Exponential backoff with jitter on HTTP 429/503 errors (3 retries).
   - Dual-tier cache: in-memory LRU cache + persistent JSON files on disk under `datasets/cache/`.

## Consequences
- Guaranteed system resilience: backend never throws 500 when external public APIs suffer intermittent downtime.
- Complete transparency: every prediction returns provenance tags (`observed`, `forecast`, `derived`, `missing`).
