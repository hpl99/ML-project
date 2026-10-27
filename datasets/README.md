# URsafe Datasets Guide

## Overview
All data ingested by URsafe is 100% REAL, publicly accessible, and requires zero paid API tokens.

### Datasets Summary
1. **NASA COOLR / Global Landslide Catalog (GLC)**:
   - Primary source for real historical landslide events.
   - Endpoint: NASA Earthdata ArcGIS FeatureServer / CSV exports.
   - License: NASA Open Data Policy (Public Domain).
2. **Open-Meteo Copernicus DEM 90m Elevation API**:
   - High-resolution topography, slope, aspect, curvature, roughness, relief.
   - License: Attribution (CC BY 4.0 / Open Data).
3. **Open-Meteo Historical Weather & Forecast API**:
   - ERA5 & ERA5-Land reanalysis daily precipitation series (2007-present), and live 30-day past + 3-day forecast.
   - License: Attribution (CC BY 4.0 / Open Data).
4. **NASA POWER Agroclimatology (daily point API)**:
   - Soil wetness (GWETTOP surface soil wetness, GWETROOT root zone soil wetness).
   - License: NASA Open Data Policy (Public Domain).
5. **Natural Earth Physical (10m Rivers & Lakes)**:
   - Major hydrographic centerlines for distance-to-water calculations.
   - License: Public Domain.

## How to Re-download or Refresh
To execute a clean fetch and processing cycle:
```bash
python -m ml.run_pipeline --refresh-data
```
Or with sample limit for smoke testing:
```bash
python -m ml.run_pipeline --limit 200
```
Raw files reside in `datasets/raw/`. Interim tables are stored in `datasets/interim/`. Processed training sets reside in `datasets/processed/`.
Cached API cell responses reside in `datasets/cache/` to ensure full determinism and avoid rate limits.
