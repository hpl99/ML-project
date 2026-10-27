# Feature Registry & Engineering Rationale

This registry catalogs every candidate feature considered, its physical definition, source, missingness, correlation, and whether it was selected or rejected for model training.

## Candidate Features Table

| Feature Name | Definition | Units | Source | Time Window | Missing Rate | Label Corr (r) | Included? | Rationale |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `elevation_m` | Center ground elevation above sea level | meters | Open-Meteo DEM 90m | Static | 0.0% | Moderate (+) | YES | High elevations and mountainous relief provide the gravitational potential for slope failures. |
| `slope_deg` | Maximum rate of change in elevation across central 3x3 grid (Horn's method) | degrees (0-90) | Open-Meteo DEM 90m | Static | 0.0% | Strong (+) | YES | Primary mechanical trigger: shear stress increases with slope. *Note: 90m DEM smooths micro-slopes; acts as conservative lower bound.* |
| `aspect_sin` | Sine of terrain orientation angle (East-West component) | dimensionless (-1 to 1) | Open-Meteo DEM 90m | Static | 0.0% | Low | YES | Captures cyclical aspect without boundary discontinuity at 0°/360°. Influences monsoon solar exposure & windward moisture. |
| `aspect_cos` | Cosine of terrain orientation angle (North-South component) | dimensionless (-1 to 1) | Open-Meteo DEM 90m | Static | 0.0% | Low | YES | Orthogonal cyclical aspect component (North vs. South face insolation and freeze-thaw weathering). |
| `curvature` | Profile & planform rate of slope change (discrete 2D Laplacian over 3x3) | 1/m * 100 | Open-Meteo DEM 90m | Static | 0.0% | Low-Moderate | YES | Concave slopes focus surface hydrology and colluvial sediment accumulation. |
| `roughness` | Standard deviation of elevation in 5x5 sub-grid (~500m window) | meters | Open-Meteo DEM 90m | Static | 0.0% | Strong (+) | YES | High topographic ruggedness indicates active geomorphic erosion and fractured bedrock. |
| `relief_m` | Difference between maximum and minimum elevation in 5x5 sub-grid | meters | Open-Meteo DEM 90m | Static | 0.0% | Strong (+) | YES | Direct proxy for total local gravitational head and canyon/valley incision depth. |
| `rain_1d` | Total precipitation on event date (or prior 24h) | mm | Open-Meteo Archive / Forecast | Day t | <2.0% | Moderate (+) | YES | Immediate hydraulic trigger initiating rapid pore-water pressure spike. |
| `rain_3d` | Cumulative precipitation over 3 days (t-2 to t) | mm | Open-Meteo Archive / Forecast | 3 days | <2.0% | Strong (+) | YES | Critical window for shallow translational slides and debris flows. |
| `rain_7d` | Cumulative precipitation over 7 days (t-6 to t) | mm | Open-Meteo Archive / Forecast | 7 days | <2.0% | Strong (+) | YES | Intermediate saturation accumulation window. |
| `rain_14d` | Cumulative precipitation over 14 days (t-13 to t) | mm | Open-Meteo Archive / Forecast | 14 days | <2.0% | Strong (+) | YES | Sustained wet spell leading to regolith weakening. |
| `rain_30d` | Cumulative precipitation over 30 days (t-29 to t) | mm | Open-Meteo Archive / Forecast | 30 days | <2.0% | Moderate-Strong | YES | Deep infiltration and base water table elevation over prolonged monsoon episodes. |
| `rain_max_day_30d` | Single highest daily rainfall event observed within 30 days preceding event | mm | Open-Meteo Archive / Forecast | 30 days | <2.0% | Strong (+) | YES | Identifies extreme storm bursts that destabilized slopes even if surrounding days were dry. |
| `api_30d` | Antecedent Precipitation Index: $\sum_{i=1}^{30} P_{t-i} \cdot (0.85)^i$ | mm equivalent | Open-Meteo Archive / Forecast | 30 days | <2.0% | Strong (+) | YES | Hydrological memory metric tracking decaying ground moisture without requiring soil sensors. |
| `soil_wetness_top` | Fractional volumetric soil wetness at surface layer (0-5 cm) | fraction (0-1) | NASA POWER (GWETTOP) | Day t | ~5% (imputed) | Moderate (+) | YES | Surface saturation leading to liquefaction and mudslide initiation. |
| `soil_wetness_root_7d`| 7-day rolling mean root zone soil wetness (0-100 cm) | fraction (0-1) | NASA POWER (GWETROOT) | 7 days | ~5% (imputed) | Moderate (+) | YES | Deep root-zone saturation causing slip plane detachment. |
| `distance_to_water_m` | Geodesic distance to nearest major river centerline or lake boundary | meters | Natural Earth 10m Physical | Static | 0.0% | Moderate (-) | YES | River toe-cutting undermines slope bases, increasing instability nearby. |
| `log_dist_to_water` | Natural log of $(1 + \text{distance\_to\_water\_m})$ | log(m) | Natural Earth 10m Physical | Static | 0.0% | Moderate (-) | YES | Compresses long tail of continental distances into linear feature representation. |
| `month_sin` | Sine of event month in annual cycle: $\sin(2\pi \cdot \text{month} / 12)$ | dimensionless (-1 to 1) | Event Date | Monthly | 0.0% | Moderate | YES | Captures cyclical seasonal monsoon rhythm (June-September peak in South Asia). |
| `month_cos` | Cosine of event month in annual cycle: $\cos(2\pi \cdot \text{month} / 12)$ | dimensionless (-1 to 1) | Event Date | Monthly | 0.0% | Moderate | YES | Orthogonal cyclical seasonal component (winter vs. summer). |
| `ndvi` | Normalized Difference Vegetation Index (climatology) | index (-1 to 1) | NASA NEO / Fallback | Monthly | Best-effort | Low-Moderate | Included if available | Root tensile strength provides mechanical soil reinforcement. |
| `region_group` | Categorical geographic cluster (e.g. Western Ghats, Himalayas, etc.) | category | Spatial coordinates | Static | 0.0% | Categorical | YES (OHE) | Allows model to separate distinct climatological and tectonic regimes. |

## Rejected Candidate Features

| Feature Name | Proposed Definition | Reason for Rejection |
|:---|:---|:---|
| `raw_aspect_deg` | Raw angle 0° to 360° | Circular discontinuity: 359° and 1° are adjacent north-facing slopes but have maximal arithmetic difference (358°). Replaced by `aspect_sin` and `aspect_cos`. |
| `raw_month_int` | Integer 1 to 12 | Discontinuity between December (12) and January (1). Replaced by `month_sin` and `month_cos`. |
| `rain_21d` | Cumulative 21-day rainfall | Collinearity: Pearson correlation $r > 0.96$ with `rain_14d` and `rain_30d`. Excluded to prevent multi-collinearity. |
| `soil_texture_sand_pct` | SoilGrids Sand fraction | Missing rate exceeded 40% across remote Himalayan border sectors due to SoilGrids coverage limits; queries heavily throttled. Excluded by default per Section 4 rule. |
| `instantaneous_wind_speed` | Surface wind speed | Correlation with landslide occurrence was negligible ($|r| < 0.03$), physically secondary compared to rainfall and slope. |
