# Dataset Registry

This file documents all data sources ingested into URsafe, including provenance, access dates, record counts, and inherent sampling biases.

## Registered Datasets

### 1. NASA COOLR / Global Landslide Catalog (GLC)
- **Source**: NASA Earthdata ArcGIS FeatureServer / Cooperative Open Online Landslide Repository (COOLR)
- **Primary URL**: `https://gis.earthdata.nasa.gov/gis05/rest/services/Landslides/COOLR_Events_Points/FeatureServer/0`
- **Fallback URL**: `https://maps.nccs.nasa.gov/mapping/rest/services/COOLR/COOLR_Events_Point/FeatureServer` & GPM Open Data
- **License**: NASA Open Data Policy (Public Domain / Free for all use)
- **Temporal Coverage**: 2007 to present
- **Spatial Coverage**: Global (filtered to target region, e.g. South Asia: lon 60 to 100, lat 5 to 38, expanded as per volume rules)
- **Key Fields Ingested**: `event_id`, `event_date`, `latitude`, `longitude`, `location_accuracy`, `landslide_category`, `trigger`
- **Bias Profile**: Catalog is biased toward populated corridors, transportation routes (highways, railways), and events with news/media coverage. Remote uninhabited mountain slopes with unobserved landslides are under-represented.

### 2. Open-Meteo Copernicus DEM 90m Elevation Service
- **Source**: Copernicus DEM (GLO-90) via Open-Meteo Elevation API
- **URL**: `https://api.open-meteo.com/v1/elevation`
- **License**: Open Data / Attribution (CC BY 4.0)
- **Resolution**: ~90 meters
- **Sampling Pattern**: 5x5 spatial grid at 0.001° spacing (approx. 100m grid) centered on sample coordinates.
- **Derived Metrics**: Center elevation, slope (Horn 3x3), aspect (sin/cos), curvature (Laplacian), roughness (standard deviation of elevation), local relief (max - min).

### 3. Open-Meteo Historical Weather (ERA5 / ERA5-Land Reanalysis)
- **Source**: ECMWF ERA5 & ERA5-Land via Open-Meteo Archive API
- **URL**: `https://archive-api.open-meteo.com/v1/archive`
- **License**: Copernicus Climate Change Service (Open Data)
- **Parameters**: Daily `precipitation_sum` (mm, UTC)
- **Windows Computed**: 1-day, 3-day, 7-day, 14-day, 30-day cumulative rainfall; 30-day maximum 1-day rain; 30-day Antecedent Precipitation Index (API, decay 0.85).

### 4. NASA POWER Agroclimatology API
- **Source**: NASA Prediction Of Worldwide Energy Resources (POWER)
- **URL**: `https://power.larc.nasa.gov/api/temporal/daily/point`
- **License**: NASA Open Data Policy (Public Domain)
- **Parameters**: `GWETTOP` (surface soil wetness 0-5cm), `GWETROOT` (root zone soil wetness 0-100cm)
- **Lag & Fallback**: NASA POWER daily data exhibits a 3-5 day lag. For live real-time predictions, latest available observation is fetched with date flagged. Fallback is the 30-day Antecedent Precipitation Index (API).

### 5. Natural Earth Physical (10m Water Features)
- **Source**: Natural Earth public domain map dataset
- **URL**: `https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/physical/ne_10m_rivers_lake_centerlines.zip`
- **License**: Public Domain
- **Usage**: Calculation of geodesic distance to nearest major water body via haversine BallTree indexing.
- **Bias / Limitation**: Captures major rivers and permanent lakes only; intermittent monsoon streams and micro-gullies are not captured at 1:10m scale.

### 6. Pseudo-Absence Generation (Negatives)
- **Source**: Algorithmic generation on validated landmasses within target bounding box.
- **Constraint**: Minimum 5.0 km geodesic buffer from any recorded landslide event; date drawn uniformly from month distribution of positive events.
- **Ratio**: 1:2 (1 positive to 2 negatives).
