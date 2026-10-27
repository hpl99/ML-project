# ADR 0005: Feature Selection, Physics Checks and Collinearity Thresholds

- **Status**: Accepted
- **Date**: 2026-10-27
- **Author**: URsafe Autonomous Engineering Team

## Context
High-dimensional terrain and rainfall modeling can easily introduce multi-collinear predictors and unphysical artifact values. Section 4 mandates physical bounds checks (slope 0-90°, NDVI -1 to 1, non-negative rainfall), removal of features with $>40\%$ missingness, and pruning pairwise correlated features with Pearson $|r| > 0.95$.

## Decision
1. **Physics Validation Layer**:
   - `slope_deg`: bounded $[0.0, 90.0]$. Out-of-bounds flagged as NaN.
   - `rain_*`: bounded $[0.0, \infty)$. Any negative value replaced with NaN.
   - `soil_wetness_*`: bounded $[0.0, 1.0]$.
   - `elevation_m`: bounded $[-450.0, 9000.0]$.
2. **Terrain Engineering**:
   - Slope computed via Horn's finite-difference method on the 3x3 DEM window.
   - Aspect decomposed into circular components $\sin(\text{aspect})$ and $\cos(\text{aspect})$ to eliminate the $0^\circ / 360^\circ$ north boundary artifact.
   - Micro-roughness defined as the standard deviation of elevation across the 5x5 sub-grid; local relief defined as $\max(Z) - \min(Z)$ over the 5x5 window.
3. **Hydrological Windows**:
   - 1d, 3d, 7d, 14d, 30d cumulative precipitation, 30d max daily rain, and 30d Antecedent Precipitation Index ($API = \sum_{k=1}^{30} P_{t-k} \cdot 0.85^k$).
   - Pruning: intermediate cumulative metrics (e.g. 21d) exhibiting $r > 0.95$ with 14d or 30d were discarded.
4. **Soil & Vegetation**:
   - SoilGrids is disabled by default due to high latency and $>40\%$ missing values in border mountainous regions.
   - Soil wetness utilizes NASA POWER topsoil (GWETTOP) and 7-day root zone (GWETROOT).

## Consequences
- Clean, physically interpretable feature matrices without numerical leakage or extreme collinearity.
- Recorded completely in `adr/FEATURES.md`.
