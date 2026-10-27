# ADR 0003: Label Sources and Pseudo-Absence Sampling Strategy

- **Status**: Accepted
- **Date**: 2026-10-27
- **Author**: URsafe Autonomous Engineering Team

## Context
Landslide repositories only record positive failure events. To formulate a supervised binary classification and risk scoring task, representative negative samples ("pseudo-absences") must be synthesized from real physical geography. If negatives are sampled naively (e.g. over oceans or right next to an active debris flow), the resulting model will either learn trivial land-vs-water discriminators or suffer label noise from false negatives.

## Decision
1. **Primary Positive Source**: NASA COOLR / Global Landslide Catalog (GLC). Only include records from 2007-01-01 onward with verified geographic coordinates and spatial uncertainty radius $\le 25\text{ km}$ where recorded.
2. **Pseudo-Absence Ratio**: Generate 2 negatives for every 1 positive sample (ratio 1:2).
3. **Spatial Exclusion Buffer**: Every negative candidate must be located on valid continental landmasses within the region, at least 5.0 km geodesic distance away from all known landslide event locations in the catalog.
4. **Temporal Seasonality Matching**: Dates for pseudo-absences are sampled uniformly from the empirical month distribution of the positive events. This guarantees that negative samples reflect realistic seasonal conditions (such as monsoon months) rather than learning simple winter-vs-monsoon month shortcuts.
5. **Reproducibility**: Negatives are generated using a fixed random seed (`seed=42`).

## Alternatives Considered
- **Equal 1:1 Sampling**: Rejected. Landslides are geographically rare events; a 1:2 ratio better reflects the natural spatial sparsity while maintaining balanced gradient signals for tree and linear classifiers.
- **Random Uniform Temporal Sampling**: Rejected. Would bias negatives towards dry months, causing models to predict landslides simply based on calendar date rather than true antecedent precipitation and soil saturation.

## Consequences
- The resulting predictions represent relative susceptibility under realistic seasonal conditions.
- Documented in product disclaimers as a relative score rather than a probability.
