# ADR 0012: System Limitations, Known Weaknesses, and Safety Disclaimers

- **Status**: Accepted
- **Date**: 2026-10-27
- **Author**: URsafe Autonomous Engineering Team

## Context
Predicting geohazards is life-critical. Machine learning models trained on open-source satellite and reanalysis data possess inherent limitations that must be transparently surfaced to prevent user over-reliance.

## Decision
1. **Catalog Bias & Reporting Artifacts**:
   - The NASA COOLR / GLC catalog reflects reporting bias: landslides blocking highways or occurring near towns are reported promptly; deep backcountry failures often go unrecorded.
   - Negative samples are pseudo-absences on land, not guaranteed stable slopes across all geologic history.
2. **Resolution Limits**:
   - DEM resolution is ~90 meters (Copernicus GLO-90). Road cuts, micro-gullies, and roadside cliffs narrower than 90m are smoothed, making DEM-derived slope a conservative lower bound.
   - Weather reanalysis (ERA5) has a spatial resolution of ~25–30 km. Localized cloudbursts (e.g. violent convective cells in a single Himalayan valley) may not be fully resolved.
3. **Product Honesty & Disclaimers**:
   - The system displays explicit disclaimers across all endpoints and UI views:
     *"Model-based relative estimates only. Not a substitute for official warnings or local judgment."*
   - Official national meteorological and disaster agency links are prominently featured:
     - Geological Survey of India (GSI): [https://www.gsi.gov.in](https://www.gsi.gov.in)
     - National Disaster Management Authority (NDMA): [https://ndma.gov.in](https://ndma.gov.in)
     - India Meteorological Department (IMD): [https://mausam.imd.gov.in](https://mausam.imd.gov.in)
4. **Known Weaknesses in Model Card**:
   - The machine-readable `models/model_card.json` and UI Model Card explicitly enumerate:
     - Under-prediction of localized cloudburst triggers outside the 25km ERA5 grid.
     - Conservatism in micro-slope steepness due to 90m smoothing.
     - Inability to account for real-time roadside construction blasting or artificial slope destabilization without geotechnical sensors.

## Consequences
- Full compliance with Hard Rule 6 (Honesty in the product); shields users from misleading guarantees.
