# ADR 0009: Explainability, Plain-Language Mapping, and AI Companion Guidance

- **Status**: Accepted
- **Date**: 2026-10-27
- **Author**: URsafe Autonomous Engineering Team

## Context
Raw machine learning predictions (e.g. `score=78`) fail to provide travelers with actionable context. Non-expert users need to understand *why* a particular mountain section is hazardous and *what* physical drivers contributed to the warning. Section 5.7 and 5.8 specify using SHAP (SHapley Additive exPlanations) TreeExplainer, fallback to permutation importance, plain-language translation of top factors, traveler profile multipliers, and a shared AI Companion engine.

## Decision
1. **Explainability Engine (`ml/explain.py`)**:
   - Primary: `shap.TreeExplainer` on the fitted tree ensemble to calculate exact additive feature contributions for every query point.
   - Fallback: Scikit-learn permutation importance or normalized tree feature importances if SHAP runtime exceeds budget.
2. **Feature-to-Language Translation (`ml/feature_text.py`)**:
   - Map technical features into human-readable phrases:
     - `slope_deg` $\to$ "Steep slope (>25°)" or "Gentle terrain"
     - `rain_3d` $\to$ "Heavy 3-day rainfall accumulation"
     - `soil_wetness_top` $\to$ "High surface soil water saturation"
     - `roughness` $\to$ "Rugged, dissected mountain topography"
     - `distance_to_water_m` $\to$ "Proximity to major river channel"
3. **AI Companion Engine (`ml/companion.py`)**:
   - Function: `ai_companion_guide(terrain_sample)`.
   - Incorporates Traveler Profile multipliers before banding:
     - **Hiker / Solo Backpacker**: $1.0\times$ (baseline physical agility)
     - **Rider / Motorcyclist**: $1.0\times$ (road traction sensitivity)
     - **Family**: $1.2\times$ (higher vulnerability, slower evacuation)
     - **Elderly / High-Vulnerability**: $1.25\times$ (conservative safety margin)
   - Evaluates data completeness and distance from training distribution to assign Confidence (`High`, `Medium`, `Low`).
   - Produces structured output: Risk Score, Alert Band, Confidence, Terrain Zone, Top Factors with provenance tags (`observed`, `forecast`, `derived`, `missing`), and actionable Plain-Language Advice.
   - Shared verbatim between backend API endpoints and Jupyter practicals.

## Consequences
- Transforms raw numerical models into an empathetic, transparent, and actionable travel assistant.
