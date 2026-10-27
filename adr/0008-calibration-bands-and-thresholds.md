# ADR 0008: Probability Calibration, Score Scaling, and Risk Bands

- **Status**: Accepted
- **Date**: 2026-10-27
- **Author**: URsafe Autonomous Engineering Team

## Context
Raw predicted probabilities from complex ensemble trees often cluster around 0.3-0.7 and exhibit miscalibration under class imbalance or pseudo-absence sampling. Furthermore, travelers require clear action boundaries rather than uncalibrated numbers. Section 5.6 mandates calibrating the model with isotonic or sigmoid regression, reporting Brier score and calibration curves, and selecting the Orange threshold to achieve $\ge 0.80$ recall on the spatial hold-out if achievable.

## Decision
1. **Calibration**:
   - Fit a `CalibratedClassifierCV` (isotonic regression if sample size permits, or sigmoid/Platt scaling) on the validation partition.
   - Evaluate reliability diagrams and Brier score before and after calibration.
2. **Score Transformation**:
   - Map calibrated posterior $P$ to a 0–100 integer score: $\text{Risk Score} = \text{round}(P \cdot 100)$.
3. **Operational Risk Bands**:
   - **Green (Low Risk, 0 to 25)**: Standard terrain safety; routine caution.
   - **Yellow (Moderate Risk, 26 to 55)**: Rolling terrain or light rainfall; monitor localized conditions.
   - **Orange (High Risk, 56 to 80)**: Elevated vulnerability; operational threshold calibrated so validation recall $\ge 0.80$; recommend alternative corridors.
   - **Red (Severe Hazard, 81 to 100)**: Active extreme combination of steep slopes and intense antecedent/recent precipitation; travel strictly discouraged.
4. **Model Card Integration**:
   - Thresholds, recall rates, false alarm rates, and Brier scores are recorded in `models/model_card.json` and rendered in the UI Model Card.

## Consequences
- Provides mathematically justified alert bands that prioritize traveler safety by capturing at least 80% of true events in the upper warning tiers.
