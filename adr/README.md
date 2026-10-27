# URsafe Architecture Decision Records (ADR) Index

This directory serves as the **Single Source of Truth** for all engineering and scientific decisions made in the URsafe hazard prediction system. These documents are updated programmatically by the pipeline and manual engineering reviews.

## ADR Index

| ID | Title | Status | Scope |
|:---|:---|:---|:---|
| [0001](0001-problem-framing.md) | Problem Framing & Relative Risk vs. Probability | Accepted | Problem Formulation |
| [0002](0002-region-and-volume-scope.md) | Region & Volume Scope Expansion Strategy | Accepted | Data Collection |
| [0003](0003-label-source-and-negative-sampling.md) | Label Sources & Pseudo-Absence Sampling | Accepted | Labels & Sampling |
| [0004](0004-data-sources-and-fallbacks.md) | Data Sources, Resilience & Upstream Fallbacks | Accepted | Features & Data Pipeline |
| [0005](0005-feature-selection.md) | Feature Selection, Physics Checks & Collinearity | Accepted | Feature Engineering |
| [0006](0006-splitting-and-leakage-control.md) | Spatial Block Splitting & Leakage Prevention | Accepted | Validation Strategy |
| [0007](0007-model-selection-and-tuning.md) | Model Comparison, CV & Hyperparameter Tuning | Accepted | Modeling |
| [0008](0008-calibration-bands-and-thresholds.md) | Probability Calibration, Bands & Alert Thresholds | Accepted | Calibration & Risk Bands |
| [0009](0009-explainability-and-companion-rules.md) | SHAP Explainability & AI Companion Logic | Accepted | Companion & Reasoning |
| [0010](0010-routing-approach.md) | Multi-Criteria Route Optimization & Risk Aggregation | Accepted | Navigation Engine |
| [0011](0011-ui-design-decisions.md) | Editorial Cinematic UI & Accessibility Decisions | Accepted | Frontend Design System |
| [0012](0012-limitations-and-safety.md) | Limitations, Failure Modes & Safety Disclaimers | Accepted | Safety & Product Honesty |

## Additional Registries
- [DATASETS.md](DATASETS.md): Complete catalog of datasets, URLs, licenses, row counts, and bias profiles.
- [FEATURES.md](FEATURES.md): Comprehensive feature registry documenting definitions, physical units, missing rates, and retention/rejection rationale.
- [RUN_LOG.md](RUN_LOG.md): Append-only run execution audit log recording execution time, commit hashes, data counts, and performance metrics.
- [generated/](generated/): Machine-readable pipeline artifacts (`dataset_stats.json`, `feature_manifest.json`, `metrics_snapshot.json`).
