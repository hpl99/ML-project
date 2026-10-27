# ADR 0001: Problem Framing — Relative Risk Score vs. Absolute Probability

- **Status**: Accepted
- **Date**: 2026-10-27
- **Author**: URsafe Autonomous Engineering Team

## Context
Predicting landslides requires determining the likelihood of slope failure given static terrain morphology and dynamic hydrometeorological triggers. However, global and regional landslide inventories (such as NASA COOLR and GLC) only record observed positive events, typically reported by emergency services, transport authorities, or media outlets. They do not record true non-events ("safe days" on every hill).

Furthermore, generating pseudo-absences (negative samples) relies on heuristic spatial-temporal sampling rather than confirmed absence of slope instability. Under these conditions, standard supervised machine learning models do not estimate true Bayesian probability $P(Y=1|X)$, but rather relative susceptibility or risk conditioned on the pseudo-absence sampling frequency.

## Decision
We formulate the model output strictly as a **Relative Hazard Risk Score** scaled from 0 to 100, rather than an absolute event probability.
1. The user interface, API schemas, and documentation will never use the word "probability of landslide".
2. Outputs are categorized into four intuitive action bands: Green (Low Risk, 0-25), Yellow (Moderate Risk, 26-55), Orange (High Risk, 56-80), and Red (Severe Hazard, 81-100).
3. Every prediction payload returns explicit provenance tags (`observed`, `forecast`, `derived`, `missing`) and confidence metrics (`High`, `Medium`, `Low`).

## Alternatives Considered
- **Binary Classification Probability Output ($0.0 - 1.0$)**: Rejected. Misleads travelers into interpreting e.g. "0.08" as negligible risk, when in reality an 8% prior in a 1:2 sampled dataset signifies severe vulnerability.
- **Physical Factor of Safety (Limit Equilibrium Modelling)**: Rejected. Requires geotechnical borehole parameters (cohesion, internal friction angle, regolith depth) which do not exist continuously across South Asia at regional scale.

## Consequences
- **Positive**: Complete scientific and product honesty; prevents false sense of security; transparent communication to users and authorities.
- **Trade-offs**: Requires explicit calibration and band threshold selection to align operational alert levels with public safety needs.
