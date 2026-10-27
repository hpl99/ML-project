# ADR 0002: Region and Volume Scope Expansion Strategy

- **Status**: Accepted
- **Date**: 2026-10-27
- **Author**: URsafe Autonomous Engineering Team

## Context
Section 3.1 mandates an empirical volume rule: aim for at least 2,000 real positive landslide events for reliable statistical learning, while capping at 6,000 positive events to maintain reasonable pipeline runtimes. The primary geographic focus is South Asia (longitude 60°E to 100°E, latitude 5°N to 38°N), encompassing high-vulnerability mountain ranges: the Himalayas, Western Ghats, Eastern Ghats, and the Hindu Kush.

If the regional query yields fewer than 2,000 validated positive events with 2007+ dates and adequate spatial accuracy, the ingestion pipeline must deterministically expand its geographic envelope.

## Decision
1. **Initial Filter**: South Asia bounding box (`min_lon=60.0`, `max_lon=100.0`, `min_lat=5.0`, `max_lat=38.0`).
2. **Expansion Step 1**: If valid positives $< 2,000$, expand to Pan-Asia (`min_lon=60.0`, `max_lon=150.0`, `min_lat=-10.0`, `max_lat=55.0`), encompassing Southeast Asia, Japan, and Central Asia.
3. **Expansion Step 2**: If still $< 2,000$, expand to Global coverage, capping at 6,000 positives.
4. **Regime Separation**: When expanding beyond South Asia, add a categorical feature `region_group` (e.g. `South Asia`, `East/Southeast Asia`, `Americas`, `Europe`, `Other`) to allow tree ensembling and linear baselines to separate distinct precipitation and geologic regimes.

## Alternatives Considered
- **Strictly Enforcing South Asia regardless of count**: Rejected. If only 400 events are retrieved, spatial cross-validation would suffer high variance and overfitting.
- **Synthesizing positive records**: Forbidden by Hard Rule 1 ("REAL DATA ONLY").

## Consequences
- Guarantees sufficient statistical power while preserving geographic representativeness.
- Any geographic expansion step taken is automatically logged to `adr/RUN_LOG.md` and `adr/DATASETS.md`.
