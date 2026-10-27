# ADR 0006: Spatial Block Splitting and Spatial Autocorrelation Control

- **Status**: Accepted
- **Date**: 2026-10-27
- **Author**: URsafe Autonomous Engineering Team

## Context
Tobler's First Law of Geography states: "everything is related to everything else, but near things are more related than distant things." Standard random train/test splitting (e.g. `train_test_split(shuffle=True)`) randomly assigns adjacent points within the same valley or mountain ridge into both train and test partitions. Because topographic elevation and rainfall events have high spatial autocorrelation over 10-50 km distances, random splitting leaks ground-truth information, producing artificially inflated test scores ("leakage gap") that fail when the model is evaluated in a new mountain valley.

## Decision
1. **Primary Evaluation Strategy**: Spatial Block Splitting using 1.0° geographic grid blocks (approx. $110\text{ km} \times 110\text{ km}$).
   - Every sample point is assigned a unique spatial block ID: `block_id = f"{int(np.floor(lat))}_{int(np.floor(lon))}"`.
   - `GroupShuffleSplit` (80% train, 20% test, `random_state=42`) is performed on `block_id`. No spatial block is ever shared between train and test sets.
2. **Leakage Demonstration**: For academic viva and rigorous comparison, we also evaluate a standard random split once to explicitly report the leakage gap in the model report card.
3. **Cross-Validation**: 5-fold `GroupKFold` on spatial blocks is used for all hyperparameter tuning.

## Consequences
- Metrics reported on the hold-out test set reflect genuine generalization to unseen terrain and unseen weather fronts.
- Prevents over-optimistic risk claims for unexplored navigation routes.
