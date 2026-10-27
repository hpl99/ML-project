# URsafe Pipeline Execution Run Log

This is an append-only audit trail updated automatically at each pipeline invocation (`python -m ml.run_pipeline`).

| Timestamp (UTC) | Git Commit | Command | Positives | Negatives | Total Rows | Features Selected | Final Model | ROC-AUC | PR-AUC | Warnings / Notes |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 2026-10-27 06:59:31 | `4509a9b` | `python -m ml.run_pipeline` | 200 | 400 | 600 | 21 features | URsafe Terrain Hazard Model (TunedRandomForest) | 0.9115 | 0.6705 | Completed in 7.6s. Optimal K=5. Orange band threshold=28. |
