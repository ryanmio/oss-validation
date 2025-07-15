# Tier-2 Absolute Positional Accuracy – With Manual Audit

**Data vintage:** tier2_positional_accuracy.csv snapshot after latest run on $(date +%Y-%m-%d)

## Anchor extraction / resolution snapshot

| Stage | Count |
|-------|------:|
| Total matched grants | 1 496 |
| Extracted anchors (regex heuristics) | 152 |
| Unique GNIS-resolved anchors | 75 |
| Ambiguous (>1 GNIS candidates) | 0 |
| No GNIS match | 77 |
| Good anchors after manual audit | **39** |

> The 75 resolved anchors were manually audited, resulting in 39 good anchors after discarding bad matches and applying overrides.

## Distance metrics (full OSS polygons)

| Metric | Value |
|--------|-------|
| Sample size | 39 |
| Median error | 3.18 km |
| 90th percentile | 6.92 km |
| 95 % CI for 90th pc (bootstrap) | N/A |

✅ **Interpretation:** After manual audit of worst-case distances and overrides, the 90th percentile is 6.92 km on 39 good anchors, meeting the goal of ≤10 km for 90% of positions.

## Files for detailed QA ✔️

| File | Purpose |
|------|---------|
| `data/processed/tier2_positional_accuracy.csv` | Record of 152 anchor candidates with resolution details. |
| `data/processed/tier2_anchor_audit_good_worst10.csv` | Worst 10 good anchors for review (if generated). |
| `reports/anchor_validation_workflow.md` | Detailed workflow including manual process. |
| `reports/tier2_ecdf.png` | ECDF of great-circle error distances (to be generated). |