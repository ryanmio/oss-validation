# Spatial Validation Summary

Total matched grants validated: 1496

## Classification Results

- **Exact (historical)**: 1317 ( 88.0%) - Point in county valid for grant year
- **Boundary tol**:        112 (  7.5%) - ≤50 m from polygon edge
- **Near split**:            6 (  0.4%) - In county swapped within ±2 yr
- **Exact (modern)**:       11 (  0.7%) - Matches modern county (no split)
- **Adjacent (modern)**:    46 (  3.1%) - In modern neighbour (likely split)
- **Mismatch**:              4 (  0.3%) - No reasonable match
- **No county**:             0 (  0.0%) - Point outside all polygons

## Confidence Statistics by Category

### Exact Hist
- Count: 1317
- Mean confidence: 86.2
- Median confidence: 87.0
- Range: 61.5 - 100.0

### Exact Modern
- Count: 11
- Mean confidence: 87.8
- Median confidence: 90.5
- Range: 70.4 - 95.9

### Adjacent Modern
- Count: 46
- Mean confidence: 84.2
- Median confidence: 85.0
- Range: 67.4 - 100.0

### Mismatch
- Count: 4
- Mean confidence: 91.4
- Median confidence: 91.1
- Range: 91.0 - 92.2

## Files Generated

- Results CSV: `data/processed/spatial_validation.csv`
- Error GeoJSON: `data/processed/spatial_validation_errors.geojson`
- This report: `reports/spatial_validation.md`