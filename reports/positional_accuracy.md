# Positional Accuracy Summary

Total matched grants analysed: 1496

## County-level Accuracy (centroid in correct historical county)
- Correct count: **1435 / 1496**
- Accuracy: **95.9 %**  (95 % CI 94.8 – 96.8)

## Acreage Accuracy
- |Δacres| ≤ 30 %: **84.0 %**  (95 % CI 82.0 – 85.7)

### Distribution of |percentage acreage error|
- Mean: 21.33 %  (95 % CI 17.93 – 25.34)
- Median: 6.00 %  (95 % CI 5.42 – 6.88)
- 90th percentile: 45.8 %
- 95th percentile: 74.7 %

## Both County & Acreage Accurate (≤30 % error)
- 1203 / 1496 = **80.4 %**  (95 % CI 78.3 – 82.3)

## Methodology
1. County correctness includes exact historical matches, ≤50 m boundary tolerance, and near-split (±2 yr) cases.
2. Acreage error calculated in equal-area CRS (EPSG 5070).
3. Wilson score interval used for proportions; percentile bootstrap (5 000 reps, seed = 42) for mean/median.