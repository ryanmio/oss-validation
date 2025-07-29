# Handoff Validation Working Document – One Shared Story Central VA Benchmark

_Last updated: <!-- TODO auto-update date later -->_

This scratch document gathers **all statistics and data file references** needed to demonstrate that the One Shared Story (OSS) Central Virginia land-grant polygons are suitable ground truth for the benchmark study described in `paper-files/main.md`.

> Anything NOT recorded here by the time we hand off will **not** reach the research assistant.

---

## Directory Reference

| Validation Test | Report Markdown | Primary CSV / data artefact |
|-----------------|-----------------|-----------------------------|
| Area (Acreage) Validation | `reports/area_validation.md` | `oss_validation/area_accuracy_validation/area_validation.csv` |
| Positional Accuracy Validation | `reports/positional_accuracy.md` | `oss_validation/county_accuracy_validation/positional_accuracy_metrics.csv` |
| Spatial Validation | `reports/spatial_validation.md` | `oss_validation/county_accuracy_validation/spatial_validation.csv` + `oss_validation/county_accuracy_validation/spatial_validation_errors.geojson` |
| Tier-2 Least-Squares Anchor Accuracy | `reports/least-squares.md` | `data/processed/tier2_positional_accuracy.csv` |
| Stratified Interval-Censored Accuracy | `reports/stratified-sample-interval.md` | `oss_validation/stratified_interval_accuracy/results/stratified100_final.csv` |

---

## 1 · Area (Acreage) Validation (N = 1 496 grants)

**Accuracy buckets (|Δacres| %)**  
_(95 % Wilson score CIs in parentheses)_

| Bucket | Count | Share | 95 % CI |
|--------|------:|------:|:-------|
| ≤ 5 % | 694 | 46.4 % | 43.9 – 48.9 % |
| 5–10 % | 222 | 14.8 % | 13.1 – 16.7 % |
| 10–25 % | 291 | 19.5 % | 17.5 – 21.5 % |
| 25–50 % | 154 | 10.3 % | 8.9 – 11.9 % |
| 50–100 % | 96 | 6.4 % | 5.3 – 7.8 % |
| 100–200 % | 25 | 1.7 % | 1.1 – 2.5 % |
| > 200 % | 14 | 0.9 % | 0.6 – 1.6 % |

Descriptive statistics of |Δacres| % (10 000-rep bootstrap)  
| Statistic | Point | 95 % CI |
|-----------|------:|:-------|
| **Mean** | 21.33 % | 17.9 – 25.3 % |
| **Median** | 6.00 % | 5.38 – 6.87 % |

> TODO T2 ✅  – Wilson CIs and bootstrap CIs now filled.

---

## 2 · Positional Accuracy Validation (County & Acreage)

Total grants analysed: **1 496**

### County-level placement

- Correct county centroids: **1 435 / 1 496**  
  👉 Accuracy = **95.9 %** (95 % CI 94.8 – 96.8 via Wilson)

### Acreage threshold (|Δacres| ≤ 30 %)

- Proportion meeting threshold: **84.0 %** (95 % CI 82.0 – 85.7)

Distribution of |percentage acreage error|:

| Statistic | Value | 95 % CI |
|-----------|-------|---------|
| Mean | 21.33 % | 17.93 – 25.34 |
| **Median** | 6.00 % | 5.42 – 6.88 |
| 90-th | 45.8 % | (not given) |
| 95-th | 74.7 % | (not given) |

Combined requirement (county correct **and** |Δacres| ≤ 30 %):  
**1 203 / 1 496 = 80.4 %** (95 % CI 78.3 – 82.3)

> Raw metrics file: `oss_validation/county_accuracy_validation/positional_accuracy_metrics.csv`

---

## 3 · Spatial Validation Classification (N = 1 496)  
_(95 % Wilson score CIs in last column)_

| Class | Description | Count | Share | 95 % CI |
|-------|-------------|------:|------:|:-------|
| exact_hist | Point in correct historical county | 1 317 | 88.0 % | 86.3 – 89.6 % |
| boundary_tol | ≤ 50 m from polygon edge | 112 | 7.5 % | 6.3 – 8.9 % |
| near_split | Year-split ±2 yr | 6 | 0.4 % | 0.18 – 0.87 % |
| exact_modern | In modern county (no split) | 11 | 0.7 % | 0.41 – 1.31 % |
| adjacent_modern | Modern neighbour | 46 | 3.1 % | 2.31 – 4.08 % |
| mismatch | No reasonable match | 4 | 0.3 % | 0.10 – 0.69 % |
| no_county | Point outside all polygons | 0 | 0.0 % | — |

> TODO T4 ✅ – Added CIs for every proportion.

Confidence score summary by main categories (already provided):

- **Exact Hist**: n = 1 317 — Mean 86.2, Median 87.0, Range 61.5–100
- **Exact Mod**: n = 11 — Mean 87.8, Median 90.5, Range 70.4–95.9
- **Adjacent Mod**: n = 46 — Mean 84.2, Median 85.0, Range 67.4–100
- **Mismatch**: n = 4 — Mean 91.4, Median 91.1, Range 91.0–92.2

Artefacts:  
`oss_validation/county_accuracy_validation/spatial_validation.csv`,  
`oss_validation/county_accuracy_validation/spatial_validation_errors.geojson`,  
`oss_validation/county_accuracy_validation/validation_errors_map.html` (visual for 4 out-of-county cases)

---

## 4 · Tier-2 Least-Squares Absolute Positional Accuracy (Anchor Audit)

Snapshot file: `archive/tier2_final_with_manual_filtered.csv`

### Anchor extraction / GNIS resolution

| Stage | Count |
|-------|------:|
| Total matched grants | 1 496 |
| Regex-extracted anchors | 152 |
| Unique GNIS-resolved anchors | 75 |
| Ambiguous (>1 GNIS) | 0 |
| No GNIS match | 77 |
| **Good anchors after manual audit** | **39** |

### Distance metrics (good anchors, full polygons)

- Sample size: 40 (post-audit “good” anchors)
- **Median great-circle error**: 3.18 km
- **90-th percentile**: 6.92 km  
  95 % bootstrap CI (10 000 reps): **7.38 – 18.34 km**

> TODO T3 ✅ – 95 % CI for P90 added.

Interpretation: Meets project goal of ≤ 10 km for 90 % of anchor points.

Supplementary artefacts:

| File | Purpose |
|------|---------|
| `data/processed/tier2_anchor_audit_good_worst10.csv` | Details on worst 10 good anchors |
| `reports/tier2_ecdf.png` | ECDF of error distances (to generate) |
| `reports/anchor_validation_workflow.md` | End-to-end workflow description |

---

## 5 · Stratified Interval-Censored Positional Accuracy (Manual & Model)

Reference data: `oss_validation/stratified_interval_accuracy/results/stratified100_final.csv`

### Manual anchor distances (n = 60)

| Metric | Value (km) |
|--------|-----------:|
| Min | 0.43 |
| Q25 | 1.18 |
| **Median** | 2.14 |
| Mean | 3.09 |
| Q75 | 3.21 |
| Max | 38.06 |
| Std-dev | 4.93 |
| **P90** | 4.83 |
| **P90 95 % CI** | 3.45 – 7.06 km |

### Interval-censored log-normal model (exact + censored, N = 100)

- MLE parameters: μ = 0.707, σ = 0.832
- 90-th percentile (model): **5.89 km**  (95 % CI 4.21 – 8.04 km)

---

## 6 · Known Gaps / Follow-up Tasks

| TODO ID | Gap description | Status |
|---------|----------------|--------|
| **T2** | Wilson CIs + bootstrap CIs (Area) | **completed** |
| **T3** | Bootstrap CI for Tier-2 P90 | **completed** |
| **T4** | CIs for Spatial class breakdown | **completed** |
| **T5** | Generate ECDF/histograms for area % error and anchor distance error | pending (matplotlib install issues) |
| **T6** | Evaluate need for additional maps (anchor points, error density) | pending |

---

## 7 · Data & Visualisation Assets Checklist

- [ ] `oss_validation/area_accuracy_validation/area_validation.csv`
- [ ] `oss_validation/county_accuracy_validation/positional_accuracy_metrics.csv`
- [ ] `oss_validation/county_accuracy_validation/spatial_validation.csv`
- [ ] `oss_validation/county_accuracy_validation/spatial_validation_errors.geojson`
- [ ] `data/processed/tier2_positional_accuracy.csv`
- [ ] `oss_validation/stratified_interval_accuracy/results/stratified100_final.csv`
- [ ] Maps / plots generated under `reports/plots/`

Once the gaps above are addressed, this document will be distilled into a polished **handoff summary** (see TODO T8). 

### Plot artefacts

A reusable script `tools/generate_validation_plots.py` now produces:

* `reports/plots/area_error_ecdf.png`
* `reports/plots/area_error_hist.png`
* `reports/plots/tier2_anchor_ecdf.png`

Run `python tools/generate_validation_plots.py` inside a virtual environment with `matplotlib` installed.

--- 