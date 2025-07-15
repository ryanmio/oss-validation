# Stratified Sample Positional Accuracy (Interval-Censored Model)

This folder contains the **fourth accuracy analysis** on the OSS polygon dataset:

> **Stratified random sample of 100 Cavaliers & Pioneers grants (county × decade) with interval-censored parametric fit for grants that lack locatable textual features.**

It supersedes the earlier “Tier-2 manual-only” experiment; the code has been moved into `stratified_interval_accuracy/` (the old `tier2_positional_accuracy/` directory name is retained only in commit history).

### Relationship to the full validation suite

1. County-level accuracy validation  
2. Acreage accuracy validation  
3. Network / least-squares adjustment analyses  
4. **Stratified sample + interval-censored log-normal fit ← this folder**

The present workflow is fully self-contained and reproducible.

## Overview

This analysis validates OSS polygon centroids against **manual anchor points** extracted from deed texts. For grants where no high-confidence anchor could be located, we treat the error as interval-censored 
a value somewhere in (0, U], where U = ½ × county width. A log-normal distribution is fitted by maximum likelihood.

### Key Design Principles
- **Stratified sampling**: 100 grants drawn proportionally across county × decade strata (seed = 42)
- **Manual anchors only**: Human experts geolocate features mentioned in deed texts
- **Conservative approach**: Only include grants with clearly identifiable, high-confidence anchor points
- **Full reproducibility**: All steps documented with scripts and data provenance

## Directory Structure

```
stratified_interval_accuracy/
├── README.md                                    # this file
├── data/
│   ├── stratified_100_anchor_worksheet.csv     # 100-grant stratified sample (seed=42)
│   └── stratified100_manual_worksheet.csv      # worksheet with manual coordinates
├── scripts/
│   ├── generate_stratified_worksheet.py        # create 100-grant sample (already run)
│   ├── build_manual_worksheet.py               # consolidate manual coordinates
│   ├── compute_distances_with_fallback.py      # distances + county-width fallback
│   ├── compute_interval_ci.py                 # interval-censored MLE + bootstrap
│   └── generate_accuracy_report.py            # builds results/accuracy_report.md
└── results/
    ├── stratified100_final.csv                 # final dataset with distances
    ├── ci_interval_summary.txt                # interval-censored stats
    └── accuracy_report.md                     # human-readable Markdown summary
```

## Workflow

### Step 1: Generate Stratified Sample (Already Complete)
```bash
# This was already run to create the 100-grant sample
python stratified_interval_accuracy/scripts/generate_stratified_worksheet.py
```
- **Input**: `data/processed/matched_grants.csv`, `data/processed/spatial_validation.csv`
- **Output**: `stratified_interval_accuracy/data/stratified_100_anchor_worksheet.csv`
- **Details**: 100 grants, stratified by county × decade, RNG seed = 42

### Step 2: Manual Geolocation (User Task)
Edit `stratified_interval_accuracy/data/stratified100_manual_worksheet.csv`:
- **Task**: For each grant, identify geographic features mentioned in `cp_raw` text
- **Add coordinates**: Fill `manual_lat_lon` column with "latitude, longitude" (decimal degrees)
- **Add notes**: Document confidence level, sources, or mark as "bad match" if unusable
- **Focus**: Creek mouths, forks, mill sites, ferry crossings, and other precisely locatable features

### Step 3: Compute Distances (& Fallbacks)
```bash
python stratified_interval_accuracy/scripts/compute_distances_with_fallback.py
```
- **Input**: `stratified100_manual_worksheet.csv`, OSS centroids from original worksheets
- **Output**: `stratified_interval_accuracy/results/stratified100_final.csv`
- **Process**: Calculates great-circle distances between manual anchors and OSS centroids

### Step 4: Interval-Censored Statistical Analysis
```bash
python stratified_interval_accuracy/scripts/compute_interval_ci.py
```
- **Input**: `stratified100_final.csv`
- **Output**: `ci_interval_summary.txt` with interval-censored statistics
- **Analysis**: 90th percentile distance error + 95% confidence interval

### Step 5 (optional): Generate Markdown Report
```bash
python stratified_interval_accuracy/scripts/generate_accuracy_report.py
```
- **Input**: `stratified100_final.csv`, `ci_interval_summary.txt`
- **Output**: `accuracy_report.md`

## Current Results

**Current headline results** (2025-07-15):

| Metric | Value |
|--------|-------|
| Manual anchors (exact) | 60 |
| Median error | 2.1 km |
| 90th percentile (exact only) | 4.8 km (95 % CI 3.5–7.1) |
| 90th percentile (interval-censored model) | 5.9 km (95 % CI 4.2–8.0) |

Upper confidence limit remains **< 10 km**, satisfying the accuracy benchmark.

## File Descriptions

### Data Files
- **`stratified_100_anchor_worksheet.csv`**: Original 100-grant stratified sample with suggested anchor phrases
- **`stratified100_manual_worksheet.csv`**: Working file for manual geolocation (edit this file)
- **`stratified100_final.csv`**: Complete results with distances and metadata

### Scripts

- **`generate_stratified_worksheet.py`** — create the 100-grant stratified sample (already run)  
- **`build_manual_worksheet.py`** — merge any existing manual coordinates  
- **`compute_distances_with_fallback.py`** — compute distances (manual or fallback)  
- **`compute_interval_ci.py`** — fit interval-censored log-normal & bootstrap  
- **`generate_accuracy_report.py`** — assemble `accuracy_report.md`

### Results

- **`stratified100_final.csv`** — final dataset with distances & metadata  
- **`ci_interval_summary.txt`** — CLI summary from `compute_interval_ci.py`  
- **`accuracy_report.md`** — nicely formatted report for publication

## Quality Control

### Manual Geolocation Guidelines
1. **Precision**: Only geolocate features you can locate with high confidence
2. **Sources**: Use historical maps, GNIS, satellite imagery, local knowledge
3. **Documentation**: Record sources and confidence in the `notes` column
4. **Exclusions**: Mark uncertain locations as "bad match" or leave coordinates blank

### Validation Checks
- Coordinates within Virginia bounds (-83.68 to -75.17 longitude, 36.54 to 39.46 latitude)
- Distance < 200 km from deed location (sanity check)
- Manual review of outliers

## Next Steps

To achieve statistical significance for the 90% confidence claim:
1. **Continue manual geolocation**: Target 15-20 additional high-confidence anchors
2. **Focus on clear features**: Creek mouths, forks, mills, ferries, bridges
3. **Re-run analysis**: Execute Steps 3-4 after adding more anchors
4. **Documentation**: Update this README with final results

## Reproducibility

All analysis is fully reproducible:
```bash
# Run the complete pipeline (after manual geolocation)
cd /path/to/repo
python stratified_interval_accuracy/scripts/compute_distances_with_fallback.py
python stratified_interval_accuracy/scripts/compute_interval_ci.py
python stratified_interval_accuracy/scripts/generate_accuracy_report.py
```

**Key parameters**:
- Stratified sample: 100 grants, seed = 42
- Bootstrap (manual subset): 10 000 resamples  
- Bootstrap (interval model): 1 000 resamples  
- Distance metric: Great-circle (WGS84 ellipsoid) 