# Tier-2 Positional Accuracy: Manual-Only Pipeline

This folder contains a **clean, reproducible workflow** for assessing OSS positional accuracy using manually geocoded anchor points from a stratified sample of 100 C&P grants.

## Overview

The **Tier-2 positional accuracy assessment** validates OSS polygon centroids against independently geocoded historical landmarks mentioned in Cavaliers & Pioneers deed texts. This implementation uses **manual geolocation only** (no automated GNIS resolution) to ensure high-quality, defensible anchor points.

### Key Design Principles
- **Stratified sampling**: 100 grants drawn proportionally across county × decade strata (seed = 42)
- **Manual anchors only**: Human experts geolocate features mentioned in deed texts
- **Conservative approach**: Only include grants with clearly identifiable, high-confidence anchor points
- **Full reproducibility**: All steps documented with scripts and data provenance

## Directory Structure

```
tier2_positional_accuracy/
├── README.md                                    # this file
├── data/
│   ├── stratified_100_anchor_worksheet.csv     # 100-grant stratified sample (seed=42)
│   └── stratified100_manual_worksheet.csv      # worksheet with manual coordinates
├── scripts/
│   ├── generate_stratified_worksheet.py        # create 100-grant sample (already run)
│   ├── build_manual_worksheet.py               # consolidate manual coordinates
│   ├── compute_distances_manual.py             # calculate centroid-anchor distances
│   └── compute_ci.py                           # bootstrap confidence intervals
└── results/
    ├── stratified100_final.csv                 # final dataset with distances
    └── ci_summary.txt                          # statistical summary
```

## Workflow

### Step 1: Generate Stratified Sample (Already Complete)
```bash
# This was already run to create the 100-grant sample
python tier2_positional_accuracy/scripts/generate_stratified_worksheet.py
```
- **Input**: `data/processed/matched_grants.csv`, `data/processed/spatial_validation.csv`
- **Output**: `tier2_positional_accuracy/data/stratified_100_anchor_worksheet.csv`
- **Details**: 100 grants, stratified by county × decade, RNG seed = 42

### Step 2: Manual Geolocation (User Task)
Edit `tier2_positional_accuracy/data/stratified100_manual_worksheet.csv`:
- **Task**: For each grant, identify geographic features mentioned in `cp_raw` text
- **Add coordinates**: Fill `manual_lat_lon` column with "latitude, longitude" (decimal degrees)
- **Add notes**: Document confidence level, sources, or mark as "bad match" if unusable
- **Focus**: Creek mouths, forks, mill sites, ferry crossings, and other precisely locatable features

### Step 3: Compute Distances
```bash
python tier2_positional_accuracy/scripts/compute_distances_manual.py
```
- **Input**: `stratified100_manual_worksheet.csv`, OSS centroids from original worksheets
- **Output**: `tier2_positional_accuracy/results/stratified100_final.csv`
- **Process**: Calculates great-circle distances between manual anchors and OSS centroids

### Step 4: Statistical Analysis
```bash
python tier2_positional_accuracy/scripts/compute_ci.py > tier2_positional_accuracy/results/ci_summary.txt
```
- **Input**: `stratified100_final.csv`
- **Output**: `ci_summary.txt` with bootstrap confidence intervals
- **Analysis**: 90th percentile distance error + 95% confidence interval

## Current Results

**Status**: 2 of 100 grants manually geocoded

| Metric | Value |
|--------|-------|
| Sample size | 2 |
| Median error | 1.51 km |
| Min error | 0.91 km |
| Max error | 2.11 km |

> ⚠️ **Sample too small**: Need ~15-20 additional manual anchors for meaningful confidence intervals.

## File Descriptions

### Data Files
- **`stratified_100_anchor_worksheet.csv`**: Original 100-grant stratified sample with suggested anchor phrases
- **`stratified100_manual_worksheet.csv`**: Working file for manual geolocation (edit this file)
- **`stratified100_final.csv`**: Complete results with distances and metadata

### Scripts
- **`generate_stratified_worksheet.py`**: Creates the initial 100-grant sample (already run)
- **`build_manual_worksheet.py`**: Merges existing manual coordinates into the worksheet
- **`compute_distances_manual.py`**: Computes great-circle distances using manual anchors only
- **`compute_ci.py`**: Bootstrap analysis for confidence intervals

### Results
- **`stratified100_final.csv`**: Final dataset (100 rows, distances for valid anchors)
- **`ci_summary.txt`**: Statistical summary with confidence intervals

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
python tier2_positional_accuracy/scripts/compute_distances_manual.py
python tier2_positional_accuracy/scripts/compute_ci.py > tier2_positional_accuracy/results/ci_summary.txt
```

**Key parameters**:
- Stratified sample: 100 grants, seed = 42
- Bootstrap: 10,000 resamples, seed = 42
- Distance metric: Great-circle (WGS84 ellipsoid) 