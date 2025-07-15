# OSS Validation Pipeline

This repository implements a fully reproducible Python workflow to quantify how closely **One Shared Story** (OSS) land-grant centroids agree with the geographic information contained in the *Cavaliers & Pioneers, Volume&nbsp;3* textual abstracts.

---

## Quick start

1. Create a fresh Python (≥3.9) environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

2. Download the required datasets into the locations below:

Use oss_preprocessing/download_reference.py to download the reference GIS layers.

```
# raw inputs
data/raw/cp_vol3_abstracts.txt
data/raw/oss_centroids.geojson

# reference GIS layers
data/reference/tiger_va_counties_2023.gpkg
data/reference/nhd_flowlines_hu4_0208.gpkg
data/reference/nhd_flowlines_hu4_0301.gpkg
data/reference/gnis_features_va.gpkg
data/reference/osm_roads_va.gpkg
```

3. Run the preprocessing scripts
oss_preprocessing/oss_extract.py
oss_preprocessing/cp_extract.py
oss_preprocessing/match_grants.py
oss_preprocessing/hist_modern_overlap.py

4. Run the validation analyses:

There are 4 separate validation analyses, each in its own directory:

1. **oss_validation/area_accuracy_validation/**
    - This analysis validates the area of the OSS polygons. It calculates the area (in acres) of the OSS polygons and compares it to the acreage recorded in the Cavaliers & Pioneers abstracts.
    - **Result:** 80.7 % of grants (1 207 / 1 496) have acreage error ≤ 25 % — median error just 6 %.

2. **oss_validation/county_accuracy_validation/**
    - This analysis validates the county of the OSS polygons. It determines the county that each OSS polygon lies within and compares it to the county recorded in the Cavaliers & Pioneers abstracts. It uses the tigher_va_counties_2023.gpkg file with historical split normalization.
    - **Result:** County-level centroid accuracy = 95.9 % (1435 / 1496 grants correct).

3. **oss_validation/least-squares_validation/**
    - This analysis validates the least-squares network adjustment of the OSS polygons. It uses manually plotted anchor points for modern identifiable anchor points and constrains the network using polygon shared edges.
    - **Result:** 90th-percentile anchor error = 6.9 km on 39 audited anchors

4. **oss_validation/stratified_interval_accuracy/**
    - This analysis validates the positional accuracy of the OSS polygons. It uses a stratified random sample of 100 matched OSS-C&P grants and a interval-censored model to estimate the positional accuracy.
    - **Result:** 90th-percentile positional error = 5.9 km (95 % CI 4.21 – 8.04 km) from interval-censored model.

---

## License

MIT 