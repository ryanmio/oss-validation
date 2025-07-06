# OSS Validation Pipeline

This repository implements a fully reproducible Python workflow to quantify how closely **One Shared Story** (OSS) land-grant centroids agree with the geographic information contained in the *Cavaliers & Pioneers, Volume&nbsp;3* textual abstracts.

---

## Repository layout

```
.
├── data/
│   ├── raw/                # Original inputs (C&P abstracts, OSS centroids)
│   ├── reference/          # Reference GIS layers downloaded locally
│   └── processed/          # Intermediate artifacts created by the pipeline
├── reports/
│   └── figures/            # QC plots and notebooks
├── src/                    # All pipeline code (importable as `oss_validation`)
│   ├── __init__.py
│   ├── config.py           # Global constants & paths
│   └── ...                 # Parsing, resolver, zone, scoring modules
├── tests/                  # Unit tests (pytest)
├── pyproject.toml          # Dependency lockfile / build metadata
└── README.md               # You are here
```

---

## Quick start

1. Create a fresh Python (≥3.9) environment and install the package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

2. Download the required datasets into the locations below (exact commands will be added later):

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

3. Run the full pipeline:

```bash
python -m oss_validation.pipeline
```

Outputs defined in the project brief will be written to `reports/`:

* `oss_validation_results.csv`
* `summary_stats.json`
* `feasible_zones.gpkg`
* `run_log.txt`

---

## Development workflow

* All source code lives under `src/` and is **import-safe** (no top-level I/O).
* Unit tests reside in `tests/` and are executed with `pytest -q`.
* Continuous integration (GitHub Actions) will lint with `ruff` and run the test suite on Ubuntu and macOS.

---

## License

MIT 