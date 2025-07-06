"""Global configuration constants for the OSS validation pipeline."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Core paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"

# Ensure sub-directories exist (creates empty folders on first import)
for _dir in (RAW_DIR, REFERENCE_DIR, PROCESSED_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Coordinate reference systems
# ---------------------------------------------------------------------------
CRS_DISTANCE = "EPSG:3857"  # World Mercator – metres, good for local distances
CRS_LATLON = "EPSG:4326"    # WGS84 – lat/lon for final outputs

# ---------------------------------------------------------------------------
# Buffer sizes (kilometres)
# ---------------------------------------------------------------------------
BUFFER_RIVER_SIDE_KM = 2.0          # "N side of River" corridors
BUFFER_ROAD_OR_CREEK_KM = 5.0       # Vague linear-feature references
ADJ_NEIGHBOUR_BUFFER_KM = 2.0       # "adjacent" neighbour centroids

# Conversion helper (km → metres) ------------------------------------------------
KM_TO_M = 1_000.0


def km_to_m(km: float | int) -> float:
    """Convert kilometres to metres (Shapely/GPKG use metres in EPSG:3857)."""
    return float(km) * KM_TO_M 