"""Global configuration constants for the OSS validation pipeline."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Core paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"

# Ensure sub-directories exist
for _dir in (RAW_DIR, REFERENCE_DIR, PROCESSED_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# Reproducibility -----------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# CRS ----------------------------------------------------------------------
CRS_DISTANCE = "EPSG:3857"
CRS_LATLON = "EPSG:4326"

# Buffer sizes (km) ---------------------------------------------------------
BUFFER_RIVER_SIDE_KM = 2.0
BUFFER_ROAD_OR_CREEK_KM = 5.0
ADJ_NEIGHBOUR_BUFFER_KM = 2.0
BUFFER_POINT_KM = 1.0

# Helpers -------------------------------------------------------------------
KM_TO_M = 1_000.0

def km_to_m(km: float | int) -> float:
    """Convert kilometres to metres."""
    return float(km) * KM_TO_M 