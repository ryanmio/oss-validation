"""Tier-2 positional-accuracy pipeline (anchor resolution only).

Run:
    python -m oss_validation.tier2_positional_accuracy

Steps
------
1. Load anchor candidates from data/processed/anchor_resolution_CANDIDATES.csv
2. Resolve each anchor to GNIS coordinates using oss_validation.anchor_resolution
3. Write resolved file to data/processed/tier2_positional_accuracy.csv
   (column order fixed for downstream distance analysis)
4. Print quick summary to stdout.

This script is purely I/O orchestration; all matching logic lives in
oss_validation.anchor_resolution.
"""
from __future__ import annotations

from pathlib import Path
import logging

import pandas as pd
import geopandas as gpd

from oss_preprocessing import config  # project-level dirs
from oss_validation.least_squares_validation import anchor_resolution as ar
# Avoid circular import: use the constant already defined in anchor_resolution
POS_CSV = ar.TIER2_OUT_CSV

CANDIDATES_CSV = config.PROCESSED_DIR / "anchor_resolution_CANDIDATES.csv"
OUT_CSV = ar.TIER2_OUT_CSV  # points to data/processed/tier2_positional_accuracy.csv

# Spatial validation results containing parsed counties (avoid re-parsing)
SPATIAL_CSV = config.PROCESSED_DIR / "spatial_validation.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Desired column order for downstream processing / paper tables
OUT_COLS = [
    "cp_id",
    "cp_county",  # from spatial_validation
    "oss_id",
    "centroid_lat",  # diagnostics
    "centroid_lon",
    "anchor_phrase_raw",
    "anchor_type",
    "anchor_feature_name",
    "gnis_name",
    "gnis_class",
    "gnis_county",
    "anchor_quality",
    "anchor_lat",
    "anchor_lon",
    "match_score",
    "ambiguity_flag",
    "exclude_reason",
]

def load_centroids() -> pd.DataFrame:
    """Compute representative points for OSS polygons and return (oss_id, centroid_lat, centroid_lon)."""
    oss_geojson = config.RAW_DIR / "CentralVAPatents_PLY-shp" / "centralva.geojson"
    gdf = gpd.read_file(oss_geojson)[["OBJECTID", "geometry"]]
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    gdf["geometry"] = gdf.geometry.representative_point()
    gdf["oss_id"] = gdf.OBJECTID.astype(str)
    gdf["centroid_lat"] = gdf.geometry.y
    gdf["centroid_lon"] = gdf.geometry.x
    return gdf[["oss_id", "centroid_lat", "centroid_lon"]]

def main() -> None:
    if not CANDIDATES_CSV.exists():
        raise FileNotFoundError(
            f"Expected candidates file {CANDIDATES_CSV} missing – run extraction first."
        )

    logging.info("Loading anchor-resolution candidates (%s) …", CANDIDATES_CSV.name)
    df_cand = pd.read_csv(CANDIDATES_CSV)

    # Harmonise id types for join operations
    df_cand["cp_id"] = df_cand["cp_id"].astype(str)
    if "oss_id" in df_cand.columns:
        df_cand["oss_id"] = df_cand["oss_id"].astype(str)

    # ------------------------------------------------------------------
    # Merge in county information from the existing spatial_validation run
    # ------------------------------------------------------------------
    if SPATIAL_CSV.exists():
        spatial_df = pd.read_csv(SPATIAL_CSV, dtype={"cp_id": str})[
            ["cp_id", "cp_county"]
        ]
        before_merge = len(df_cand)
        df_cand = df_cand.merge(spatial_df, on="cp_id", how="left")
        logging.info(
            "Merged county info from spatial_validation: %d rows matched (of %d)",
            df_cand["cp_county"].notna().sum(),
            before_merge,
        )
    else:
        logging.warning("Spatial validation file %s not found – county merge skipped", SPATIAL_CSV)

    # Add centroid coordinates so resolver can pick nearest GNIS point
    centroids = load_centroids()
    df_cand = df_cand.merge(centroids, on="oss_id", how="left")

    logging.info("Resolving %d anchors …", len(df_cand))
    df_res = ar.resolve_anchors(df_cand)

    # Reorder columns (any missing columns are appended automatically)
    cols = [c for c in OUT_COLS if c in df_res.columns] + [c for c in df_res.columns if c not in OUT_COLS]
    df_res = df_res[cols]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(OUT_CSV, index=False)

    n_unique = int((df_res["ambiguity_flag"] == 0).sum())
    logging.info("Written %s (%d rows, %d unique matches)", OUT_CSV, len(df_res), n_unique)

if __name__ == "__main__":
    main() 