"""Compute distance between resolved anchor points and OSS centroids.

Run:
    python -m oss_validation.tier2_distance

Inputs:
    • data/processed/tier2_positional_accuracy.csv (from tier2_positional_accuracy.py)
    • Raw OSS polygons (centralva.geojson)

Outputs:
    • data/processed/tier2_positional_accuracy.csv (overwritten with d_km + flags)

Columns added:
    d_km              Great-circle distance, kilometres (float)
    within_10km       bool flag (distance ≤10 km)
"""
from __future__ import annotations

from pathlib import Path
import logging

import geopandas as gpd
import pandas as pd
from pyproj import Geod

from . import config
from .tier2_positional_accuracy import OUT_CSV as POS_CSV  # existing output path

OSS_GEOJSON = config.RAW_DIR / "CentralVAPatents_PLY-shp" / "centralva.geojson"
GEOD = Geod(ellps="WGS84")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_centroids() -> pd.DataFrame:
    """Compute representative points for OSS polygons and return (oss_id, lat, lon)."""
    logging.info("Loading OSS polygons …")
    gdf = gpd.read_file(OSS_GEOJSON)[["OBJECTID", "geometry"]]
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    gdf["geometry"] = gdf.geometry.representative_point()
    gdf["oss_id"] = gdf.OBJECTID.astype(str)
    gdf["centroid_lat"] = gdf.geometry.y
    gdf["centroid_lon"] = gdf.geometry.x
    return gdf[["oss_id", "centroid_lat", "centroid_lon"]]


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance (km) between two WGS84 points."""
    _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0


def main() -> None:
    if not POS_CSV.exists():
        raise FileNotFoundError(f"Expected positional accuracy file {POS_CSV} not found — run tier2_positional_accuracy first.")

    logging.info("Loading resolved anchors …")
    df = pd.read_csv(POS_CSV)
    df["oss_id"] = df["oss_id"].astype(str)

    if "centroid_lat" not in df.columns:
        centroids = load_centroids()
        logging.info("Merging anchor data with centroids …")
        df = df.merge(centroids, on="oss_id", how="left")

    # Compute distance where we have both coords and unique match
    mask = (df["ambiguity_flag"] == 0) & df["anchor_lat"].notna() & df["centroid_lat"].notna()
    df.loc[mask, "d_km"] = df.loc[mask].apply(
        lambda r: great_circle_km(r.anchor_lat, r.anchor_lon, r.centroid_lat, r.centroid_lon), axis=1
    )
    df["within_10km"] = df["d_km"] <= 10.0

    # Write back same CSV (idempotent)
    df.to_csv(POS_CSV, index=False)

    usable = df["d_km"].notna().sum()
    median = df["d_km"].median()
    pct90 = df["d_km"].quantile(0.9)
    logging.info("Distance metrics on %d anchors: median %.2f km, 90th percentile %.2f km", usable, median, pct90)

if __name__ == "__main__":
    main() 