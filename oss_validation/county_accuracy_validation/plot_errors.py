"""Generate an interactive Folium map of spatial-validation mismatches.

Usage:
    python -m oss_validation.plot_errors  [--limit 200]

Outputs `reports/validation_errors_map.html`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import folium  # type: ignore
import geopandas as gpd
import pandas as pd
from folium.plugins import MarkerCluster  # type: ignore

from oss_preprocessing import config

VAL_CSV = config.PROCESSED_DIR / "spatial_validation.csv"
OSS_GEOJSON = config.RAW_DIR / "CentralVAPatents_PLY-shp" / "centralva.geojson"
OUT_HTML = config.REPORTS_DIR / "validation_errors_map.html"


def build_map(limit: int | None = None) -> Path:
    """Create Folium map of mismatches and return output path."""
    # Read validation results
    df = pd.read_csv(VAL_CSV, dtype={"oss_id": str})
    mism = df[(df["classification"] == "mismatch") & df["cp_county"].notna() & (df["cp_county"].str.strip() != "")].copy()
    if limit:
        mism = mism.head(limit)

    if mism.empty:
        raise RuntimeError("No mismatch rows found in validation CSV.")

    # Load centroids of OSS polygons
    gdf_poly = gpd.read_file(OSS_GEOJSON)
    gdf_poly["oss_id"] = gdf_poly["OBJECTID"].astype(str)
    gdf_cent = gdf_poly[["oss_id", "geometry"]].copy()
    gdf_cent["geometry"] = gdf_cent.geometry.representative_point()
    gdf_cent = gdf_cent.to_crs(config.CRS_LATLON)

    # Merge with mismatch table
    gdf = mism.merge(gdf_cent, on="oss_id", how="left")
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=config.CRS_LATLON)

    # Build Folium map centered on VA approx
    m = folium.Map(location=[37.5, -78.7], zoom_start=7, tiles="CartoDB positron")
    cluster = MarkerCluster().add_to(m)

    for _, row in gdf.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        popup = (
            f"<b>OSS ID:</b> {row.oss_id}<br>"
            f"<b>CP ID:</b> {row.cp_id}<br>"
            f"<b>Year:</b> {row.get('year_cp', row.get('year_oss'))}<br>"
            f"<b>Stated County:</b> {row.cp_county}<br>"
            f"<b>Historical county hit:</b> {row.notes.split(';')[0].replace('Hist=', '')}<br>"
            f"<b>Confidence:</b> {row.confidence:.1f}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="red",
            fill=True,
            fill_opacity=0.8,
            popup=folium.Popup(popup, max_width=300),
        ).add_to(cluster)

    m.save(OUT_HTML)
    return OUT_HTML


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot validation mismatches on Folium map.")
    parser.add_argument("--limit", type=int, default=200, help="Max number of points to plot.")
    args = parser.parse_args()

    out = build_map(args.limit)
    print(f"Interactive map written → {out.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main() 