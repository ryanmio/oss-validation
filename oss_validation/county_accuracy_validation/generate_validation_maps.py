# -*- coding: utf-8 -*-
"""Generate interactive maps for each county-validation error class.

Outputs one HTML file per class in *reports/* so they can be shared
independently (e.g. validation_mismatch_map.html).

Run as module:
    python -m oss_validation.generate_validation_maps

Optionally set a maximum number of points per map:
    python -m oss_validation.generate_validation_maps --limit 500
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import folium  # type: ignore
import geopandas as gpd
import pandas as pd
from folium.plugins import MarkerCluster  # type: ignore

from oss_preprocessing import config

# ---------------------------------------------------------------------------
# IO paths
# ---------------------------------------------------------------------------
VAL_CSV = config.PROCESSED_DIR / "spatial_validation.csv"
OSS_GEOJSON = config.RAW_DIR / "CentralVAPatents_PLY-shp" / "centralva.geojson"
OUT_DIR = config.REPORTS_DIR

# ---------------------------------------------------------------------------
# Classification groups to map  →  (marker-colour, nice label)
# ---------------------------------------------------------------------------
CATEGORIES: Dict[str, tuple[str, str]] = {
    "mismatch": ("red", "Centroid in wrong county (no reasonable match)"),
    "adjacent_modern": ("orange", "Centroid in adjacent modern county"),
    "boundary_tol": ("purple", "≤50 m outside historical county boundary"),
    "no_county": ("black", "Centroid outside all county polygons"),
    "near_split": ("blue", "Grant near county split (±2 yr)")
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_centroids() -> gpd.GeoDataFrame:
    gdf_poly = gpd.read_file(OSS_GEOJSON)
    gdf_poly["oss_id"] = gdf_poly["OBJECTID"].astype(str)
    cent = gdf_poly[["oss_id", "geometry"]].copy()
    cent["geometry"] = cent.geometry.representative_point()
    return cent.to_crs(config.CRS_LATLON)


def _build_one_map(df: pd.DataFrame, centroids: gpd.GeoDataFrame, colour: str, title: str, out_path: Path, limit: int | None = None) -> None:
    if limit:
        df = df.head(limit)
    if df.empty:
        print(f"[skip] 0 rows for {title}")
        return

    gdf = gpd.GeoDataFrame(
        df.merge(centroids, on="oss_id", how="left"),
        geometry="geometry",
        crs=config.CRS_LATLON,
    )

    m = folium.Map(location=[37.5, -78.7], zoom_start=7, tiles="CartoDB positron")
    folium.map.CustomPane("labels").add_to(m)  # keep tiles above base layers
    cluster = MarkerCluster().add_to(m)

    for _, row in gdf.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        popup_html = (
            f"<b>OSS ID:</b> {row.oss_id}<br>"
            f"<b>CP ID:</b> {row.cp_id}<br>"
            f"<b>Year:</b> {row.get('year_cp', row.get('year_oss'))}<br>"
            f"<b>Stated County:</b> {row.cp_county}<br>"
            f"<b>Notes:</b> {row.notes}<br>"
            f"<b>Confidence:</b> {row.confidence:.1f}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(cluster)

    m.get_root().html.add_child(folium.Element(f"<h4 style='position:absolute; top:10px; left:10px; z-index:9999; background:rgba(255,255,255,0.8); padding:4px 8px;'>{title}</h4>"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(out_path)
    print(f"[ok] {out_path.relative_to(config.ROOT_DIR)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Folium maps for each validation error class.")
    ap.add_argument("--limit", type=int, default=None, help="Maximum points per map (omit for all).")
    args = ap.parse_args()

    df = pd.read_csv(VAL_CSV, dtype={"oss_id": str})
    centroids = _load_centroids()

    for cls, (colour, label) in CATEGORIES.items():
        subset = df[df["classification"] == cls].copy()
        if subset.empty:
            print(f"[skip] no rows for class '{cls}'")
            continue
        out_file = OUT_DIR / f"validation_{cls}_map.html"
        _build_one_map(subset, centroids, colour, label, out_file, limit=args.limit)


if __name__ == "__main__":  # pragma: no cover
    main() 