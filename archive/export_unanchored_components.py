"""Utility: Export each *unanchored* connected component of the OSS polygon
network to its own GeoJSON for visual QA in QGIS.

Usage
-----
    python -m oss_validation.export_unanchored_components

Outputs one file per component in ``results/unanchored_components`` named::

    comp_<id>_parcels_<n>.geojson

Where <id> is the component index (0-based) and <n> is the parcel count.
Only components that currently have **zero anchors** are exported.  The script
reuses Stage-2 helpers so it reflects exactly the same anchoring logic that
the network-adjustment run uses (distance-aware join, seat-snap, etc.).
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
from loguru import logger

from oss_validation.least_squares_validation import network_adjustment as na
from oss_preprocessing import config

OUT_DIR = config.ROOT_DIR / "results" / "unanchored_components"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:  # noqa: C901 – linear utility
    logger.info("Loading OSS polygons + anchors via Stage-2 helpers …")

    na._abort_if_missing()
    polys = na._load_polygons()  # projected CRS (metric)
    anchors = na._load_anchors()
    anchors = na._snap_far_anchors(anchors, polys)
    joined = na._assign_anchors_to_polygons(anchors, polys)

    logger.info("Building connectivity graph …")
    G = na._build_edges(polys)
    comps = na._connected_components(G)

    # Map nodes → comp_id for quick lookup
    node_to_comp: dict[int, int] = {}
    for cid, nodes in enumerate(comps):
        for n in nodes:
            node_to_comp[n] = cid

    joined["comp_id"] = joined["poly_idx"].map(node_to_comp)
    anchored_comps = set(joined["comp_id"].dropna().astype(int))

    exported = 0
    for cid, nodes in enumerate(comps):
        if cid in anchored_comps:
            continue  # component already anchored
        gdf_comp = polys.iloc[nodes].copy().to_crs(config.CRS_LATLON)
        out_fp = OUT_DIR / f"comp_{cid}_parcels_{len(nodes)}.geojson"
        gdf_comp.to_file(out_fp, driver="GeoJSON")
        exported += 1

    logger.success(f"Exported {exported} unanchored components → {OUT_DIR.relative_to(config.ROOT_DIR)}")
    if exported == 0:
        logger.info("All components already anchored – nothing to export.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1) 