"""Export unanchored C&P components as GeoPackage layers.

Generates **data/processed/unanchored_blocks.gpkg** with two layers:

1.  ``blocks``         – dissolved multipart polygons of each *connected component* that
    • contains ≥ 1 C&P parcel, **and**
    • *fails* the anchor rule (anchors < 2 and parcels ≥ 5 **or** anchors < 1 and parcels < 5).

    Attributes: ``comp_id`` (int), ``parcel_cnt`` (int), ``anchor_cnt`` (int)

2.  ``needed_points``   – representative point of each such component with the same
    attributes **plus** ``county`` (modern TIGER county name).

Usage (from repository root)::

    python -m oss_validation.export_gaps

A short pandas summary (top-20 by ``parcel_cnt``) is printed to stdout.
"""
from __future__ import annotations

from typing import List, Set, Dict

import geopandas as gpd
import pandas as pd
from loguru import logger

from . import config
from . import spatial_validation as sv
from .network_adjustment import (
    _load_polygons,
    _load_anchors,
    _snap_far_anchors,
    _assign_anchors_to_polygons,
    _build_edges,
    _connected_components,
    CENTRALVA,
    MATCHED_GRANTS_CSV,
)

# ---------------------------------------------------------------------------
# Output path ----------------------------------------------------------------
OUT_GPKG = config.PROCESSED_DIR / "unanchored_blocks.gpkg"

# ---------------------------------------------------------------------------
# Helpers --------------------------------------------------------------------

def _load_cp_overlap() -> Set[str]:
    """Return the set of grant_id present in *matched_grants.csv* (C&P subset)."""
    if not MATCHED_GRANTS_CSV.exists():
        logger.error("Required matched_grants.csv missing → %s", MATCHED_GRANTS_CSV)
        raise SystemExit(1)

    mg_df = pd.read_csv(MATCHED_GRANTS_CSV, dtype=str)
    ids: List[str] = []
    for col in ("grant_id", "cp_id", "oss_id"):
        if col in mg_df.columns:
            ids.extend(mg_df[col].astype(str).tolist())
    if not ids:
        logger.error("matched_grants.csv missing expected id columns (grant_id/cp_id/oss_id)")
        raise SystemExit(1)
    logger.success(f"Loaded C&P overlap list → {len(ids)} grant_ids")
    return set(ids)


def _component_stats(
    comps: List[List[int]],
    anchors_joined: pd.DataFrame,
    polys: gpd.GeoDataFrame,
    cp_overlap: Set[str],
) -> pd.DataFrame:
    """Return DataFrame with per-component statistics & failure flag."""
    rows: List[Dict[str, int | bool]] = []
    anchor_counts = anchors_joined.groupby("poly_idx").size()

    for comp_id, nodes in enumerate(comps):
        parcel_cnt = len(nodes)
        # anchor count – sum over parcels in component
        cnt = sum(anchor_counts.get(idx, 0) for idx in nodes)

        # Does component contain any C&P parcel?
        has_cp = any(str(polys.grant_id.iloc[idx]) in cp_overlap for idx in nodes)

        # Anchor rule failure ------------------------------------------------
        fails_rule = False
        if parcel_cnt >= 5:
            fails_rule = cnt < 2
        else:  # <5 parcels
            fails_rule = cnt < 1

        if has_cp and fails_rule:
            rows.append(
                {
                    "comp_id": comp_id,
                    "parcel_cnt": parcel_cnt,
                    "anchor_cnt": int(cnt),
                }
            )
    return pd.DataFrame(rows)


def _dissolve_component_geometry(comp_nodes: List[int], polys: gpd.GeoDataFrame):
    """Return a dissolved (multi)polygon for the component."""
    sub = polys.iloc[comp_nodes]
    # GeoPandas dissolve without dissolve_by column: unary_union
    return sub.geometry.unary_union

# ---------------------------------------------------------------------------
# Main CLI -------------------------------------------------------------------

def main() -> None:  # noqa: C901 (single orchestrator)
    logger.info("=== EXPORT UNANCHORED C&P BLOCKS ===")

    # 1. Load core data ------------------------------------------------------
    if not CENTRALVA.exists():
        logger.error("OSS polygons missing → %s", CENTRALVA)
        raise SystemExit(1)

    polys = _load_polygons()  # projected CRS (EPSG:3857)
    logger.success(f"Loaded {len(polys)} OSS polygons → {CENTRALVA.relative_to(config.ROOT_DIR)}")

    cp_overlap = _load_cp_overlap()

    # Ensure anchor CSV fallback logic mirrors network_adjustment ---------------
    from .network_adjustment import _abort_if_missing  # local import to avoid circulars
    _abort_if_missing()

    # Anchors (after snapping & parcel assignment) ---------------------------
    anchors = _snap_far_anchors(_load_anchors(), polys)
    anchors_joined = _assign_anchors_to_polygons(anchors, polys)

    # 2. Connected components -----------------------------------------------
    G = _build_edges(polys)
    comps = _connected_components(G)

    # 3. Per-component stats & filter failures ------------------------------
    stats_df = _component_stats(comps, anchors_joined, polys, cp_overlap)
    if stats_df.empty:
        logger.success("No unanchored components detected – nothing to export.")
        return

    # 4. Build GeoDataFrames -------------------------------------------------
    blocks_geoms: List = []
    for _, row in stats_df.iterrows():
        comp_id = int(row.comp_id)
        geom = _dissolve_component_geometry(comps[comp_id], polys)
        blocks_geoms.append(geom)

    blocks_gdf = gpd.GeoDataFrame(stats_df.copy(), geometry=blocks_geoms, crs=polys.crs)

    # Reproject to lat/lon before writing
    blocks_latlon = blocks_gdf.to_crs(config.CRS_LATLON)

    # Needed points ----------------------------------------------------------
    needed_pts = blocks_latlon.copy()
    needed_pts["geometry"] = needed_pts.geometry.representative_point()

    # Attach county via spatial join (within) --------------------------------
    counties = sv.load_tiger_counties()[["NAME", "geometry"]].copy()
    if counties.crs != config.CRS_LATLON:
        counties = counties.to_crs(config.CRS_LATLON)

    needed_pts = gpd.sjoin(needed_pts, counties, how="left", predicate="within")
    needed_pts.rename(columns={"NAME": "county"}, inplace=True)
    needed_pts = needed_pts.drop(columns=["index_right"])

    # 5. Write GeoPackage ----------------------------------------------------
    OUT_GPKG.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file to avoid layer overwrite errors ------------------
    if OUT_GPKG.exists():
        OUT_GPKG.unlink()

    blocks_latlon.to_file(OUT_GPKG, layer="blocks", driver="GPKG")
    needed_pts.to_file(OUT_GPKG, layer="needed_points", driver="GPKG")
    logger.success(f"GeoPackage written → {OUT_GPKG.relative_to(config.ROOT_DIR)}")

    # 6. Console summary -----------------------------------------------------
    summary = (
        stats_df.sort_values("parcel_cnt", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print("\nTOP-20 UNANCHORED C&P COMPONENTS (by parcel count):")
    print(summary.to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Interrupted by user – exiting.") 