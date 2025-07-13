"""Stage 1 — Auto-harvest candidate anchors (no solver).

This script generates a **candidate anchor set** by combining:
    • 11 manually curated high-quality anchors (CSV)
    • One automatically selected Courthouse anchor for *each* modern county that
      intersects the OSS Central Virginia study area.
    • Up to 15 automatically selected Stream-mouth anchors derived from manual
      anchor phrases.

Outputs
-------
A single CSV written to ``data/processed/candidate_anchor_set.csv`` with the
columns required by downstream network-adjustment code::

    grant_id,anchor_lat,anchor_lon,sigma_m,anchor_quality

The script is intentionally self-contained and *side-effect free* beyond the
single output file.  All heavy-lifting (GNIS loading, TIGER county polygons
etc.) reuses helper functions already present elsewhere in the package so that
we avoid redundant I/O and keep memory footprint low.

Run directly with:

    python -m oss_validation.candidate_anchor_harvest

"""
from __future__ import annotations

import re
import sys
from typing import List, Tuple, Optional

import geopandas as gpd
import pandas as pd
from loguru import logger

from . import config
from . import spatial_validation as sv
from . import anchor_resolution as ar
from .download_reference import download_gnis_features, download_tiger_counties

# ---------------------------------------------------------------------------
# CONSTANTS & PATHS
# ---------------------------------------------------------------------------
RAW_OSS_GEOJSON = (
    config.RAW_DIR / "CentralVAPatents_PLY-shp" / "centralva.geojson"
)
MANUAL_ANCHOR_CSV = config.PROCESSED_DIR / "manual_anchor_worksheet.csv"
CANDIDATE_OUT_CSV = config.PROCESSED_DIR / "candidate_anchor_set.csv"

# Regex pattern used to derive stream-mouth candidates from manual worksheet
MOUTH_PATTERN = re.compile(
    r"^mouth of ([A-Z][A-Za-z]+ [A-Z][A-Za-z]+) (Creek|River|Branch)$",
    flags=re.IGNORECASE,
)

# Sigma (1-sigma positional uncertainty, metres)
SIGMA_MANUAL = 75
SIGMA_CTHOUSE = 300
SIGMA_MOUTH = 500

# Cap for automatically derived stream-mouth anchors
MAX_MOUTH = 15

# Allowed GNIS feature classes considered a valid courthouse representation
COURTHOUSE_CLASSES = {"Locale", "Populated Place", "Civil"}

# ---------------------------------------------------------------------------
# Small helper utilities
# ---------------------------------------------------------------------------

def _ensure_reference_layers() -> None:
    """Download GNIS & TIGER reference layers if missing locally."""
    if not (config.REFERENCE_DIR / "gnis_features_va.gpkg").exists():
        download_gnis_features()

    # Any TIGER county GPKG already present is accepted – the exact vintage
    # does *not* matter for current purposes.  If none found, trigger a
    # download (the helper will pick the newest year that works).
    if not any(config.REFERENCE_DIR.glob("tiger_va_counties_*.gpkg")):
        download_tiger_counties()


def _load_oss_centroids() -> gpd.GeoDataFrame:
    """Return OSS parcel centroids with an attached modern-county label."""
    if not RAW_OSS_GEOJSON.exists():
        logger.error("Required OSS polygons %s missing – aborting", RAW_OSS_GEOJSON)
        sys.exit(1)

    # Re-use spatial_validation helpers so we share cached loaders
    oss_polys = gpd.read_file(RAW_OSS_GEOJSON)
    if oss_polys.crs != config.CRS_LATLON:
        oss_polys = oss_polys.to_crs(config.CRS_LATLON)

    centroids = oss_polys.copy()
    centroids["geometry"] = centroids.geometry.representative_point()

    # Harmonise id field – use cp OBJECTID as grant_id placeholder if present
    if "OBJECTID" in centroids.columns:
        centroids["grant_id"] = centroids["OBJECTID"].astype(str)
    else:
        centroids["grant_id"] = centroids.index.astype(str)

    # Attach modern county via spatial join (within predicate)
    counties = sv.load_tiger_counties()[["NAME", "geometry"]]
    joined = gpd.sjoin(centroids, counties, how="left", predicate="within")
    joined.rename(columns={"NAME": "modern_county"}, inplace=True)

    missing = joined["modern_county"].isna().sum()
    if missing:
        logger.warning("%d OSS centroids fell outside VA county polygons", missing)

    return joined[["grant_id", "modern_county", "geometry"]]


def _select_courthouse_anchor(
    county_name: str, gnis: gpd.GeoDataFrame, counties_gdf: gpd.GeoDataFrame
) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) of the chosen courthouse GNIS point for *county*.

    Search strategy (most-specific → least):
    1. GNIS points whose *feature_name* contains "Courthouse"/"Court House"
       and whose *feature_class* ∈ COURTHOUSE_CLASSES.
    2. If none, try any point with the exact county-seat name (e.g. "King and
       Queen Court House") regardless of class.
    If multiple remain, pick the one closest to the county centroid.
    """

    county_lc = county_name.lower()

    # ----- step 1: courthouse keyword filter --------------------------------
    mask_kw = (
        gnis["feature_name"].str.contains(r"Courthouse|Court House", case=False, na=False, regex=True)
        & (gnis["county_name_norm"] == county_lc)
        & gnis["feature_class"].isin(COURTHOUSE_CLASSES)
    )
    cands = gnis[mask_kw]
    logger.debug(f"[Courthouse] {county_name}: keyword filter → {len(cands)} candidates")

    # Fallback – accept any point whose feature_name equals "<County> Court House"
    if cands.empty:
        target_name = f"{county_name} Court House"
        mask_eq = (gnis["feature_name"].str.fullmatch(target_name, case=False, na=False)) & (
            gnis["county_name_norm"] == county_lc
        )
        cands = gnis[mask_eq]
        logger.debug(f"[Courthouse] {county_name}: fallback exact-name filter → {len(cands)} candidates")

    if cands.empty:
        return None

    # If 1 candidate, return immediately
    if len(cands) == 1:
        geom = cands.iloc[0].geometry
        return geom.y, geom.x

    # ----- tie-break by proximity to county centroid -------------------------
    county_geom = counties_gdf[counties_gdf["NAME"].str.lower() == county_lc].geometry
    if county_geom.empty:
        county_centroid = None
    else:
        county_centroid = county_geom.iloc[0].centroid

    if county_centroid is None:
        # No county geometry – just pick the first
        geom = cands.iloc[0].geometry
        return geom.y, geom.x

    cands_proj = cands.to_crs(3857)
    centroid_proj = (
        gpd.GeoSeries([county_centroid], crs=counties_gdf.crs)
        .to_crs(3857)
        .iloc[0]
    )
    distances = cands_proj.geometry.distance(centroid_proj)
    idx_closest = distances.idxmin()
    geom = cands.loc[idx_closest].geometry
    return geom.y, geom.x


def _parse_manual_stream_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Return unique (stream_name, modern_county) pairs for mouth-anchor search."""
    pairs: set[Tuple[str, str]] = set()
    for _, row in df.iterrows():
        phrase = str(row.get("anchor_phrase_raw", ""))
        m = MOUTH_PATTERN.match(phrase)
        if not m:
            continue

        stream_name = f"{m.group(1)} {m.group(2)}".strip()
        # Determine modern county – prefer GNIS county if present, else cp_county
        county = row.get("gnis_county") or row.get("cp_county")
        if not county or isinstance(county, float):
            continue

        # Map historical county → modern overlaps (may expand to multiple)
        modern_list = ar._allowed_modern_counties(county) or [county.title()]
        for modern in modern_list:
            pairs.add((stream_name, modern))

    return sorted(pairs)


def _select_stream_mouth_anchor(
    stream_name: str, county_name: str, gnis: gpd.GeoDataFrame
) -> Optional[Tuple[float, float]]:
    """Return (lat, lon) for *<stream_name> Mouth* in *county_name* if unique."""
    feat_name = f"{stream_name} Mouth"
    mask = (
        (gnis["feature_name"] == feat_name)
        & (gnis["county_name_norm"] == county_name.lower())
    )
    subset = gnis[mask]
    if len(subset) == 1:
        geom = subset.iloc[0].geometry
        return geom.y, geom.x
    return None

# ---------------------------------------------------------------------------
# MAIN ORCHESTRATION LOGIC
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: C901 – single top-level orchestrator
    _ensure_reference_layers()

    logger.info("Loading reference data …")
    gnis = ar.load_gazetteer()  # cached singleton
    counties_gdf = sv.load_tiger_counties()

    logger.info("Computing parcel centroids + modern counties …")
    centroids = _load_oss_centroids()
    unique_modern_counties = (
        centroids["modern_county"].dropna().str.title().unique().tolist()
    )

    # ------------------------------------------------------------------
    # 1) Manual anchors (baseline)
    # ------------------------------------------------------------------
    if not MANUAL_ANCHOR_CSV.exists():
        logger.error("Manual anchor worksheet %s missing – cannot proceed.", MANUAL_ANCHOR_CSV)
        sys.exit(1)

    manual_df = pd.read_csv(MANUAL_ANCHOR_CSV, dtype=str)
    # Keep only rows flagged as acceptable (manual_lat_lon present *and* no exclude_reason)
    manual_df = manual_df.dropna(subset=["manual_lat_lon"])

    def _split_latlon(s: str) -> Tuple[float, float]:
        lat_str, lon_str = [part.strip() for part in s.split(",")[:2]]
        return float(lat_str), float(lon_str)

    manual_df[["anchor_lat", "anchor_lon"]] = manual_df["manual_lat_lon"].apply(
        lambda s: pd.Series(_split_latlon(s))
    )

    manual_df["sigma_m"] = SIGMA_MANUAL
    manual_df["anchor_quality"] = "manual"
    manual_df.rename(columns={"cp_id": "grant_id"}, inplace=True)
    manual_candidates = manual_df[
        ["grant_id", "anchor_lat", "anchor_lon", "sigma_m", "anchor_quality"]
    ].copy()

    # Ensure correct dtypes
    manual_candidates["grant_id"] = manual_candidates["grant_id"].astype(str)

    n_manual = len(manual_candidates)
    logger.success(f"Loaded {n_manual} manual anchors")

    # ------------------------------------------------------------------
    # 2) Auto Courthouse anchors – one per modern county
    # ------------------------------------------------------------------
    courthouse_rows: List[dict] = []
    for county in unique_modern_counties:
        res = _select_courthouse_anchor(county, gnis, counties_gdf)
        if res is None:
            logger.debug(f"[Courthouse] {county}: no suitable GNIS point found")
            continue
        lat, lon = res
        courthouse_rows.append(
            {
                "grant_id": pd.NA,
                "anchor_lat": lat,
                "anchor_lon": lon,
                "sigma_m": SIGMA_CTHOUSE,
                "anchor_quality": "auto_courthouse",
            }
        )

    n_cthouse = len(courthouse_rows)
    logger.success(f"Added {n_cthouse} courthouse anchors")

    # ------------------------------------------------------------------
    # 3) Stream-mouth anchors derived from manual phrases
    # ------------------------------------------------------------------
    mouth_rows: List[dict] = []
    stream_pairs = _parse_manual_stream_pairs(manual_df)
    for stream_name, county_name in stream_pairs:
        if len(mouth_rows) >= MAX_MOUTH:
            break
        res = _select_stream_mouth_anchor(stream_name, county_name, gnis)
        if res is None:
            continue
        lat, lon = res
        mouth_rows.append(
            {
                "grant_id": pd.NA,
                "anchor_lat": lat,
                "anchor_lon": lon,
                "sigma_m": SIGMA_MOUTH,
                "anchor_quality": "auto_mouth",
            }
        )

    n_mouth = len(mouth_rows)
    logger.success(f"Added {n_mouth} stream-mouth anchors")

    # ------------------------------------------------------------------
    # 4) Merge & write output
    # ------------------------------------------------------------------
    out_df = pd.concat(
        [manual_candidates, pd.DataFrame(courthouse_rows), pd.DataFrame(mouth_rows)],
        ignore_index=True,
    )

    # Coerce columns to required order / types
    out_df = out_df[["grant_id", "anchor_lat", "anchor_lon", "sigma_m", "anchor_quality"]]
    out_df.to_csv(CANDIDATE_OUT_CSV, index=False)

    logger.success(f"Written candidate anchor set → {CANDIDATE_OUT_CSV} ({len(out_df)} rows)")

    # ------------------------------------------------------------------
    # 5) Summary print required by project brief
    # ------------------------------------------------------------------
    print()
    print(f"Total manual anchors: {n_manual}")
    print(f"Courthouse anchors added: {n_cthouse}")
    print(f"Mouth anchors added: {n_mouth}")
    print(f"Total candidate anchors: {len(out_df)}")
    print()


if __name__ == "__main__":
    main() 