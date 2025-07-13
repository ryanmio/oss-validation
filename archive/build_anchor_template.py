"""Stage 1b – Build manual anchor-template worksheet (no geo lookup).

Generates **data/processed/anchor_template.csv** containing:
    anchor_id, anchor_type, county, feature_name, anchor_lat, anchor_lon, sigma_m

Rows:
1. One *_seat* row per unique modern county intersecting the Central VA parcel
   dataset (sigma_m = 700 m).
2. Up to 20 *_mouth* rows for two-word stream names parsed from the existing
   *manual_anchor_worksheet.csv* (sigma_m = 500 m).

No external geocoding is performed – lat/lon columns are left blank for manual
entry.  The file is intended for the user to fill in precise coordinates.
"""
from __future__ import annotations

import re
import sys
from typing import List, Tuple

import geopandas as gpd
import pandas as pd
from loguru import logger

from . import config
from . import spatial_validation as sv

RAW_OSS_GEOJSON = config.RAW_DIR / "CentralVAPatents_PLY-shp" / "centralva.geojson"
MANUAL_ANCHOR_CSV = config.PROCESSED_DIR / "manual_anchor_worksheet.csv"
TEMPLATE_OUT_CSV = config.PROCESSED_DIR / "anchor_template.csv"

# Regex to capture "mouth of X Y (River|Creek|Branch)" (two-word names only)
MOUTH_REGEX = re.compile(
    r"mouth of ([A-Z][A-Za-z]+ [A-Z][A-Za-z]+) (Creek|River|Branch)$",
    flags=re.IGNORECASE,
)

MAX_MOUTHS = 20  # user-specified cap

# County → seat town mapping (partial list provided by user)
SEAT_MAP = {
    "Albemarle": "Charlottesville",
    "Augusta": "Staunton",
    "Buckingham": "Buckingham",
    "Campbell": "Rustburg",
    "Caroline": "Bowling Green",
    "Chesterfield": "Chesterfield",
    "Culpeper": "Culpeper",
    "Dinwiddie": "Dinwiddie",
    "Fluvanna": "Palmyra",
    "Franklin": "Rocky Mount",
    "Goochland": "Goochland",
    "Hanover": "Hanover",
    "Henrico": "Richmond",
    "Louisa": "Louisa",
    "Madison": "Madison",
    "Nelson": "Lovingston",
    "Orange": "Orange",
    "Powhatan": "Powhatan",
    "Prince George": "Prince George",
    "Rockbridge": "Lexington",
    "Rockingham": "Harrisonburg",
    "Spotsylvania": "Spotsylvania Courthouse",
    "Surry": "Surry",
    "Sussex": "Sussex",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_modern_counties() -> List[str]:
    """Return sorted list of unique modern counties intersecting study area."""
    if not RAW_OSS_GEOJSON.exists():
        logger.error("Missing %s – cannot continue", RAW_OSS_GEOJSON)
        sys.exit(1)

    logger.info("Loading OSS polygons …")
    gdf = gpd.read_file(RAW_OSS_GEOJSON)[["geometry"]].copy()
    if gdf.crs != config.CRS_LATLON:
        gdf = gdf.to_crs(config.CRS_LATLON)

    # Representative points → spatial join with TIGER counties
    gdf["geometry"] = gdf.geometry.representative_point()
    counties_gdf = sv.load_tiger_counties()[["NAME", "geometry"]]
    joined = gpd.sjoin(gdf, counties_gdf, how="left", predicate="within")

    modern = (
        joined["NAME"]
        .dropna()
        .str.replace(r"\s+(County|city)$", "", regex=True)
        .str.title()
        .unique()
        .tolist()
    )
    modern_sorted = sorted(modern)
    logger.success(f"Identified {len(modern_sorted)} modern counties")
    return modern_sorted


def _extract_stream_mouths() -> List[Tuple[str, str]]:
    """Return list of (stream_name, hist_county) up to MAX_MOUTHS."""
    if not MANUAL_ANCHOR_CSV.exists():
        logger.warning("Manual anchor worksheet %s missing; no mouth anchors", MANUAL_ANCHOR_CSV)
        return []

    df = pd.read_csv(MANUAL_ANCHOR_CSV, dtype=str)
    pairs: List[Tuple[str, str]] = []

    for _, row in df.iterrows():
        phrase = str(row.get("anchor_phrase_raw", ""))
        m = MOUTH_REGEX.match(phrase)
        if not m:
            continue
        stream = f"{m.group(1)} {m.group(2)}".strip()
        county = str(row.get("cp_county", "")).title() if pd.notna(row.get("cp_county")) else ""
        pair = (stream, county)
        if pair not in pairs:
            pairs.append(pair)
        if len(pairs) >= MAX_MOUTHS:
            break

    logger.success(f"Extracted {len(pairs)} mouth stream names (limit {MAX_MOUTHS})")
    return pairs

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main() -> None:
    modern_counties = _extract_modern_counties()
    mouth_pairs = _extract_stream_mouths()

    rows: List[dict] = []

    # County seats ------------------------------------------------------
    for county in modern_counties:
        rows.append(
            {
                "anchor_id": f"{county.replace(' ', '_')}_seat",
                "anchor_type": "seat",
                "county": county,
                "feature_name": SEAT_MAP.get(county, county),
                "anchor_lat": "",
                "anchor_lon": "",
                "sigma_m": 700,
            }
        )

    # Stream mouths -----------------------------------------------------
    for stream, county in mouth_pairs:
        anchor_id = f"{stream.replace(' ', '_')}_mouth"
        rows.append(
            {
                "anchor_id": anchor_id,
                "anchor_type": "mouth",
                "county": county,
                "feature_name": f"{stream} Mouth",
                "anchor_lat": "",
                "anchor_lon": "",
                "sigma_m": 500,
            }
        )

    out_df = pd.DataFrame(rows)
    TEMPLATE_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(TEMPLATE_OUT_CSV, index=False)

    logger.success(f"Written anchor template → {TEMPLATE_OUT_CSV} ({len(out_df)} rows)")

    # Print quick preview
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main() 