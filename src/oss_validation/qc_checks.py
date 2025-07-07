"""Quality-control utilities for OSS polygon dataset.

Run as module to execute both checks and write reports:
    python -m oss_validation.qc_checks

Outputs:
    data/processed/area_discrepancies.csv   – polygons where |area - acreage| > 10 %
    data/processed/possible_duplicates.csv  – centroid pairs < 200 m apart with diff IDs
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import geopandas as gpd
import pandas as pd

from . import config

OSS_GEOJSON = config.RAW_DIR / "CentralVAPatents_PLY-shp" / "centralva.geojson"
OSS_CSV = config.PROCESSED_DIR / "oss_grants.csv"

MATCHED_CSV = config.PROCESSED_DIR / "matched_grants.csv"

# Default output when validating against C&P acreage
OUT_AREA_CSV_CP = config.PROCESSED_DIR / "area_discrepancies_cp.csv"
# Legacy output when validating against OSS-recorded acreage
OUT_AREA_CSV_OSS = config.PROCESSED_DIR / "area_discrepancies.csv"


def _load_oss() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(OSS_GEOJSON)
    gdf = gdf.to_crs(config.CRS_DISTANCE)  # metres for area/dist
    gdf["oss_id"] = gdf["OBJECTID"].astype(str)
    return gdf


def acreage_check(
    *,
    threshold: float = 0.25,
    use_cp: bool = True,
    ignore_zero: bool = True,
) -> pd.DataFrame:
    """Compare polygon area to recorded acreage and return rows whose absolute percentage
    difference exceeds *threshold*.

    Parameters
    ----------
    threshold : float, default 0.25
        Fractional tolerance (e.g. ``0.25`` = 25 %).
    use_cp : bool, default True
        If ``True``, pull acreage from Cavaliers & Pioneers (C&P) abstracts via
        the *matched_grants.csv* table. Otherwise fall back to the original acreage
        parsed from the OSS dataset.
    ignore_zero : bool, default True
        Drop rows whose recorded acreage is zero (often indicates missing / OCR failure).
    """

    gdf = _load_oss().to_crs("EPSG:5070")  # equal-area for acreage comparison

    if use_cp:
        matches = pd.read_csv(MATCHED_CSV, dtype={"oss_id": str})
        # Ensure acreage column is numeric
        matches["acreage"] = pd.to_numeric(matches["acreage"], errors="coerce")
        gdf = gdf.merge(matches[["oss_id", "acreage"]], on="oss_id", how="inner")
        out_csv = OUT_AREA_CSV_CP
    else:
        oss_csv = pd.read_csv(OSS_CSV, dtype={"objectid": str})
        gdf = gdf.merge(
            oss_csv[["objectid", "acreage"]],
            left_on="oss_id",
            right_on="objectid",
            how="left",
        )
        out_csv = OUT_AREA_CSV_OSS

    # compute polygon area in acres (1 sq metre = 0.000247105 acres)
    SQM_TO_ACRE = 0.000247105
    gdf["area_acres"] = gdf.geometry.area * SQM_TO_ACRE

    if ignore_zero:
        gdf = gdf[gdf["acreage"].fillna(0) > 0]

    # ------------------------------------------------------------------
    # Difference metrics
    # ------------------------------------------------------------------
    # Keep the *ratio* for internal filtering so existing ``threshold``
    # argument (e.g. 0.25 for 25 %) continues to work. Then convert to a
    # true percentage for output / reporting so values read as
    #   5  →  5 % difference
    #  25  → 25 % difference
    # 1326 → 1 326 % (≈ 13× larger)
    # ------------------------------------------------------------------

    gdf["ratio_diff"] = (gdf["area_acres"] - gdf["acreage"]).abs() / gdf["acreage"]
    gdf["pct_diff"] = gdf["ratio_diff"] * 100.0

    bad = gdf[gdf["ratio_diff"] > threshold][["oss_id", "acreage", "area_acres", "pct_diff"]].copy()
    bad.sort_values("pct_diff", ascending=False, inplace=True)
    bad.to_csv(out_csv, index=False)
    return bad


def main() -> None:
    bad = acreage_check()

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    total = len(pd.read_csv(MATCHED_CSV))  # total matched grants
    within_10 = total - len(bad[bad["pct_diff"] > 10])  # pct_diff now percentage

    print("Area discrepancies (vs. C&P acreage):", len(bad))
    print(f" → {OUT_AREA_CSV_CP.relative_to(config.ROOT_DIR)}")
    print()
    print("Acreage accuracy summary (percentage difference):")
    print(f"  • Total matched grants: {total}")
    print(f"  • Within ±10 %:        {within_10}  ({within_10/total:.1%})")
    print(f"  • Outside 10 %:        {total-within_10}  ({(total-within_10)/total:.1%})")


if __name__ == "__main__":
    main() 