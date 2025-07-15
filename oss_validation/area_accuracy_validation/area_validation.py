# -*- coding: utf-8 -*-
"""Area (acreage) validation for OSS polygons.

Compares the polygon area (computed in an equal-area CRS) with the acreage
recorded in Cavaliers & Pioneers abstracts (via *matched_grants.csv*).
Generates two artefacts:
  • Full results CSV   → data/processed/area_validation.csv
  • Markdown report    → reports/area_validation.md

Run as a module:
    python -m oss_validation.area_validation
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import geopandas as gpd

from ...src.oss_preprocessing import config

# ---------------------------------------------------------------------------
# IO paths
# ---------------------------------------------------------------------------
OSS_GEOJSON = config.RAW_DIR / "CentralVAPatents_PLY-shp" / "centralva.geojson"
MATCHED_CSV = config.PROCESSED_DIR / "matched_grants.csv"

OUT_CSV = config.PROCESSED_DIR / "area_validation.csv"
OUT_MD = config.REPORTS_DIR / "area_validation.md"

# Conversion constant – square-metres → acres
SQM_TO_ACRE = 0.000247105

# Buckets for percentage difference breakdown
BUCKETS = {
    "≤5%": (0, 5),
    "5–10%": (5, 10),
    "10–25%": (10, 25),
    "25–50%": (25, 50),
    "50–100%": (50, 100),
    "100–200%": (100, 200),
    ">200%": (200, float("inf")),
}


def load_data() -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Load OSS polygons (projected) and matched acreage table."""
    gdf = gpd.read_file(OSS_GEOJSON).to_crs("EPSG:5070")  # equal-area
    gdf["oss_id"] = gdf["OBJECTID"].astype(str)

    matches = pd.read_csv(MATCHED_CSV, dtype={"oss_id": str})
    matches["acreage"] = pd.to_numeric(matches["acreage"], errors="coerce")

    merged = gdf.merge(matches[["oss_id", "acreage"]], on="oss_id", how="inner")
    return merged, matches


def compute_metrics(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Return DataFrame with computed area and pct_diff (percentage)."""
    df = gdf.copy()
    df["area_acres"] = df.geometry.area * SQM_TO_ACRE
    df["pct_diff"] = (df["area_acres"] - df["acreage"]).abs() / df["acreage"] * 100.0
    return df[["oss_id", "acreage", "area_acres", "pct_diff"]]


def bucket_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label, (lo, hi) in BUCKETS.items():
        counts[label] = int(df[(df["pct_diff"] > lo) & (df["pct_diff"] <= hi)].shape[0])
    return counts


def write_report(df: pd.DataFrame) -> None:
    total = len(df)
    counts = bucket_counts(df)

    # Basic descriptive stats
    pct_stats = df["pct_diff"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]).round(2)

    lines: list[str] = [
        "# Area (Acreage) Validation Summary",
        "",
        f"Total matched grants analysed: {total}",
        "",
        "## Accuracy Breakdown (percentage difference)",
    ]
    for label, cnt in counts.items():
        pct = cnt / total * 100
        lines.append(f"- **{label}**: {cnt}  ({pct:.1f}%)")

    lines.extend(
        [
            "",
            "## Descriptive Statistics of Percentage Difference",
            "",
            f"- Min: {pct_stats['min']} %",
            f"- 25th percentile: {pct_stats['25%']} %",
            f"- Median: {pct_stats['50%']} %",
            f"- 75th percentile: {pct_stats['75%']} %",
            f"- 90th percentile: {pct_stats['90%']} %",
            f"- 95th percentile: {pct_stats['95%']} %",
            f"- Max: {pct_stats['max']} %",
            "",
            "## Files Generated",
            "",
            f"- Full results CSV: `{OUT_CSV.relative_to(config.ROOT_DIR)}`",
            f"- This report: `{OUT_MD.relative_to(config.ROOT_DIR)}`",
        ]
    )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    gdf, _ = load_data()
    df = compute_metrics(gdf)

    # save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # write markdown report
    write_report(df)

    print("Area validation report written:")
    print("  •", OUT_MD.relative_to(config.ROOT_DIR))
    print("  •", OUT_CSV.relative_to(config.ROOT_DIR))


if __name__ == "__main__":
    main() 