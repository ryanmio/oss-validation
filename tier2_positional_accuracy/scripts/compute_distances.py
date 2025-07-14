"""compute_distances.py
========================
Merge automated GNIS anchor resolutions (tier2_positional_accuracy.csv) with
manual overrides from audit worksheets, compute great-circle distances to OSS
polygon centroids, and export:

* results/final_with_manual.csv           – full 152-row table with final_lat,
  final_lon, final_d_km, coord_source, etc.
* results/worst10_good_anchors.csv        – ten largest residuals among the
  _good_ subset (manual+auto, excluding bad/discard rows).

The logic is identical to the interactive workflow used during analysis; this
stand-alone script makes the process reproducible for reviewers.

Run:
    python scripts/compute_distances.py  # uses default paths inside repo

Command-line options (see --help) allow overriding any path.
"""
from __future__ import annotations

import re
import glob
from pathlib import Path
from typing import Tuple, List

import click
import numpy as np
import pandas as pd
from pyproj import Geod

from . import config

GEOD = Geod(ellps="WGS84")

###############################################################################
# Helpers
###############################################################################

def _parse_latlon(val: str | float | None) -> Tuple[float | None, float | None]:
    """Parse a decimal-degree "lat, lon" pair. Return (lat, lon) or (None, None)."""
    if val is None or isinstance(val, float):
        return None, None
    val = str(val).strip()
    m = re.match(r"^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)", val)
    if not m:
        return None, None
    try:
        lat, lon = float(m.group(1)), float(m.group(2))
        # loose sanity check – VA-ish bounding box
        if 34 <= lat <= 40 and -85 <= lon <= -70:
            return lat, lon
    except ValueError:
        pass
    return None, None


def _great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0

###############################################################################
# Core pipeline
###############################################################################

def load_manual_overrides(patterns: List[str]) -> pd.DataFrame:
    """Read one or more audit worksheets and return cp_id → overrides."""
    frames: list[pd.DataFrame] = []
    for pat in patterns:
        for path in glob.glob(pat):
            df = pd.read_csv(path)
            # normalise column names
            if "Manual Latlon" in df.columns:
                df = df.rename(columns={"Manual Latlon": "manual_lat_lon"})
            if "Manual LatLon" in df.columns:
                df = df.rename(columns={"Manual LatLon": "manual_lat_lon"})
            if "manual_lat_lon" not in df.columns:
                df["manual_lat_lon"] = ""
            if "Notes" not in df.columns:
                df["Notes"] = ""
            df["cp_id"] = df["cp_id"].astype(str)
            df["manual_bad"] = (
                df["manual_lat_lon"].fillna("").str.lower().str.contains("bad match|discard")
                | df["Notes"].fillna("").str.lower().str.contains("bad match|discard")
            )
            # parse coords unless bad
            coords = df.apply(
                lambda r: (None, None)
                if r.manual_bad
                else _parse_latlon(r.manual_lat_lon),
                axis=1,
                result_type="expand",
            )
            df["manual_lat"], df["manual_lon"] = coords[0], coords[1]
            frames.append(df[["cp_id", "manual_lat", "manual_lon", "manual_bad"]])
    if not frames:
        raise FileNotFoundError("No manual worksheets matched the given patterns.")
    merged = pd.concat(frames, ignore_index=True)
    # consolidate duplicates: if any row says bad, mark bad; else first coords
    final = (
        merged.sort_values(["manual_bad"], ascending=False)
        .groupby("cp_id", as_index=False)
        .agg({
            "manual_bad": "max",  # True dominates
            "manual_lat": "first",
            "manual_lon": "first",
        })
    )
    return final


def apply_overrides(
    base_csv: Path,
    overrides: pd.DataFrame,
    out_final: Path,
    out_worst: Path,
    worst_n: int = 10,
):
    tier2 = pd.read_csv(base_csv)
    tier2["cp_id"] = tier2["cp_id"].astype(str)
    overrides["cp_id"] = overrides["cp_id"].astype(str)

    merged = tier2.merge(overrides, on="cp_id", how="left")

    # Determine coordinate source & final coords
    merged["coord_source"] = np.where(
        merged["manual_bad"] == True,
        "bad",
        np.where(merged["manual_lat"].notna(), "manual", np.where(merged["anchor_lat"].notna(), "auto", "none")),
    )
    merged["final_lat"] = np.where(merged["coord_source"] == "manual", merged["manual_lat"], merged["anchor_lat"])
    merged["final_lon"] = np.where(merged["coord_source"] == "manual", merged["manual_lon"], merged["anchor_lon"])

    # Distances ------------------------------------------------------------
    mask = merged["final_lat"].notna() & merged["centroid_lat"].notna()
    merged.loc[mask, "final_d_km"] = merged.loc[mask].apply(
        lambda r: _great_circle_km(r.final_lat, r.final_lon, r.centroid_lat, r.centroid_lon), axis=1
    )

    merged.to_csv(out_final, index=False)

    # Good subset (same filter we’ve been using)
    good = merged[
        (merged["coord_source"] != "bad")
        & (merged["ambiguity_flag"] == 0)
        & (merged["exclude_reason"].isna())
        & merged["final_lat"].notna()
    ].copy()

    # Metrics --------------------------------------------------------------
    if len(good) > 0:
        median = good["final_d_km"].median()
        p90 = good["final_d_km"].quantile(0.9)
    else:
        median = p90 = float("nan")

    print("Good anchors:", len(good))
    print(f"Median distance = {median:.2f} km")
    print(f"90th percentile  = {p90:.2f} km")

    # Worst-N list ---------------------------------------------------------
    worst = good.sort_values("final_d_km", ascending=False).head(worst_n)
    cols = [
        "cp_id",
        "cp_county",
        "oss_id",
        "final_d_km",
        "coord_source",
        "anchor_phrase_raw",
        "anchor_feature_name",
        "anchor_type",
        "gnis_name",
        "gnis_county",
        "anchor_quality",
        "final_lat",
        "final_lon",
        "centroid_lat",
        "centroid_lon",
        "match_score",
        "cp_raw",
    ]
    worst[cols].to_csv(out_worst, index=False)
    print(f"Worst-{worst_n} list written → {out_worst.relative_to(config.ROOT_DIR)}")

###############################################################################
# CLI
###############################################################################


@click.command()
@click.option("--base", type=click.Path(exists=True, path_type=Path), default=Path("data/processed/tier2_positional_accuracy.csv"), help="Base Tier-2 CSV (auto GNIS results).")
@click.option(
    "--worksheets",
    multiple=True,
    default=["data/processed/100 anchor worksheet - Sheet*.csv"],
    help="Glob(s) for manual audit worksheets.",
)
@click.option("--out-final", type=click.Path(path_type=Path), default=Path("data/processed/tier2_final_with_manual.csv"))
@click.option("--out-worst", type=click.Path(path_type=Path), default=Path("data/processed/tier2_anchor_audit_good_worst10.csv"))
def cli(base: Path, worksheets: Tuple[str, ...], out_final: Path, out_worst: Path):
    """Apply manual overrides and export updated stats + worst-10 list."""

    overrides = load_manual_overrides(list(worksheets))
    apply_overrides(base, overrides, out_final, out_worst)


if __name__ == "__main__":
    cli() 