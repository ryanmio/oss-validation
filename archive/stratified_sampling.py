# -*- coding: utf-8 -*-
"""Select a stratified random sample of OSS ↔ C&P matched grants.

The goal is to obtain **50 grants** that are reasonably diverse across (historical)
counties *and* across decades.  We therefore treat the combination of
``cp_county`` and ``decade`` (derived from the C&P grant year) as our *stratum*
identifier and draw a proportionally-allocated sample from those strata.

Usage
-----
Run from the project root:

    python -m oss_validation.stratified_sampling

The script writes the sample to
``data/processed/stratified_sample_50.csv`` and prints a short summary of the
allocation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from oss_preprocessing import config

# ---------------------------------------------------------------------------
# Input / output paths
# ---------------------------------------------------------------------------
SPATIAL_CSV = config.PROCESSED_DIR / "spatial_validation.csv"
MATCHED_CSV = config.PROCESSED_DIR / "matched_grants.csv"
OUT_CSV = config.PROCESSED_DIR / "stratified_sample_50.csv"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 50
SEED = config.RANDOM_SEED  # reproducibility


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_merged() -> pd.DataFrame:
    """Load spatial + matched grant data and prepare columns.

    The returned DataFrame contains:
        oss_id, cp_id, cp_county, year, decade, stratum
    """
    # Read core datasets (ensure oss_id is *string* for reliable merge)
    spatial = pd.read_csv(SPATIAL_CSV, dtype={"oss_id": str})
    matched = pd.read_csv(
        MATCHED_CSV,
        usecols=["oss_id", "cp_name", "acreage", "year_cp", "year_oss"],
        dtype={"oss_id": str},
    )

    # Merge and choose the most reliable year column (prefer C&P year)
    df = spatial.merge(matched, on="oss_id", how="left")
    df["year"] = df["year_cp"].fillna(df["year_oss"])
    df.dropna(subset=["cp_county", "year"], inplace=True)

    # Convert year → integer and derive decade (e.g. 1734 → 1730)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df.dropna(subset=["year"], inplace=True)
    df["decade"] = (df["year"] // 10) * 10

    # Form stratum identifier
    df["stratum"] = df["cp_county"].str.title().str.strip() + "_" + df["decade"].astype(str)

    return df


def _allocate_quota(df: pd.DataFrame) -> pd.Series:
    """Determine how many samples to take from each stratum (proportional)."""
    group_sizes = df.groupby("stratum").size()
    total = len(df)

    # Initial (possibly fractional) quotas
    fractional = group_sizes * SAMPLE_SIZE / total
    quotas = np.floor(fractional).astype(int)

    # Distribute remainder based on largest fractional parts ----------------
    remainder = SAMPLE_SIZE - quotas.sum()
    if remainder > 0:
        frac_part = fractional - quotas
        top_up_strata = frac_part.sort_values(ascending=False).index[:remainder]
        quotas.loc[top_up_strata] += 1

    # Edge-case: strata with size < quota (rare).  Reduce to available rows
    overfull = quotas[quotas > group_sizes]
    if not overfull.empty:
        diff = (overfull - group_sizes[overfull.index]).sum()
        quotas.loc[overfull.index] = group_sizes[overfull.index]
        # Redistribute the shortfall randomly to strata with spare capacity
        spare = quotas.index.difference(overfull.index)
        for stratum in np.random.default_rng(SEED).choice(spare, diff, replace=True):
            quotas[stratum] += 1

    assert quotas.sum() == SAMPLE_SIZE, "Quota allocation failed to sum to target size."
    return quotas


def draw_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Return a stratified random sample of the desired size."""
    quotas = _allocate_quota(df)
    rng = np.random.default_rng(SEED)

    sample_parts: list[pd.DataFrame] = []
    for stratum, n in quotas.items():
        subgroup = df[df["stratum"] == stratum]
        sampled = subgroup.sample(n=n, random_state=rng.integers(0, 2**32 - 1))
        sample_parts.append(sampled)

    return pd.concat(sample_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# CLI / entry-point
# ---------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – simple imperative description is fine here
    """Write the stratified 50-grant sample to CSV."""
    df = _load_merged()
    sample = draw_sample(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # Re-order columns for easier downstream use
    col_order = [
        "cp_id",
        "cp_name",
        "acreage",
        "year_cp",
        "cp_county",
        "oss_id",
        "year_oss",
    ]
    cols = [c for c in col_order if c in sample.columns] + [c for c in sample.columns if c not in col_order]

    sample[cols].to_csv(OUT_CSV, index=False)

    # Quick summary ---------------------------------------------------------
    county_counts = sample["cp_county"].value_counts().sort_index()
    decade_counts = sample["decade"].value_counts().sort_index()
    print("Sample written to", OUT_CSV.relative_to(config.ROOT_DIR))
    print("\nBy historical county:\n", county_counts.to_string())
    print("\nBy decade:\n", decade_counts.sort_index().to_string())


if __name__ == "__main__":
    main() 