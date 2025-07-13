# -*- coding: utf-8 -*-
"""Rigorous positional-accuracy evaluation for OSS polygons.

Produces *per-patent* error fields plus a dataset-level statistical report that
includes Wilson score confidence intervals for proportions and percentile
bootstrap CIs for mean/median percentage acreage error.

Run:
    python -m oss_validation.positional_accuracy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .stats_utils import wilson_ci, bootstrap_ci

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SPATIAL_CSV = config.PROCESSED_DIR / "spatial_validation.csv"
AREA_CSV = config.PROCESSED_DIR / "area_validation.csv"

OUT_CSV = config.PROCESSED_DIR / "positional_accuracy_metrics.csv"
OUT_MD = config.REPORTS_DIR / "positional_accuracy.md"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
CORRECT_CLASSES = {"exact_hist", "boundary_tol", "near_split"}
ACRE_THRESHOLDS = [30]  # percentage difference considered acceptable
BOOT_REPS = 5000

# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    sv = pd.read_csv(SPATIAL_CSV, dtype={"oss_id": str})
    area = pd.read_csv(AREA_CSV, dtype={"oss_id": str})
    df = sv.merge(area[["oss_id", "pct_diff"]], on="oss_id", how="inner")

    df["county_correct"] = df["classification"].isin(CORRECT_CLASSES)
    for th in ACRE_THRESHOLDS:
        df[f"acre_within{th}"] = df["pct_diff"] <= th
    return df


def summarise(df: pd.DataFrame) -> str:
    n = len(df)
    lines: list[str] = ["# Positional Accuracy Summary", "", f"Total matched grants analysed: {n}", ""]

    # 1. County accuracy -----------------------------------------------------
    succ = int(df["county_correct"].sum())
    rate = succ / n
    ci_l, ci_u = wilson_ci(succ, n)
    lines.extend([
        "## County-level Accuracy (centroid in correct historical county)",
        f"- Correct count: **{succ} / {n}**",
        f"- Accuracy: **{rate*100:.1f} %**  (95 % CI {ci_l*100:.1f} – {ci_u*100:.1f})",
        "",
    ])

    # 2. Acreage accuracy thresholds ----------------------------------------
    lines.append("## Acreage Accuracy")
    for th in ACRE_THRESHOLDS:
        col = f"acre_within{th}"
        suc = int(df[col].sum())
        p = suc / n
        cl, cu = wilson_ci(suc, n)
        lines.append(
            f"- |Δacres| ≤ {th} %: **{p*100:.1f} %**  (95 % CI {cl*100:.1f} – {cu*100:.1f})"
        )

    # continuous stats
    pct = df["pct_diff"].abs()
    mean = pct.mean()
    median = pct.median()
    mean_ci = bootstrap_ci(pct, np.mean, reps=BOOT_REPS)
    med_ci = bootstrap_ci(pct, np.median, reps=BOOT_REPS)

    lines.extend(
        [
            "",
            "### Distribution of |percentage acreage error|",
            f"- Mean: {mean:.2f} %  (95 % CI {mean_ci[0]:.2f} – {mean_ci[1]:.2f})",
            f"- Median: {median:.2f} %  (95 % CI {med_ci[0]:.2f} – {med_ci[1]:.2f})",
            f"- 90th percentile: {np.percentile(pct, 90):.1f} %",
            f"- 95th percentile: {np.percentile(pct, 95):.1f} %",
            "",
        ]
    )

    # 3. Combined strict correctness ----------------------------------------
    primary_th = ACRE_THRESHOLDS[0]
    combo_col = f"acre_within{primary_th}"
    combo = df["county_correct"] & df[combo_col]
    suc_c = int(combo.sum())
    p_c = suc_c / n
    cl_c, cu_c = wilson_ci(suc_c, n)
    lines.extend(
        [
            f"## Both County & Acreage Accurate (≤{primary_th} % error)",
            f"- {suc_c} / {n} = **{p_c*100:.1f} %**  (95 % CI {cl_c*100:.1f} – {cu_c*100:.1f})",
            "",
            "## Methodology",
            "1. County correctness includes exact historical matches, ≤50 m boundary tolerance, and near-split (±2 yr) cases.",
            "2. Acreage error calculated in equal-area CRS (EPSG 5070).",
            "3. Wilson score interval used for proportions; percentile bootstrap (5 000 reps, seed = 42) for mean/median.",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_data()

    # write per-patent CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    # write report
    OUT_MD.write_text(summarise(df), encoding="utf-8")

    print("Positional accuracy metrics written:")
    print("  •", OUT_MD.relative_to(config.ROOT_DIR))
    print("  •", OUT_CSV.relative_to(config.ROOT_DIR))


if __name__ == "__main__":
    main() 