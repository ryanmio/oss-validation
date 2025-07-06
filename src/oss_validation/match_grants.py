from __future__ import annotations

"""Match OSS polygon-excerpt grants with Cavaliers & Pioneers abstracts.

Matching rule (heuristic):
  • acreage must match exactly after rounding to 1 decimal (some OSS values are xx.0)
  • years must match within ±1
  • grantee names compared with a token-set similarity; accept if ≥ 75

Output: data/processed/matched_grants.csv with columns:
    oss_id, cp_id, sim, acreage, year_oss, year_cp, oss_name, cp_name
"""

from pathlib import Path
from typing import List, Dict
import math
import sys
import re

import pandas as pd

try:
    from rapidfuzz.fuzz import token_set_ratio  # type: ignore
except ImportError:  # fall back to stdlib
    from difflib import SequenceMatcher

    def token_set_ratio(a: str, b: str) -> float:  # type: ignore
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100.0

from . import config

OSS_CSV = config.PROCESSED_DIR / "oss_grants.csv"
CP_CSV = config.PROCESSED_DIR / "cp_grants.csv"
OUT_CSV = config.PROCESSED_DIR / "matched_grants.csv"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_acre(val: float | int | str | None) -> float | None:
    """Round acreage to one decimal so that 348 == 348.0."""
    try:
        if pd.isna(val):
            return None
        f = float(val)
        return round(f, 1)
    except (TypeError, ValueError):
        return None


def _similar(a: str | None, b: str | None) -> float:
    if not a or not b:
        return 0.0
    return token_set_ratio(a, b)

# Pre-compile simple tokenizer regex to strip punctuation / numbers for raw-text comparison
_TOKEN_RE = re.compile(r"[^a-z\s]+")

def _clean_text(txt: str | None) -> str:
    """Very lightweight cleanup: lower-case, remove punctuation/digits for fuzzy raw text match."""
    if not isinstance(txt, str):
        return ""
    txt = txt.lower()
    txt = _TOKEN_RE.sub(" ", txt)
    return " ".join(txt.split())


def _raw_similarity(a: str | None, b: str | None) -> float:
    """Token-set similarity on lightly-normalised raw excerpt strings."""
    return _similar(_clean_text(a), _clean_text(b))

# ---------------------------------------------------------------------------


def build_matched_table(
    out_csv: Path = OUT_CSV,
    name_sim_threshold: float = 80.0,
    raw_sim_threshold: float = 35.0,
    weight_name: float = 0.5,
    weight_raw: float = 0.5,
) -> pd.DataFrame:
    """Match OSS polygons to Cavaliers & Pioneers abstracts.

    1. Candidate filtering on (acreage, year ±1).
    2. Require *name* similarity ≥ *name_sim_threshold* **and** *raw* similarity ≥ *raw_sim_threshold*.
    3. Score = ``weight_name * name_sim + weight_raw * raw_sim`` (weights default 0.5/0.5).
    4. One-to-one greedy assignment: highest-scoring pairs taken first.

    The function writes *out_csv* and prints detailed overlap stats.
    """

    # ------------------------------------------------------------------
    # Load + preprocess core fields
    # ------------------------------------------------------------------
    df_oss = pd.read_csv(OSS_CSV, dtype={"objectid": str})
    df_cp = pd.read_csv(CP_CSV, dtype={"grant_id": str})

    df_oss["acre"] = df_oss["acreage"].apply(_norm_acre)
    df_cp["acre"] = df_cp["acreage"].apply(_norm_acre)

    df_oss = df_oss.dropna(subset=["acre", "year", "name_std", "raw_excerpt"]).copy()
    df_cp = df_cp.dropna(subset=["acre", "year", "name_std", "raw_entry"]).copy()

    # Quick duplicate diagnostics (groups with >1 row sharing name+acre+year)
    dup_oss = (
        df_oss.groupby(["name_std", "acre", "year"]).size().gt(1).sum()
    )
    dup_cp = (
        df_cp.groupby(["name_std", "acre", "year"]).size().gt(1).sum()
    )

    # Index OSS by acreage for quick candidate retrieval
    oss_by_acre: Dict[float, pd.DataFrame] = {
        a: grp.reset_index(drop=True) for a, grp in df_oss.groupby("acre")
    }

    # ------------------------------------------------------------------
    # Build candidate list with similarity scores
    # ------------------------------------------------------------------
    cand_rows: List[Dict[str, object]] = []

    from collections import Counter

    for _, cp in df_cp.iterrows():
        acre = cp["acre"]
        if acre not in oss_by_acre:
            continue  # no acreage match at all

        oss_pool = oss_by_acre[acre]

        # YEAR must match exactly ---------------------------------------
        yr_cp = int(cp["year"])
        oss_filtered = oss_pool[oss_pool["year"] == yr_cp]
        if oss_filtered.empty:
            continue

        for _, oss in oss_filtered.iterrows():
            name_sim = _similar(cp["name_std"], oss["name_std"])
            if name_sim < name_sim_threshold:
                continue

            raw_sim = _raw_similarity(cp["raw_entry"], oss["raw_excerpt"])
            if raw_sim < raw_sim_threshold:
                continue

            confidence = weight_name * name_sim + weight_raw * raw_sim

            cand_rows.append({
                "confidence": confidence,
                "name_sim": round(name_sim, 1),
                "raw_sim": round(raw_sim, 1),
                "cp_id": cp["grant_id"],
                "oss_id": oss["objectid"],
                "acreage": acre,
                "year_cp": yr_cp,
                "year_oss": int(oss["year"]),
                "cp_name": cp["name_std"],
                "oss_name": oss["name_std"],
                "cp_raw": cp["raw_entry"],
                "oss_raw": oss["raw_excerpt"],
            })

        # keep track of number of oss candidates per cp row
        # (for diagnostic of potential ambiguity)
        # We will update after outer loop finishes.

    if not cand_rows:
        print("No candidate matches found – check thresholds.")
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Greedy assignment: highest-scoring first (one-to-one mapping)
    # ------------------------------------------------------------------
    cand_rows.sort(key=lambda d: d["confidence"], reverse=True)

    matched_cp: set[str] = set()
    matched_oss: set[str] = set()
    final_rows: List[Dict[str, object]] = []

    for row in cand_rows:
        if row["cp_id"] in matched_cp or row["oss_id"] in matched_oss:
            continue  # already taken
        matched_cp.add(row["cp_id"])
        matched_oss.add(row["oss_id"])
        final_rows.append(row)

    # ------------------------------------------------------------------
    # Write & report stats
    # ------------------------------------------------------------------
    df_match = pd.DataFrame(final_rows)
    df_match.to_csv(out_csv, index=False)

    # Ambiguity diagnostics -------------------------------------------------
    cp_cand_counts = Counter(row["cp_id"] for row in cand_rows)
    ambiguous_cp = sum(1 for _id, cnt in cp_cand_counts.items() if cnt > 1)

    # Overlap diagnostics --------------------------------------------------
    unmatched_cp = len(df_cp) - len(matched_cp)
    unmatched_oss = len(df_oss) - len(matched_oss)

    from statistics import mean, median

    conf_values = [row["confidence"] for row in final_rows]
    conf_mean = mean(conf_values) if conf_values else 0
    conf_median = median(conf_values) if conf_values else 0
    conf_min = min(conf_values) if conf_values else 0
    conf_max = max(conf_values) if conf_values else 0

    pct_match_cp = len(df_match) / len(df_cp) * 100 if df_cp is not None and len(df_cp)>0 else 0
    pct_match_oss = len(df_match) / len(df_oss) * 100 if len(df_oss)>0 else 0

    summary_lines = []
    summary_lines.append("\n=== Matching summary ===")
    summary_lines.append(f"Total OSS rows:       {len(df_oss):6d}")
    summary_lines.append(f"Total C&P rows:       {len(df_cp):6d}")
    summary_lines.append(f"Duplicate OSS groups: {dup_oss:6d}")
    summary_lines.append(f"Duplicate C&P groups: {dup_cp:6d}")
    summary_lines.append(f"Unique matches:       {len(df_match):6d}  ({pct_match_cp:5.1f}% of C&P, {pct_match_oss:5.1f}% of OSS)")
    summary_lines.append(f"Unmatched OSS:        {unmatched_oss:6d}  ({unmatched_oss/len(df_oss)*100:5.1f}%)")
    summary_lines.append(f"Unmatched C&P:        {unmatched_cp:6d}  ({unmatched_cp/len(df_cp)*100:5.1f}%)")
    summary_lines.append(f"Ambiguous C&P rows:   {ambiguous_cp:6d}  (had >1 OSS candidate)")
    summary_lines.append("-- Confidence scores --")
    summary_lines.append(f"min: {conf_min:5.1f}  median: {conf_median:5.1f}  mean: {conf_mean:5.1f}  max: {conf_max:5.1f}")
    summary_lines.append(f"Output → {out_csv.relative_to(config.ROOT_DIR)}")

    # Echo to console
    print("\n".join(summary_lines))

    # ------------------------------------------------------------------
    # Write summary to markdown report
    # ------------------------------------------------------------------
    report_path = config.REPORTS_DIR / "match_stats.md"
    md_lines = ["# OSS ↔ C&P Matching Summary", "", *summary_lines[1:]]  # skip initial blank line
    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown report written → {report_path.relative_to(config.ROOT_DIR)}")

    return df_match


if __name__ == "__main__":
    build_matched_table() 