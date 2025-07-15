"""Extract grantee name, acreage and year from OSS polygon layer.

Writes a clean CSV (oss_grants.csv) containing one row per polygon with:
    objectid, name_std, acreage, year

The extraction relies on the fact that *exactly one* of the truncated
DBF-text columns (e.g. ``Albemarl_1``) holds the first line of the printed
abstract that starts with the acreage, while another column (e.g. ``Albemarle_``)
contains the grantee name + optional year.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import pandas as pd
import pyogrio

from . import config

OSS_PATH = config.RAW_DIR / "CentralVAPatents_PLY-shp" / "centralva.geojson"
OUT_CSV = config.PROCESSED_DIR / "oss_grants.csv"

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Acreage like "334a", "3060 ac", "99.5a" (case-insensitive)
_ACRE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*a(?:c|s)?", re.IGNORECASE)
# Year 16xx–18xx (allow trailing ?)
_YEAR_RE = re.compile(r"(16|17|18)\d{2}\??")
# Grantee name before first comma, strip common titles
_TITLE_RE = re.compile(r"^(?:COL\.|CAPT\.|MR\.|MRS\.|LT\.|MAJ\.|GEN\.|DR\.)\s+", re.IGNORECASE)
_NAME_RE = re.compile(r"^([A-Za-z .&'\-]{3,}?),")


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _normalise_name(raw: str) -> str:
    """Strip titles, trim whitespace, title-case."""
    raw = _TITLE_RE.sub("", raw).strip()
    # take portion before comma if still present
    if "," in raw:
        raw = raw.split(",", 1)[0]
    return " ".join(w.capitalize() for w in raw.split())


def _extract_from_row(row: pd.Series) -> Tuple[Optional[str], Optional[float], Optional[int], Optional[str]]:
    """Return (name, acreage, year, excerpt) extracted from the row's string columns."""
    name: Optional[str] = None
    acreage: Optional[float] = None
    year: Optional[int] = None
    excerpt: Optional[str] = None

    for col, val in row.items():
        if not isinstance(val, str) or not val.strip():
            continue
        text = val.strip()
        # Acreage
        if acreage is None:
            m_ac = _ACRE_RE.search(text)
            if m_ac:
                try:
                    acreage = float(m_ac.group(1))
                except ValueError:
                    pass
                excerpt = text  # keep the first line that contains acreage
        # Year
        if year is None:
            m_yr = _YEAR_RE.search(text)
            if m_yr:
                try:
                    yr = int(m_yr.group(0)[:4])  # strip ? if present
                    if 1600 <= yr <= 1899:
                        year = yr
                except ValueError:
                    pass
        # Name
        if name is None:
            m_name = _NAME_RE.match(text)
            if m_name:
                name = _normalise_name(m_name.group(1))

        # Early exit if we have all three
        if name and acreage is not None and year is not None:
            break

    return name, acreage, year, excerpt


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def build_oss_grants_table(oss_path: Path = OSS_PATH, out_csv: Path = OUT_CSV) -> pd.DataFrame:
    """Parse the OSS polygon layer and write oss_grants.csv."""

    gdf = pyogrio.read_dataframe(oss_path, read_geometry=False)

    records: List[Dict[str, object]] = []
    misses = 0
    for idx, row in gdf.iterrows():
        name, acreage, year, excerpt = _extract_from_row(row)
        if name is None or acreage is None or year is None:
            misses += 1
        records.append({
            "objectid": row.OBJECTID if "OBJECTID" in row else idx,
            "name_std": name,
            "acreage": acreage,
            "year": year,
            "raw_excerpt": excerpt,
        })

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)

    extracted = len(df) - misses
    print(f"OSS grants parsed: {extracted}/{len(df)} rows with all three cues. → {out_csv.relative_to(config.ROOT_DIR)}")
    if misses:
        print(f"WARNING: {misses} rows missing at least one cue (left as null).")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    build_oss_grants_table() 