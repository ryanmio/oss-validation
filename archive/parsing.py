"""Grant abstract parsing and cue extraction utilities.

This first cut focuses on *loading* Cavaliers & Pioneers Volume 3
abstracts and providing a clean DataFrame we can build on.

Later versions will fill in the actual spaCy-based cue extraction.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from oss_preprocessing import config

# ---------------------------------------------------------------------------
# Load abstracts
# ---------------------------------------------------------------------------

RAW_CSV = config.RAW_DIR / "CavaliersandPioneersvol3" / "raw_cavaliers_extract.csv"


def _read_raw_csv(path: Path = RAW_CSV) -> pd.DataFrame:  # noqa: N802
    """Read the raw Cavaliers & Pioneers CSV preserving embedded newlines."""

    # The file uses embedded line-feeds inside quoted "raw_entry" – we need
    # Python's csv engine (not "c" fast-parser).
    df = pd.read_csv(
        path,
        engine="python",
        quoting=csv.QUOTE_ALL,
        keep_default_na=False,
    )

    # Give each record a stable grant_id (volume-book-row).
    df = df.reset_index(drop=True)
    df["grant_id"] = df.apply(lambda r: f"{r.volume}_{r.book}_{r.name}", axis=1)
    # Re-order
    cols = ["grant_id"] + [c for c in df.columns if c != "grant_id"]
    return df[cols]


# ---------------------------------------------------------------------------
# Cue extraction – placeholder
# ---------------------------------------------------------------------------

# Very rough regex for capitalised words (we'll replace with spaCy later)
_CAP_RE = re.compile(r"\b[A-Z][A-Za-z'‐-]{2,}\b")


def extract_cues(text: str) -> List[str]:
    """Return a **placeholder** list of candidate place names.

    For now, this grabs unique capitalised words ≥ 3 chars that are not stop-words.
    The true implementation will use spaCy's NER and a gazetteer.
    """
    matches = set(_CAP_RE.findall(text))
    # Drop obvious non-placenames (units, abbreviations)
    blacklist = {"ACS", "CO", "RIV", "PERS", "NEGROES"}
    return sorted(m for m in matches if m.upper() not in blacklist)


def parse_abstracts(path: Path = RAW_CSV) -> pd.DataFrame:
    """High-level convenience that loads the CSV and explodes cues.

    Returns
    -------
    DataFrame with columns:
        grant_id, cue, raw_text
    """
    df = _read_raw_csv(path)
    records: list[dict[str, str]] = []
    for _, row in df.iterrows():
        cues = extract_cues(row.raw_entry)
        for cue in cues:
            records.append({
                "grant_id": row.grant_id,
                "cue": cue,
                "raw_text": row.raw_entry,
            })
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    out_dir = config.PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "abstract_cues.csv"

    df = parse_abstracts()
    df.to_csv(out_path, index=False)
    print(f"Parsed {len(df)} cues → {out_path.relative_to(config.ROOT_DIR)}")


if __name__ == "__main__":
    main() 