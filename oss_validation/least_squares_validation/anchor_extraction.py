"""Anchor phrase extraction from Cavaliers & Pioneers abstracts.

This module identifies uniquely locatable geographic "anchors" inside a grant
abstract (e.g., "mouth of Cary's Creek").

The extraction strategy is intentionally conservative: we only flag phrases
that match high-precision heuristic patterns. This ensures that downstream
resolution work is not overwhelmed by noisy candidates.

Public API
----------
extract_anchors(df: pd.DataFrame, text_col: str = "cp_raw") -> pd.DataFrame
    Return a new DataFrame with three additional columns:
    • anchor_phrase  – full text of the matched phrase (str | None)
    • anchor_type    – one of {"mouth", "fork", "confluence", "courthouse"} (str | None)
    • anchor_feature_name – canonical feature name without the cue phrase (str | None)
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# Pre-processing helpers
# -----------------------------------------------------------------------------

# Abbreviation replacements – ordered by decreasing key length so that Riv. is
# replaced before shorter tokens like Br.
_ABBREV_REPLACEMENTS = {
    r"\bRiv[\.]?\b": "River",
    r"\bCr[\.]?\b": "Creek",
    r"\bSw[\.]?\b": "Swamp",
    r"\bBr[\.]?\b": "Branch",
    r"\bForks?\b": "Fork",
    r"\bCh[\. ]?\b": "Church",
}

_WHITESPACE_RE = re.compile(r"\s+")
_HYPHEN_EOL_RE = re.compile(r"(\w)-\s+(\w)")
_CURLY_APOST_RE = {
    "’": "'",
    "‘": "'",
    "“": '"',
    "”": '"',
}


def _normalize_text(text: str) -> str:
    """Return a normalised version of *text* for pattern matching."""
    if not isinstance(text, str):
        return ""

    # 1) Standardise apostrophes and quotes
    for bad, good in _CURLY_APOST_RE.items():
        text = text.replace(bad, good)

    # 2) Collapse hyphenation across newlines / spaces
    text = _HYPHEN_EOL_RE.sub(r"\1\2", text)

    # 3) Replace newlines/tabs with single space then collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text.replace("\n", " ").replace("\t", " ")).strip()

    # 4) Expand abbreviations
    for pat, repl in _ABBREV_REPLACEMENTS.items():
        text = re.sub(pat, repl, text, flags=re.IGNORECASE)

    return text

# -----------------------------------------------------------------------------
# Anchor patterns (expanded)
# -----------------------------------------------------------------------------

# Water-body anchors (mouth / fork / confluence / head)
_WATER_BODY_PATTERN = re.compile(
    r"\b(?P<kind>mouth|fork|confluence|head)\s+(?:of\s+|on\s+the\s+)?"
    r"(?P<feature>[A-Z][A-Za-z'’\-]*(?:\s+[A-Z][A-Za-z'’\-]*){0,4})"
    r"(?:\s+(?P<ftype>River|Creek|Swamp|Branch|Fork|Run))?\b",
    flags=re.IGNORECASE,
)

# Confluence pattern with joins/flows into
_JOINS_PATTERN = re.compile(
    r"\b(?P<feature>[A-Z][A-Za-z'’\-]*(?:\s+[A-Z][A-Za-z'’\-]*){0,3})\s+"
    r"(?:River|Creek|Swamp|Branch|Fork|Run)\s+"
    r"(?:joins|flows\s+into)\s+"
    r"[A-Z][A-Za-z'’\-]*(?:\s+[A-Z][A-Za-z'’\-]*){0,3}\s+"
    r"(?:River|Creek|Swamp|Branch|Fork|Run)\b",
    flags=re.IGNORECASE,
)

# Inverted mouth order – "X Creek at its mouth"
_INVERTED_MOUTH_PATTERN = re.compile(
    r"\b(?P<feature>[A-Z][A-Za-z'’\-]*(?:\s+[A-Z][A-Za-z'’\-]*){0,3})\s+"
    r"(?P<ftype>River|Creek|Swamp|Branch|Fork|Run)\s+(?:at\s+its\s+)?mouth\b",
    flags=re.IGNORECASE,
)

# Courthouse / mill / church anchors
_STRUCTURAL_PATTERN = re.compile(
    r"\b(?:beg(?:inning)?\s+at|at)\s+the\s+(?:old\s+)?(?P<feature>(?:[A-Z][A-Za-z'’\-]*\s+)?(?:court\s*house|courthouse|mill|church))\b",
    flags=re.IGNORECASE,
)

# Ordered list of (regex, anchor_type) searched sequentially
ANCHOR_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (_WATER_BODY_PATTERN, "waterbody"),  # we derive specific kind later
    (_JOINS_PATTERN, "confluence"),
    (_INVERTED_MOUTH_PATTERN, "mouth"),
    (_STRUCTURAL_PATTERN, "structural"),
]

# -----------------------------------------------------------------------------
# Core extraction
# -----------------------------------------------------------------------------

def _extract_single(raw_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (anchor_phrase_raw, anchor_type, feature_name)."""

    if not isinstance(raw_text, str) or not raw_text.strip():
        return None, None, None

    norm = _normalize_text(raw_text)

    for pattern, high_level_type in ANCHOR_PATTERNS:
        match = pattern.search(norm)
        if match:
            phrase_norm = match.group(0).strip()
            # Attempt to locate the same phrase in raw text for debugging
            try:
                raw_match = re.search(re.escape(phrase_norm), raw_text, flags=re.IGNORECASE)
                phrase_raw = raw_match.group(0).strip() if raw_match else phrase_norm
            except re.error:
                phrase_raw = phrase_norm

            # Determine anchor_type
            if pattern is _WATER_BODY_PATTERN:
                anchor_type = match.group("kind").lower()
            elif pattern is _JOINS_PATTERN:
                anchor_type = "confluence"
            elif pattern is _INVERTED_MOUTH_PATTERN:
                anchor_type = "mouth"
            elif pattern is _STRUCTURAL_PATTERN:
                # feature might include courthouse/mill/church
                feat_lower = match.group("feature").lower()
                if "court" in feat_lower:
                    anchor_type = "courthouse"
                elif "mill" in feat_lower:
                    anchor_type = "mill"
                elif "church" in feat_lower:
                    anchor_type = "church"
                else:
                    anchor_type = "structural"
            else:
                anchor_type = high_level_type

            feature_name = match.group("feature").strip() if "feature" in match.groupdict() else None

            # Generic filter --------------------------------------------------
            _GENERIC_WORDS = {
                "branch",
                "creek",
                "river",
                "swamp",
                "run",
                "fork",
                "head",
                "mouth",
                "valley",
                "spring",
                "br",
                "brs",
                "gr",
                "bet",
            }

            def _is_generic(fname: str | None) -> bool:
                if fname is None:
                    return True
                tokens = [t.lower() for t in re.split(r"\s+", fname) if t]
                # If after stripping articles we have <2 tokens or first token is generic, discard
                meaningful = [t for t in tokens if t not in {"a", "the", "of"}]
                if len(meaningful) < 2:
                    return True
                if meaningful[0] in _GENERIC_WORDS:
                    return True
                return False

            if _is_generic(feature_name):
                return None, None, None  # discard generic anchor

            return phrase_raw, anchor_type, feature_name

    return None, None, None


def extract_anchors(
    df: pd.DataFrame,
    text_col: str = "cp_raw",
    add_normalised_col: bool = False,
) -> pd.DataFrame:
    """Detect anchor phrases with expanded heuristic logic.

    New columns added:
    • abstract_norm – normalised text (optional; large, so opt-in via *add_normalised_col*)
    • anchor_phrase_raw – slice from original text
    • anchor_phrase – normalised phrase
    • anchor_type
    • anchor_feature_name
    • match_score (placeholder)
    • ambiguity_flag (placeholder)
    """

    if text_col not in df.columns:
        raise KeyError(f"Column '{text_col}' not found in DataFrame")

    # Normalise abstracts in a vectorised way
    norm_series = df[text_col].apply(_normalize_text)

    extracted = df[text_col].apply(lambda txt: pd.Series(_extract_single(txt)))
    extracted.columns = ["anchor_phrase_raw", "anchor_type", "anchor_feature_name"]

    # Derive anchor_phrase (normalised version of raw slice)
    extracted["anchor_phrase"] = extracted["anchor_phrase_raw"].apply(
        lambda s: _normalize_text(s) if isinstance(s, str) else pd.NA
    )

    # Placeholders for downstream resolution
    extracted["match_score"] = pd.NA
    extracted["ambiguity_flag"] = pd.NA

    out = pd.concat([df.reset_index(drop=True), extracted], axis=1)
    if add_normalised_col:
        out["abstract_norm"] = norm_series
    return out 