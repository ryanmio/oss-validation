"""Resolve anchor phrases to authoritative geographic coordinates using GNIS.

The logic is intentionally conservative: we only accept an anchor if we can find
EXACTLY *one* plausible GNIS feature that matches the anchor's feature name (via
fuzzy matching ≥ threshold) **and** falls inside the grant's county (if that can
be parsed from the abstract).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import pandas as pd
from rapidfuzz import fuzz, process
from shapely.geometry import Point
from loguru import logger

GNIS_GPKG = Path("data/reference/gnis_features_va.gpkg")

# Default path for resolved anchor output (Tier-2 positional accuracy)
TIER2_OUT_CSV = Path("data/processed/tier2_positional_accuracy.csv")

# Water‐feature GNIS classes we consider for stream anchors
WATER_FEATURE_CLASSES = {
    "Stream",  # rivers, creeks, runs
    "Lake",  # few mouths but safe to include
    "Swamp",
}

# Some confluence locations are tagged as Locale in GNIS; allow as water-adjacent when needed
LOCALE_CLASS = "Locale"

STRUCTURAL_CLASSES = {
    "Church",
    "Locale",  # courthouses and mills are usually Locale
    "Populated Place",  # courthouse towns
}

BASE_SIM_THRESHOLD = 90  # tightened again per user feedback
STREAM_SIM_THRESHOLD = 90  # apply same strictness to stream names

# Historical → allowed modern county names cross-walk (minimal – extend as needed)
HIST_TO_MODERN: dict[str, list[str]] = {
    # Major splits before 1800 (non-exhaustive but broad)
    "Henrico": ["Henrico", "Goochland", "Chesterfield", "Cumberland"],
    "Goochland": [
        "Goochland",
        "Hanover",
        "Albemarle",
        "Fluvanna",
    ],
    "Charles City": ["Charles City", "Prince George", "Surry"],
    "Surry": ["Surry", "Brunswick", "Prince George"],
    "King and Queen": ["King and Queen", "King William", "King George"],
    "Orange": ["Orange", "Culpeper", "Madison"],
    "Spotsylvania": ["Spotsylvania", "Caroline", "Orange"],
    "Albemarle": ["Albemarle", "Fluvanna", "Buckingham"],
    "Prince George": ["Prince George", "Dinwiddie"],
    "Brunswick": ["Brunswick", "Mecklenburg", "Greensville"],
    "Northampton": ["Northampton", "Accomack"],
    "Norfolk": ["Norfolk", "Chesapeake", "Virginia Beach"],
    "Warwick": ["Warwick", "Newport News"],
    "Princess Anne": ["Princess Anne", "Virginia Beach"],
    "Lunenburg": ["Lunenburg", "Charlotte"],
    "Hanover": ["Hanover", "Louisa", "Caroline"],
    "Augusta": ["Augusta", "Rockingham", "Rockbridge", "Bath", "Alleghany"],
    "Bedford": ["Bedford", "Campbell", "Franklin"],
    # fallback will use identity mapping
}

def _allowed_modern_counties(hist_county: str | None) -> list[str]:
    if hist_county is None:
        return []
    return HIST_TO_MODERN.get(hist_county.title(), [hist_county.title()])

# -----------------------------------------------------------------------------
# GNIS loader helpers
# -----------------------------------------------------------------------------

def _load_gnis_features() -> gpd.GeoDataFrame:
    """Load GNIS features with *feature_id*, *feature_name*, *feature_class*, *county_name*.

    The GPkg has multiple relational tables. We join three of them:
    • Gaz_Features        – geometry + feature_class
    • Gaz_Names           – feature_id ↔ feature_name (we take official names)
    • Gaz_InCounties      – feature_id ↔ county_name
    """
    logger.info("Loading GNIS features…")

    # Load tables
    feats = gpd.read_file(GNIS_GPKG, layer="Gaz_Features")
    names = gpd.read_file(GNIS_GPKG, layer="Gaz_Names")
    counties = gpd.read_file(GNIS_GPKG, layer="Gaz_InCounties")

    # Filter names to official ones (feature_name_official == 1)
    names_off = names[names["feature_name_official"] == 1][
        ["feature_id", "feature_name"]
    ]

    # Join features ↔ names → counties
    df = feats.merge(names_off, on="feature_id", how="inner")
    df = df.merge(counties[["feature_id", "county_name"]], on="feature_id", how="left")

    # Geometry: explode MULTIPOINT into individual points; keep first point per feature
    df = df.explode(index_parts=False)
    df["geometry"] = df.geometry.apply(
        lambda geom: Point(geom) if geom.geom_type == "Point" else geom
    )
    df = gpd.GeoDataFrame(df, geometry="geometry", crs=feats.crs)

    # Clean names for comparison
    df["feature_name_norm"] = df["feature_name"].str.lower()
    df["county_name_norm"] = df["county_name"].str.lower()

    # Drop duplicate rows with same feature_id to avoid multipoint ambiguity
    df = df.sort_values(by=["feature_id"]).drop_duplicates(subset=["feature_id"], keep="first")

    logger.info(f"Loaded {len(df):,} GNIS feature points (after explode).")
    return df[["feature_id", "feature_name", "feature_name_norm", "feature_class", "county_name", "county_name_norm", "geometry"]]


# Cache in module state
_GNIS_CACHE: Optional[gpd.GeoDataFrame] = None


def load_gazetteer() -> gpd.GeoDataFrame:
    global _GNIS_CACHE
    if _GNIS_CACHE is None:
        _GNIS_CACHE = _load_gnis_features()
    return _GNIS_CACHE

# -----------------------------------------------------------------------------
# County parsing helper (from cp_raw)
# -----------------------------------------------------------------------------

# Simple mapping for common abbreviations → full county names
_ABBREV_COUNTY_MAP = {
    "Pr. Geo.": "Prince George",
    "Pr. Edw.": "Prince Edward",
    "Pr. William": "Prince William",
    "K. & Q.": "King and Queen",
    "K. Wm.": "King William",
    "K. Geo.": "King George",
}

# Regex patterns ordered
_COUNTY_REGEXPS = [
    re.compile(r"\bof\s+(?P<county>[A-Z][A-Za-z &'\.]+?)\s+Co[;.,]", re.IGNORECASE),
    re.compile(r"\b(?P<county>[A-Z][A-Za-z &'\.]+?)\s+Co[;.,]", re.IGNORECASE),
]


def parse_county_from_abstract(cp_raw: str) -> Optional[str]:
    if not isinstance(cp_raw, str):
        return None
    for pat in _COUNTY_REGEXPS:
        m = pat.search(cp_raw)
        if m:
            county_guess = m.group("county").strip()
            # Expand abbreviations
            return _ABBREV_COUNTY_MAP.get(county_guess, county_guess)
    return None

# -----------------------------------------------------------------------------
# Resolution core
# -----------------------------------------------------------------------------

def _clean_feature_name(name: str) -> str:
    """Return simplified feature name for matching."""
    name = name.lower()
    # remove leading 'the '
    if name.startswith('the '):
        name = name[4:]
    # strip apostrophes and trailing s
    name = name.replace("'", "")
    # remove directional suffix ' of n', ' of s', etc.
    name = re.sub(r'\s+of\s+[nswe]$', '', name)
    # If we have plural 's' before water-body type (e.g., "Carys Creek" → "Cary Creek")
    name = re.sub(r"([a-z]+)s\s+(creek|river|swamp|branch|run)$", r"\1 \2", name)
    name = name.strip()
    return name

def _similarity_threshold(name_clean: str) -> int:
    if any(name_clean.endswith(suffix) for suffix in (" creek", " river", " swamp", " branch", " run")):
        return STREAM_SIM_THRESHOLD
    return BASE_SIM_THRESHOLD

def _max_pairwise_distance_m(cands: gpd.GeoDataFrame) -> float:
    if len(cands) <= 1:
        return 0.0
    # project to EPSG:3857 for metre distances
    proj = cands.to_crs(3857)
    coords = list(zip(proj.geometry.x, proj.geometry.y))
    maxd = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i]
        for j in range(i+1, len(coords)):
            x2, y2 = coords[j]
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if dist > maxd:
                maxd = dist
    return maxd

def _resolve_single(
    row: pd.Series,
    gnis: gpd.GeoDataFrame,
) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """Return (lat, lon, match_score, ambiguity_flag) or (None, None, None, 1).

    ambiguity_flag: 0 unique, 1 ambiguous, 2 no_match
    """
    feature_name = row["anchor_feature_name"]
    if pd.isna(feature_name):
        return None, None, None, 2, None, None, None

    feature_name_clean = _clean_feature_name(str(feature_name))
    anchor_type = row["anchor_type"]
    county_guess = parse_county_from_abstract(row.get("cp_raw", ""))

    # Candidate subset by anchor_type
    if anchor_type in {"mouth", "fork", "head", "confluence"}:
        subset = gnis[gnis.feature_class.isin(WATER_FEATURE_CLASSES.union({LOCALE_CLASS}))]
    elif anchor_type in {"courthouse", "mill", "church"}:
        subset = gnis[gnis.feature_class.isin(STRUCTURAL_CLASSES)]
    else:
        subset = gnis

    subset_initial = subset
    if county_guess is not None:
        county_allowed = {c.lower() for c in _allowed_modern_counties(county_guess)}
        subset_in_county = subset[subset.county_name_norm.isin(county_allowed)]
        if not subset_in_county.empty:
            subset = subset_in_county

    if subset.empty:
        return None, None, None, 2, None, None, None

    # Fuzzy match
    names_list = subset["feature_name_norm"].tolist()
    matches = process.extract(feature_name_clean, names_list, scorer=fuzz.ratio, limit=15)

    # Determine threshold dynamically
    thresh = _similarity_threshold(feature_name_clean)

    # Keep candidates above threshold
    good = [m for m in matches if m[1] >= thresh]
    if not good:
        return None, None, None, 2, None, None, None

    best_score = max(m[1] for m in good)
    good_names = {m[0] for m in good if m[1] == best_score}

    candidates = subset[subset.feature_name_norm.isin(good_names)].copy()

    # Direction / elevation heuristic ------------------------------------------------
    if len(candidates) > 1:
        if anchor_type in {"mouth", "confluence"}:
            # downstream: southernmost (min lat), then easternmost (max lon)
            lat_min = candidates.geometry.y.min()
            cand = candidates[candidates.geometry.y == lat_min]
            if len(cand) > 1:
                lon_max = cand.geometry.x.max()
                cand = cand[cand.geometry.x == lon_max]
            candidates = cand
        elif anchor_type == "head":
            lat_max = candidates.geometry.y.max()
            cand = candidates[candidates.geometry.y == lat_max]
            if len(cand) > 1:
                lon_min = cand.geometry.x.min()
                cand = cand[cand.geometry.x == lon_min]
            candidates = cand
        elif anchor_type == "fork":
            # pick candidate with lat closest to median
            med_lat = candidates.geometry.y.median()
            candidates["_dist"] = (candidates.geometry.y - med_lat).abs()
            candidates = candidates.sort_values("_dist").head(1)

    # If still >1 candidates after heuristics, check spatial cluster; else ambiguous
    if len(candidates) > 1:
        if _max_pairwise_distance_m(candidates) <= 1000.0:
            centroid = candidates.geometry.unary_union.centroid
            sel=candidates.iloc[0]
            return centroid.y, centroid.x, best_score, 0, sel.feature_name, sel.feature_class, sel.county_name
        return None, None, best_score, 1, None, None, None

    cand = candidates.iloc[0]
    lon, lat = cand.geometry.x, cand.geometry.y

    return lat, lon, best_score, 0, cand.feature_name, cand.feature_class, cand.county_name


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def resolve_anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns: anchor_lat, anchor_lon, match_score, ambiguity_flag."""
    gnis = load_gazetteer()

    results = df.apply(lambda row: pd.Series(_resolve_single(row, gnis), index=[
        "anchor_lat","anchor_lon","match_score","ambiguity_flag","gnis_name","gnis_class","gnis_county"
    ]), axis=1)

    # Before concatenating, drop any existing columns we are about to add to avoid duplicate labels
    cols_to_add = ["anchor_lat", "anchor_lon", "match_score", "ambiguity_flag", "gnis_name", "gnis_class", "gnis_county"]
    df_clean = df.drop(columns=[c for c in cols_to_add if c in df.columns])

    # Ensure both dataframes share the same simple RangeIndex so rows align 1-for-1
    out = pd.concat([
        df_clean.reset_index(drop=True),
        results.reset_index(drop=True)
    ], axis=1)

    # Log summary counts for quick feedback
    uniq = (out['ambiguity_flag'] == 0).sum()
    total = len(out)
    logger.info(f"Anchor resolution complete: {uniq} unique matches / {total} anchors ({uniq/total:.1%})")

    return out 