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
from shapely.ops import unary_union
from pyproj import Geod

# For distance calculations when centroid coordinates are available
_GEOD = Geod(ellps="WGS84")

def _gc_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (km) between two WGS84 points."""
    if None in (lat1, lon1, lat2, lon2):
        return float("nan")
    _, _, dist_m = _GEOD.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0

from oss_validation.county_accuracy_validation import spatial_validation as sv
from oss_preprocessing.hist_modern_overlap import get_overlap_mapping
from oss_preprocessing import config

# Load modern county polygons once
_TIGER_COUNTIES = None
_COUNTY_GEOMS = {}

def _load_county_geoms():
    """Return mapping of lowercase county name → geometry using shared loader.

    Reuse spatial_validation.load_tiger_counties so the TIGER GeoPackage is
    read exactly once across the whole project instead of re-reading it in
    multiple modules.
    """
    global _COUNTY_GEOMS
    if _COUNTY_GEOMS:
        return _COUNTY_GEOMS

    gdf = sv.load_tiger_counties()[["NAME", "geometry"]]
    _COUNTY_GEOMS = {row.NAME.lower(): row.geometry for _, row in gdf.iterrows()}
    return _COUNTY_GEOMS

_BUFFER_DEG = 0.20  # ~20 km buffer for county containment near VA

def _point_within_counties(pt: Point, counties: list[str]) -> bool:
    geoms = _load_county_geoms()
    for c in counties:
        geom = geoms.get(c.lower())
        if geom is not None:
            if pt.within(geom) or pt.distance(geom) <= _BUFFER_DEG:
                return True
    return False

GNIS_GPKG = Path("data/reference/gnis_features_va.gpkg")

# Default path for resolved anchor output (Tier-2 positional accuracy)
TIER2_OUT_CSV = Path("data/processed/tier2_positional_accuracy.csv")

# Water‐feature GNIS classes we consider for stream anchors
WATER_FEATURE_CLASSES = {
    "Stream",  # rivers, creeks, runs
    "Lake",  # few mouths but safe to include
    "Swamp",
    "Valley",  # some forks and heads recorded as valleys
}

# Some confluence locations are tagged as Locale in GNIS; allow as water-adjacent when needed
LOCALE_CLASS = "Locale"

STRUCTURAL_CLASSES = {
    "Church",
    "Locale",  # courthouses and mills are usually Locale
    "Populated Place",  # courthouse towns
}

BASE_SIM_THRESHOLD = 90  # tightened again per user feedback
STREAM_SIM_THRESHOLD = 85  # tighten to reduce mismatches

_OVERLAP_MAP = None


def _allowed_modern_counties(hist_county: str | None) -> list[str]:
    """Return list of modern county names overlapping *hist_county*."""
    global _OVERLAP_MAP
    if hist_county is None:
        return []
    if _OVERLAP_MAP is None:
        _OVERLAP_MAP = get_overlap_mapping()
    return _OVERLAP_MAP.get(hist_county.title(), [hist_county.title()])

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

    # Do NOT drop duplicates: for streams crossing multiple counties keep one
    # point per county so our later county‐containment filter can choose the
    # appropriate segment.  Performance impact is negligible (<30 k extra
    # points).
    df = df.sort_values(by=["feature_id"])  # keep natural order but retain duplicates

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
    
    # normalize curly quotes and apostrophes
    name = name.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
    
    # handle possessives: "cary's creek" -> "cary creek", "jones's run" -> "jones run"  
    name = re.sub(r"'s?\s+(creek|river|branch|run|swamp|spring|mill|pond)", r" \1", name)
    
    # strip remaining apostrophes
    name = name.replace("'", "")
    
    # remove directional suffix ' of n', ' of s', etc.
    name = re.sub(r'\s+of\s+[nswe]$', '', name)
    
    # normalize common abbreviations
    name = re.sub(r'\bcr\.?\b', 'creek', name)
    name = re.sub(r'\bbr\.?\b', 'branch', name)
    name = re.sub(r'\bmtn?\b', 'mountain', name)
    
    name = name.strip()
    return name

def _similarity_threshold(name_clean: str) -> int:
    if any(name_clean.endswith(suffix) for suffix in (" creek", " river", " swamp", " branch", " run")):
        return 78  # lower threshold for water features to catch variations
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

# Manually identified bad fuzzy matches: (anchor_clean, candidate_clean) to reject
BAD_MATCH_PAIRS = {
    ("hensons creek", "johnsons creek"),
    ("holly branch", "polly branch"),
    ("pocoson branch", "poston branch"),
    ("fort branch", "ford branch"),
    ("myery branch", "merry branch"),
    ("brushey branch", "rush branch"),
    ("indian spring branch", "finney spring branch"),
    ("haynes spring branch", "hardy spring branch"),
    ("youl branch", "yokel branch"),
    ("gentrys branch", "gent branch"),
}

# Return now includes exclude_reason (str | None) at the end.
def _resolve_single(
    row: pd.Series,
    gnis: gpd.GeoDataFrame,
)-> Tuple[
    Optional[float],  # anchor_lat
    Optional[float],  # anchor_lon
    Optional[int],    # match_score
    int,              # ambiguity_flag
    Optional[str],    # gnis_name
    Optional[str],    # gnis_class
    Optional[str],    # gnis_county
    Optional[str],    # anchor_quality
    Optional[str],    # exclude_reason (diagnostics)
]:
    """Return (lat, lon, match_score, ambiguity_flag,
    gnis_name, gnis_class, gnis_county, anchor_quality).

    anchor_quality: "exact_locale", "exact", "estimated_downstream", "estimated_upstream", "estimated_fork" or None
    ambiguity_flag: 0 unique, 1 ambiguous, 2 no_match
    """
    feature_name = row["anchor_feature_name"]
    if pd.isna(feature_name):
        return None, None, None, 2, None, None, None, None, "missing_feature_name"

    feature_name_clean = _clean_feature_name(str(feature_name))
    anchor_type = row["anchor_type"]
    county_guess = row.get("cp_county")
    # Treat NaN/None/float as missing
    if isinstance(county_guess, float) or pd.isna(county_guess):
        county_guess = None
    if not county_guess:
        county_guess = parse_county_from_abstract(row.get("cp_raw", ""))

    # Candidate subset by anchor_type
    if anchor_type in {"mouth", "fork", "head", "confluence"}:
        subset = gnis[gnis.feature_class.isin(WATER_FEATURE_CLASSES.union({LOCALE_CLASS}))]
    elif anchor_type in {"courthouse", "mill", "church"}:
        subset = gnis[gnis.feature_class.isin(STRUCTURAL_CLASSES)]
    else:
        subset = gnis

    if subset.empty:
        return None, None, None, 2, None, None, None, None, "no_candidates_subtype"

    # ------------------------------------------------------------------
    # 1. Fuzzy match across the *full* subset (no county filter yet)
    # ------------------------------------------------------------------
    names_list = subset["feature_name_norm"].tolist()
    matches = process.extract(feature_name_clean, names_list, scorer=fuzz.ratio, limit=15)

    # Determine threshold dynamically
    thresh = _similarity_threshold(feature_name_clean)

    # Keep candidates above threshold
    good = [m for m in matches if m[1] >= thresh]
    if not good:
        return None, None, None, 2, None, None, None, None, "low_similarity"

    best_score = max(m[1] for m in good)
    good_names = {m[0] for m in good if m[1] == best_score}

    candidates = subset[subset.feature_name_norm.isin(good_names)].copy()

    # ------------------------------------------------------------------
    # 1b. Prefer precise 'Locale' confluence/head features if present
    # ------------------------------------------------------------------
    if anchor_type in {"mouth", "confluence", "fork", "head"}:
        if anchor_type in {"mouth", "confluence"}:
            keywords = ["mouth", "confluence"]
        elif anchor_type == "fork":
            keywords = ["fork", "forks"]
        elif anchor_type == "head":
            keywords = ["head", "source", "spring"]
        
        # Look for Locale features containing relevant keywords
        loc_mask = (candidates.feature_class == LOCALE_CLASS) & (
            candidates.feature_name_norm.str.contains("|".join(keywords), case=False, na=False)
        )
        loc_cand = candidates[loc_mask]
        if not loc_cand.empty:
            candidates = loc_cand.copy()

    # ------------------------------------------------------------------
    # 2. County filter *after* fuzzy, but only if it retains >=1 candidate
    # ------------------------------------------------------------------
    if county_guess is not None and not candidates.empty:
        county_allowed = _allowed_modern_counties(county_guess)
        # spatial containment test (within or ≤5 km of county)
        mask_contain = candidates.geometry.apply(lambda g: _point_within_counties(g, county_allowed))
        if mask_contain.any():
            candidates = candidates[mask_contain].copy()
        else:
            # keep all for diagnostics but mark excluded
            sel = candidates.iloc[0]
            return (
                sel.geometry.y,
                sel.geometry.x,
                best_score,
                2,
                sel.feature_name,
                sel.feature_class,
                sel.county_name,
                None,
                "outside_county",
            )

    # ------------------------------------------------------------------
    # 2b. Tie-break by highest Levenshtein similarity (local)
    # ------------------------------------------------------------------
    if len(candidates) > 1:
        candidates["_sim"] = candidates.feature_name_norm.apply(lambda n: fuzz.ratio(feature_name_clean, n))
        max_sim = candidates["_sim"].max()
        candidates = candidates[candidates._sim == max_sim].copy()

    # ------------------------------------------------------------------
    # 2c. If centroid coords available pick nearest to centroid
    # ------------------------------------------------------------------
    if len(candidates) > 1 and row.get("centroid_lat") is not None and not pd.isna(row.get("centroid_lat")):
        cen_lat = row["centroid_lat"]
        cen_lon = row["centroid_lon"]
        candidates["_d_km"] = candidates.geometry.apply(lambda g: _gc_km(g.y, g.x, cen_lat, cen_lon))
        min_d = candidates["_d_km"].min()
        candidates = candidates[candidates._d_km == min_d].copy()

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
            return centroid.y, centroid.x, best_score, 0, sel.feature_name, sel.feature_class, sel.county_name, "exact", None
        return None, None, best_score, 1, None, None, None, None, "cluster_ambiguous"

    # after determining cand
    cand = candidates.iloc[0]
    # manual blacklist check
    if (feature_name_clean, cand.feature_name_norm) in BAD_MATCH_PAIRS:
        return None, None, None, 2, None, None, None, None, "blacklist_match"

    # ------------------------------------------------------------------
    # Determine final coordinate & quality
    # ------------------------------------------------------------------
    anchor_quality = "exact"

    if cand.feature_class == "Locale" and any(word in cand.feature_name.lower() for word in ("mouth", "fork")):
        anchor_quality = "exact_locale"
    elif cand.feature_class == "Stream":
        same_stream = gnis[(gnis.feature_name_norm == cand.feature_name_norm) & (gnis.feature_class == "Stream")].copy()
        # Retain only points that fall inside the modern counties overlapping the
        # historical county; this removes far-away duplicate stream names.
        if county_guess is not None:
            allowed = _allowed_modern_counties(county_guess)
            same_stream = same_stream[same_stream.geometry.apply(lambda g: _point_within_counties(g, allowed))].copy()
        if not same_stream.empty and anchor_type in {"mouth", "confluence", "head", "fork"}:
            same_stream["_lat"] = same_stream.geometry.y
            same_stream["_lon"] = same_stream.geometry.x

            if anchor_type in {"mouth", "confluence"}:
                # downstream-most: min lat, then max lon
                sel = same_stream.sort_values(["_lat", "_lon"]).iloc[0]
                anchor_quality = "estimated_downstream"
            elif anchor_type == "head":
                # upstream-most: max lat, then min lon
                sel = same_stream.sort_values(["_lat", "_lon"]).iloc[-1]
                anchor_quality = "estimated_upstream"
            elif anchor_type == "fork":
                med_lat = same_stream["_lat"].median()
                same_stream["_dist"] = (same_stream["_lat"] - med_lat).abs()
                sel = same_stream.sort_values("_dist").iloc[0]
                anchor_quality = "estimated_fork"
            lon, lat = sel.geometry.x, sel.geometry.y
        else:
            lon, lat = cand.geometry.x, cand.geometry.y
    else:
        lon, lat = cand.geometry.x, cand.geometry.y

    # Optional: flag if still very far from centroid (>50 km) for later diagnostics
    excl_reason = None
    if row.get("centroid_lat") is not None and not pd.isna(row.get("centroid_lat")):
        d_km = _gc_km(lat, lon, row["centroid_lat"], row["centroid_lon"])
        if d_km > 20:
            excl_reason = f'far_centroid_{d_km:.0f}km'

    # Tighten threshold
    thresh = 92 if any(s in feature_name_clean for s in ('creek', 'river', 'branch')) else BASE_SIM_THRESHOLD

    return (
        lat,
        lon,
        best_score,
        0,
        cand.feature_name,
        cand.feature_class,
        cand.county_name,
        anchor_quality,
        excl_reason,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def resolve_anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns: anchor_lat, anchor_lon, match_score, ambiguity_flag."""
    gnis = load_gazetteer()

    results = df.apply(lambda row: pd.Series(_resolve_single(row, gnis), index=[
        "anchor_lat",
        "anchor_lon",
        "match_score",
        "ambiguity_flag",
        "gnis_name",
        "gnis_class",
        "gnis_county",
        "anchor_quality",
        "exclude_reason",
    ]), axis=1)

    # Before concatenating, drop any existing columns we are about to add to avoid duplicate labels
    cols_to_add = ["anchor_lat", "anchor_lon", "match_score", "ambiguity_flag", "gnis_name", "gnis_class", "gnis_county", "anchor_quality"]
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