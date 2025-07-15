"""Spatial validation of matched OSS-CP grants.

For each matched grant pair:
1. Compute OSS polygon centroid
2. Test whether centroid falls within the county stated in the C&P abstract
3. Classify as exact_match, adjacent, or mismatch
4. Output results CSV and summary report

Usage:
    python -m oss_validation.spatial_validation
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from ...src.oss_preprocessing import config

# Input/output paths
MATCHED_CSV = config.PROCESSED_DIR / "matched_grants.csv"
CP_GRANTS_CSV = config.PROCESSED_DIR / "cp_grants.csv"
OSS_GEOJSON = config.RAW_DIR / "CentralVAPatents_PLY-shp" / "centralva.geojson"
TIGER_COUNTIES = config.REFERENCE_DIR / "tiger_va_counties_2023_from_us.gpkg"
OUT_CSV = config.PROCESSED_DIR / "spatial_validation.csv"
OUT_GEOJSON = config.PROCESSED_DIR / "spatial_validation_errors.geojson"
AHCB_GPKG = config.REFERENCE_DIR / "ahcb_va.gpkg"

# County name normalization mapping (C&P text -> TIGER NAME field)
COUNTY_NAME_MAP = {
    "Accomack": "Accomack",
    "Albemarle": "Albemarle", 
    "Amelia": "Amelia",
    "Brunswick": "Brunswick",
    "Caroline": "Caroline",
    "Charles City": "Charles City",
    "Chesterfield": "Chesterfield",
    "Cumberland": "Cumberland",
    "Dinwiddie": "Dinwiddie",
    "Essex": "Essex",
    "Frederick": "Frederick",
    "Gloucester": "Gloucester",
    "Goochland": "Goochland",
    "Hanover": "Hanover",
    "Henrico": "Henrico",
    "Isle Of Wight": "Isle of Wight",
    "James City": "James City",
    "King & Queen": "King and Queen",
    "King George": "King George",
    "King William": "King William",
    "Lancaster": "Lancaster",
    "Louisa": "Louisa",
    "Lunenburg": "Lunenburg",
    "Middlesex": "Middlesex",
    "Nansemond": "Nansemond",  # Historical - now part of Suffolk
    "New Kent": "New Kent",
    "Norfolk": "Norfolk",
    "Northampton": "Northampton",
    "Northumberland": "Northumberland",
    "Orange": "Orange",
    "Prince George": "Prince George",
    "Prince William": "Prince William",
    "Princess Anne": "Princess Anne",  # Historical - now Virginia Beach
    "Richmond": "Richmond",
    "Spotsylvania": "Spotsylvania",
    "Stafford": "Stafford",
    "Surry": "Surry",
    "Sussex": "Sussex",
    "Warwick": "Warwick",  # Historical - now part of Newport News
    "Westmoreland": "Westmoreland",
    "York": "York"
}

# ---------------------------------------------------------------------------
# Split events for near-split detection (parent -> child, split_year)
# ---------------------------------------------------------------------------

SPLIT_EVENTS = [
    ("King and Queen", "King William", 1702),
    ("Henrico", "Goochland", 1728),
    ("Goochland", "Hanover", 1721),
    ("Charles City", "Prince George", 1703),
    ("Surry", "Brunswick", 1720),
]

def _is_near_split(parent: str | None, child: str | None, year: int) -> bool:
    if not parent or not child:
        return False
    parent = parent.title().strip()
    child = child.title().strip()
    for p, c, y in SPLIT_EVENTS:
        if ((parent == p and child == c) or (parent == c and child == p)) and abs(year - y) <= 2:
            return True
    return False

def normalize_county_name(cp_county: str | float | None) -> Optional[str]:
    """Normalize C&P county name from C&P table to TIGER standard."""
    if cp_county is None or isinstance(cp_county, float):
        return None
    cp_county = str(cp_county).strip()
    return COUNTY_NAME_MAP.get(cp_county)


def load_oss_polygons() -> gpd.GeoDataFrame:
    """Load OSS polygons and reproject to planar CRS."""
    print(f"Loading OSS polygons from {OSS_GEOJSON.relative_to(config.ROOT_DIR)}...")
    gdf = gpd.read_file(OSS_GEOJSON)
    
    # Ensure we have the OBJECTID field
    if "OBJECTID" not in gdf.columns:
        raise ValueError("OSS polygon file missing OBJECTID field")
    
    # Reproject to lat/lon CRS for reliable joins with TIGER (EPSG:4326)
    if gdf.crs != config.CRS_LATLON:
        print(f"Reprojecting OSS polygons from {gdf.crs} to {config.CRS_LATLON}")
        gdf = gdf.to_crs(config.CRS_LATLON)
    
    return gdf


def load_tiger_counties() -> gpd.GeoDataFrame:
    """Load TIGER county boundaries."""
    print(f"Loading TIGER counties from {TIGER_COUNTIES.relative_to(config.ROOT_DIR)}...")
    gdf = gpd.read_file(TIGER_COUNTIES)
    
    # Reproject to lat/lon (EPSG:4326)
    if gdf.crs != config.CRS_LATLON:
        print(f"Reprojecting counties from {gdf.crs} to {config.CRS_LATLON}")
        gdf = gdf.to_crs(config.CRS_LATLON)
    
    return gdf


def compute_centroids(oss_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute representative points for OSS polygons (always inside)."""
    print("Computing representative points (centroids)…")
    centroids = oss_gdf.copy()
    centroids["geometry"] = centroids.geometry.representative_point()
    
    # Ensure oss_id is string for consistent joins with matched_df
    centroids.rename(columns={"OBJECTID": "oss_id"}, inplace=True)
    centroids["oss_id"] = centroids["oss_id"].astype(str)
    
    # Keep only id + geometry
    centroids = centroids[["oss_id", "geometry"]].copy()
    
    return centroids


def spatial_join_counties(centroids: gpd.GeoDataFrame, counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Spatial join centroids with counties."""
    print("Performing spatial join with counties...")
    
    # Spatial join - each centroid gets the county it falls within
    joined = gpd.sjoin(centroids, counties, how="left", predicate="within")
    
    # Keep relevant columns
    result_cols = ["oss_id", "NAME", "NAMELSAD", "geometry"]
    available_cols = [col for col in result_cols if col in joined.columns]
    joined = joined[available_cols].copy()
    
    if "NAMELSAD" in joined.columns:
        joined.rename(columns={"NAMELSAD": "tiger_county"}, inplace=True)
    elif "NAME" in joined.columns:
        joined.rename(columns={"NAME": "tiger_county"}, inplace=True)
    else:
        joined["tiger_county"] = None
    
    # Strip suffixes like " County" or " city" for easier comparison
    joined["tiger_county"] = joined["tiger_county"].str.replace(r"\s+(County|city)$", "", regex=True)
    
    return joined


def build_adjacency_graph(counties: gpd.GeoDataFrame) -> Dict[str, List[str]]:
    """Build adjacency graph of counties that share borders."""
    print("Building county adjacency graph...")
    
    adjacency = {}
    county_names = counties["NAME"].tolist() if "NAME" in counties.columns else []
    
    for i, county_a in counties.iterrows():
        name_a = county_a.get("NAME", f"County_{i}")
        adjacency[name_a] = []
        
        for j, county_b in counties.iterrows():
            if i != j:
                name_b = county_b.get("NAME", f"County_{j}")
                # Check if geometries touch (share a border)
                if county_a.geometry.touches(county_b.geometry):
                    adjacency[name_a].append(name_b)
    
    return adjacency


def _load_ahcb() -> gpd.GeoDataFrame:
    """Load the AHCB historical county polygons (VA)."""
    if not AHCB_GPKG.exists():
        raise FileNotFoundError(
            "AHCB GeoPackage not found. Run oss_validation.download_reference to fetch it."
        )

    gdf = gpd.read_file(AHCB_GPKG)
    if gdf.crs != config.CRS_LATLON:
        gdf = gdf.to_crs(config.CRS_LATLON)

    # Determine field names for start/end year and county name
    name_col = None
    for cand in ("COUNTYNAME", "NAME", "COUNTY_NAM", "COUNTY"):  # fallback list
        if cand in gdf.columns:
            name_col = cand
            break
    if name_col is None:
        raise ValueError("Could not locate county name column in AHCB layer")

    gdf = gdf.rename(columns={name_col: "hist_county"})

    # Parse years --------------------------------------------------------
    def _year_from(val: str | int | float) -> int:
        s = str(val)
        return int(s[:4]) if s and s[:4].isdigit() else 0

    start_col = next((c for c in gdf.columns if c.lower().startswith("start")), None)
    end_col = next((c for c in gdf.columns if c.lower().startswith("end")), None)
    if start_col is None or end_col is None:
        raise ValueError("Could not locate START/END date columns in AHCB layer")

    gdf["start_year"] = gdf[start_col].apply(_year_from)
    gdf["end_year"] = gdf[end_col].apply(_year_from)

    # Strip suffixes for comparisons
    gdf["hist_county_clean"] = (
        gdf["hist_county"].str.replace(r"\s+(County|city)$", "", regex=True).str.strip()
    )

    # Keep essential cols only
    return gdf[["hist_county", "hist_county_clean", "start_year", "end_year", "geometry"]]


def _clean_simple(name: str | float | None) -> str:
    if name is None or isinstance(name, float):
        return ""
    return (
        str(name)
        .replace("County", "")
        .replace("county", "")
        .replace("city", "")
        .replace("City", "")
        .strip()
    )


def classify_validation_results(
    matched_df: pd.DataFrame,
    spatial_df: gpd.GeoDataFrame,
    adjacency: Dict[str, List[str]],
    ahcb_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Classify each matched grant as exact_match, adjacent, or mismatch."""
    print("Classifying validation results...")
    
    # Load C&P grants to get county information
    print(f"Loading C&P grants for county info from {CP_GRANTS_CSV.relative_to(config.ROOT_DIR)}...")
    cp_grants = pd.read_csv(CP_GRANTS_CSV, dtype={"grant_id": str})
    
    # Merge matched grants with C&P county info
    matched_with_county = matched_df.merge(
        cp_grants[["grant_id", "county_text"]], 
        left_on="cp_id", 
        right_on="grant_id", 
        how="left"
    )
    
    results = []
    
    for _, match in matched_with_county.iterrows():
        oss_id = str(match["oss_id"])
        cp_county = match.get("county_text", "")  # County from C&P abstract
        year = int(match.get("year_cp", match.get("year_oss", 0)))
        
        # Find spatial join result for this OSS polygon
        spatial_row = spatial_df[spatial_df["oss_id"] == oss_id]
        
        if spatial_row.empty:
            # Centroid didn't fall in any county (water, etc.)
            result = {
                "oss_id": oss_id,
                "cp_id": match["cp_id"],
                "confidence": match["confidence"],
                "cp_county": cp_county,
                "tiger_county": None,
                "classification": "no_county",
                "notes": "Centroid outside all counties"
            }
        else:
            # First, year-aware historical lookup -----------------------
            point = spatial_row.iloc[0].geometry
            subset = ahcb_gdf[(ahcb_gdf["start_year"] <= year) & (ahcb_gdf["end_year"] >= year)]
            hit = subset[subset.contains(point)]
            hist_county = None
            if not hit.empty:
                hist_county = hit.iloc[0]["hist_county_clean"]

            cp_clean = _clean_simple(cp_county)

            if hist_county and cp_clean and hist_county.lower() == cp_clean.lower():
                classification = "exact_hist"
                notes = "Historical county match"
            elif hist_county and _is_near_split(hist_county, cp_clean, year):
                classification = "near_split"
                notes = f"Near-split {hist_county} ⇄ {cp_clean} (±2 yr)"
            else:
                # Fall back to modern comparison ----------------------
                tiger_county = spatial_row.iloc[0]["tiger_county"]
                cp_county_norm = normalize_county_name(cp_county)

                if cp_county_norm == tiger_county:
                    classification = "exact_modern"
                    notes = "Centroid in stated county (modern)"
                elif tiger_county and cp_county_norm and tiger_county in adjacency.get(cp_county_norm, []):
                    classification = "adjacent_modern"
                    notes = "Centroid in adjacent county (modern)"
                else:
                    # Boundary tolerance? compute distance to hist polygon
                    tol_m = 50.0
                    dist_ok = False
                    if not hit.empty:
                        poly = hit.iloc[0].geometry
                        # reproject to metres
                        poly_m = gpd.GeoSeries([poly], crs=config.CRS_LATLON).to_crs(config.CRS_DISTANCE)[0]
                        pt_m = gpd.GeoSeries([point], crs=config.CRS_LATLON).to_crs(config.CRS_DISTANCE)[0]
                        if pt_m.distance(poly_m) <= tol_m:
                            dist_ok = True
                    if dist_ok:
                        classification = "boundary_tol"
                        notes = f"≤{tol_m} m outside hist polygon ({hist_county})"
                    elif hist_county and _is_near_split(cp_clean, hist_county, year):
                        classification = "near_split"
                        notes = f"Near-split {cp_clean} ⇄ {hist_county} (±2 yr)"
                    else:
                        classification = "mismatch"
                        notes = (
                            f"Hist={hist_county or 'None'}; modern={tiger_county}; stated={cp_county_norm}"
                        )
            
            result = {
                "oss_id": oss_id,
                "cp_id": match["cp_id"],
                "confidence": match["confidence"],
                "cp_county": cp_county,
                "tiger_county": spatial_row.iloc[0]["tiger_county"],
                "classification": classification,
                "notes": notes
            }
        
        results.append(result)
    
    return pd.DataFrame(results)


def generate_error_geojson(results_df: pd.DataFrame, centroids: gpd.GeoDataFrame) -> None:
    """Generate GeoJSON of mismatch cases for visual QA."""
    print("Generating error GeoJSON...")
    
    # Filter to mismatch cases only
    errors = results_df[results_df["classification"] == "mismatch"].copy()
    
    if errors.empty:
        print("No mismatch cases found - skipping error GeoJSON")
        return
    
    # Join with centroid geometries
    error_gdf = errors.merge(centroids, on="oss_id", how="left")
    error_gdf = gpd.GeoDataFrame(error_gdf, geometry="geometry")
    
    # Reproject to lat/lon for GeoJSON
    if error_gdf.crs != config.CRS_LATLON:
        error_gdf = error_gdf.to_crs(config.CRS_LATLON)
    
    # Write GeoJSON
    error_gdf.to_file(OUT_GEOJSON, driver="GeoJSON")
    print(f"Error GeoJSON written → {OUT_GEOJSON.relative_to(config.ROOT_DIR)}")


def generate_summary_report(results_df: pd.DataFrame) -> None:
    """Generate markdown summary report."""
    print("Generating summary report...")
    
    total = len(results_df)
    exact_hist = len(results_df[results_df["classification"] == "exact_hist"])
    boundary_tol = len(results_df[results_df["classification"] == "boundary_tol"])
    near_split = len(results_df[results_df["classification"] == "near_split"])
    exact_modern = len(results_df[results_df["classification"] == "exact_modern"])
    adjacent_modern = len(results_df[results_df["classification"] == "adjacent_modern"])
    mismatch = len(results_df[results_df["classification"] == "mismatch"])
    no_county = len(results_df[results_df["classification"] == "no_county"])
    
    # Calculate percentages
    pct_exact_hist = (exact_hist / total * 100) if total > 0 else 0
    pct_boundary = (boundary_tol / total * 100) if total > 0 else 0
    pct_near = (near_split / total * 100) if total > 0 else 0
    pct_exact_mod = (exact_modern / total * 100) if total > 0 else 0
    pct_adj_mod = (adjacent_modern / total * 100) if total > 0 else 0
    pct_mismatch = (mismatch / total * 100) if total > 0 else 0
    pct_no_county = (no_county / total * 100) if total > 0 else 0
    
    # Confidence stats for each category
    conf_stats = {}
    for category in ["exact_hist", "exact_modern", "adjacent_modern", "mismatch", "no_county"]:
        subset = results_df[results_df["classification"] == category]
        if not subset.empty and "confidence" in subset.columns:
            conf_stats[category] = {
                "count": len(subset),
                "mean": subset["confidence"].mean(),
                "median": subset["confidence"].median(),
                "min": subset["confidence"].min(),
                "max": subset["confidence"].max()
            }
        else:
            conf_stats[category] = {"count": 0, "mean": 0, "median": 0, "min": 0, "max": 0}
    
    # Write markdown report
    report_path = config.REPORTS_DIR / "spatial_validation.md"
    
    lines = [
        "# Spatial Validation Summary",
        "",
        f"Total matched grants validated: {total}",
        "",
        "## Classification Results",
        "",
        f"- **Exact (historical)**: {exact_hist:4d} ({pct_exact_hist:5.1f}%) - Point in county valid for grant year",
        f"- **Boundary tol**:       {boundary_tol:4d} ({pct_boundary:5.1f}%) - ≤50 m from polygon edge",
        f"- **Near split**:         {near_split:4d} ({pct_near:5.1f}%) - In county swapped within ±2 yr",
        f"- **Exact (modern)**:     {exact_modern:4d} ({pct_exact_mod:5.1f}%) - Matches modern county (no split)",
        f"- **Adjacent (modern)**:  {adjacent_modern:4d} ({pct_adj_mod:5.1f}%) - In modern neighbour (likely split)",
        f"- **Mismatch**:           {mismatch:4d} ({pct_mismatch:5.1f}%) - No reasonable match",
        f"- **No county**:          {no_county:4d} ({pct_no_county:5.1f}%) - Point outside all polygons",
        "",
        "## Confidence Statistics by Category",
        ""
    ]
    
    for category, stats in conf_stats.items():
        if stats["count"] > 0:
            lines.extend([
                f"### {category.replace('_', ' ').title()}",
                f"- Count: {stats['count']}",
                f"- Mean confidence: {stats['mean']:.1f}",
                f"- Median confidence: {stats['median']:.1f}",
                f"- Range: {stats['min']:.1f} - {stats['max']:.1f}",
                ""
            ])
    
    lines.extend([
        "## Files Generated",
        "",
        f"- Results CSV: `{OUT_CSV.relative_to(config.ROOT_DIR)}`",
        f"- Error GeoJSON: `{OUT_GEOJSON.relative_to(config.ROOT_DIR)}`",
        f"- This report: `{report_path.relative_to(config.ROOT_DIR)}`"
    ])
    
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary report written → {report_path.relative_to(config.ROOT_DIR)}")


def run_spatial_validation() -> pd.DataFrame:
    """Run the complete spatial validation pipeline."""
    print("\n=== Spatial Validation Pipeline ===")
    
    # Load matched grants
    print(f"Loading matched grants from {MATCHED_CSV.relative_to(config.ROOT_DIR)}...")
    matched_df = pd.read_csv(MATCHED_CSV, dtype={"oss_id": str, "cp_id": str})
    print(f"Loaded {len(matched_df)} matched grants")
    
    # Load spatial data
    oss_gdf = load_oss_polygons()
    counties_gdf = load_tiger_counties()
    
    # Compute centroids
    centroids = compute_centroids(oss_gdf)
    
    # Spatial join with counties
    spatial_results = spatial_join_counties(centroids, counties_gdf)
    
    # Build adjacency graph
    adjacency = build_adjacency_graph(counties_gdf)
    
    # Load AHCB data
    ahcb_gdf = _load_ahcb()
    
    # Classify results
    results_df = classify_validation_results(matched_df, spatial_results, adjacency, ahcb_gdf)
    
    # Write results CSV
    results_df.to_csv(OUT_CSV, index=False)
    print(f"Results CSV written → {OUT_CSV.relative_to(config.ROOT_DIR)}")
    
    # Generate error GeoJSON (mismatches only)
    generate_error_geojson(results_df, centroids)
    
    # Generate summary report
    generate_summary_report(results_df)
    
    return results_df


if __name__ == "__main__":
    run_spatial_validation() 