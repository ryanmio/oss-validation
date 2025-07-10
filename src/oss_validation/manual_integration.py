# -*- coding: utf-8 -*-
"""Integrate manually located anchor coordinates into the Tier 2 pipeline.

This script takes the completed manual_anchor_worksheet.csv with filled-in
coordinates and merges them back into the tier2_positional_accuracy.csv,
recomputes distances, and generates final statistics.
"""
from __future__ import annotations

import re
from pathlib import Path
import logging

import pandas as pd
import numpy as np
from pyproj import Geod

from . import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MANUAL_WORKSHEET = config.PROCESSED_DIR / "manual_anchor_worksheet.csv"
TIER2_CSV = config.PROCESSED_DIR / "tier2_positional_accuracy.csv"
FINAL_CSV = config.PROCESSED_DIR / "tier2_final_with_manual.csv"

GEOD = Geod(ellps="WGS84")


def parse_manual_coordinates(coord_str: str) -> tuple[float | None, float | None]:
    """Parse lat,lon from manual entry. Returns (lat, lon) or (None, None)."""
    if pd.isna(coord_str) or not isinstance(coord_str, str) or not coord_str.strip():
        return None, None
    
    coord_str = coord_str.strip()
    
    # Try decimal degrees: "37.123456, -77.654321"
    match = re.match(r'^(-?\d+\.?\d*),?\s*(-?\d+\.?\d*)$', coord_str)
    if match:
        try:
            lat, lon = float(match.group(1)), float(match.group(2))
            # Basic sanity check for Virginia coordinates
            if 36.0 <= lat <= 40.0 and -84.0 <= lon <= -75.0:
                return lat, lon
        except ValueError:
            pass
    
    logging.warning(f"Could not parse coordinates: '{coord_str}'")
    return None, None


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance (km) between two WGS84 points."""
    if None in (lat1, lon1, lat2, lon2):
        return np.nan
    _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0


def integrate_manual_coordinates() -> pd.DataFrame:
    """Load manual worksheet and integrate coordinates into tier2 dataset."""
    
    if not MANUAL_WORKSHEET.exists():
        raise FileNotFoundError(f"Manual worksheet not found: {MANUAL_WORKSHEET}")
    
    if not TIER2_CSV.exists():
        raise FileNotFoundError(f"Tier2 results not found: {TIER2_CSV}")
    
    # Load datasets
    logging.info("Loading manual worksheet and tier2 results...")
    manual_df = pd.read_csv(MANUAL_WORKSHEET)
    tier2_df = pd.read_csv(TIER2_CSV)
    
    # Parse manual coordinates
    logging.info("Parsing manual coordinates...")
    manual_coords = manual_df['manual_lat_lon'].apply(parse_manual_coordinates)
    manual_df['manual_lat'] = manual_coords.apply(lambda x: x[0])
    manual_df['manual_lon'] = manual_coords.apply(lambda x: x[1])
    
    # Count successful manual entries
    manual_count = manual_df[['manual_lat', 'manual_lon']].notna().all(axis=1).sum()
    logging.info(f"Found {manual_count} valid manual coordinates")
    
    # Merge manual coordinates back into tier2
    manual_coords_only = manual_df[['cp_id', 'manual_lat', 'manual_lon']].copy()
    result_df = tier2_df.merge(manual_coords_only, on='cp_id', how='left')
    
    # Create final coordinates: use manual if available, else auto
    result_df['final_lat'] = result_df['manual_lat'].fillna(result_df['anchor_lat'])
    result_df['final_lon'] = result_df['manual_lon'].fillna(result_df['anchor_lon'])
    result_df['coord_source'] = np.where(
        result_df['manual_lat'].notna(), 'manual', 
        np.where(result_df['anchor_lat'].notna(), 'auto', 'none')
    )
    
    # Recompute distances using final coordinates
    logging.info("Recomputing distances with manual coordinates...")
    mask = (result_df['final_lat'].notna() & result_df['centroid_lat'].notna())
    result_df.loc[mask, 'final_d_km'] = result_df.loc[mask].apply(
        lambda r: great_circle_km(r.final_lat, r.final_lon, r.centroid_lat, r.centroid_lon), 
        axis=1
    )
    result_df['final_within_10km'] = result_df['final_d_km'] <= 10.0
    
    return result_df


def generate_final_report(df: pd.DataFrame) -> None:
    """Generate comprehensive final report on Tier 2 performance."""
    
    print("=" * 60)
    print("TIER 2 FINAL RESULTS (AUTO + MANUAL)")
    print("=" * 60)
    
    # Overall statistics
    total_anchors = len(df)
    auto_resolved = (df['coord_source'] == 'auto').sum()
    manual_resolved = (df['coord_source'] == 'manual').sum()
    total_resolved = auto_resolved + manual_resolved
    
    print(f"\nANCHOR RESOLUTION:")
    print(f"  Total anchor phrases: {total_anchors}")
    print(f"  Auto-resolved: {auto_resolved} ({auto_resolved/total_anchors:.1%})")
    print(f"  Manually located: {manual_resolved} ({manual_resolved/total_anchors:.1%})")
    print(f"  Total resolved: {total_resolved} ({total_resolved/total_anchors:.1%})")
    print(f"  Unresolved: {total_anchors - total_resolved}")
    
    # Distance statistics
    resolved_df = df[df['coord_source'] != 'none']
    if len(resolved_df) > 0:
        distances = resolved_df['final_d_km'].dropna()
        within_10km = (distances <= 10).sum()
        
        print(f"\nDISTANCE STATISTICS ({len(distances)} anchors with distances):")
        print(f"  Median distance: {distances.median():.2f} km")
        print(f"  90th percentile: {distances.quantile(0.9):.2f} km")
        print(f"  Within 10km: {within_10km} ({within_10km/len(distances):.1%})")
        
        # Distance distribution
        print(f"\nDISTANCE DISTRIBUTION:")
        bins = [0, 5, 10, 15, 20, 30, 50, float('inf')]
        labels = ['0-5km', '5-10km', '10-15km', '15-20km', '20-30km', '30-50km', '>50km']
        for low, high, label in zip(bins[:-1], bins[1:], labels):
            count = ((distances >= low) & (distances < high)).sum()
            print(f"  {label}: {count} anchors ({count/len(distances):.1%})")
    
    # Performance by coordinate source
    print(f"\nPERFORMANCE BY SOURCE:")
    for source in ['auto', 'manual']:
        subset = df[df['coord_source'] == source]
        if len(subset) > 0:
            dists = subset['final_d_km'].dropna()
            if len(dists) > 0:
                median_d = dists.median()
                within10 = (dists <= 10).sum()
                print(f"  {source.title()}: {len(dists)} anchors, median {median_d:.1f}km, {within10} within 10km")
    
    # Final viability assessment
    print(f"\nTIER 2 VIABILITY:")
    total_usable = len(distances) if 'distances' in locals() else 0
    median_dist = distances.median() if 'distances' in locals() and len(distances) > 0 else float('inf')
    
    if total_usable >= 100 and median_dist <= 10:
        verdict = "PUBLISH: Tier 2 meets statistical thresholds"
    elif total_usable >= 80 and median_dist <= 15:
        verdict = "MARGINAL: Consider as exploratory analysis"
    else:
        verdict = "INSUFFICIENT: Focus on county-level validation only"
    
    print(f"  {verdict}")
    print("=" * 60)


def main() -> None:
    """Main integration workflow."""
    
    # Integrate manual coordinates
    final_df = integrate_manual_coordinates()
    
    # Save final dataset
    final_df.to_csv(FINAL_CSV, index=False)
    logging.info(f"Final dataset saved: {FINAL_CSV}")
    
    # Generate report
    generate_final_report(final_df)


if __name__ == "__main__":
    main() 