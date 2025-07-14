#!/usr/bin/env python3
"""compute_distances_manual.py
==============================
Compute great-circle distances between manually geocoded anchor points 
and OSS polygon centroids for the stratified 100-grant sample.

Inputs:
  - tier2_positional_accuracy/data/stratified100_manual_worksheet.csv
  - data/processed/spatial_validation.csv (for OSS centroids)

Outputs:
  - tier2_positional_accuracy/results/stratified100_final.csv

Run:
    cd /path/to/repo
    python tier2_positional_accuracy/scripts/compute_distances_manual.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pyproj import Geod
import re

def parse_manual_coords(coord_str):
    """Parse 'lat, lon' string into (lat, lon) tuple or (None, None)."""
    if pd.isna(coord_str) or not str(coord_str).strip():
        return None, None
    
    # Try to match "lat, lon" pattern
    match = re.match(r'^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)', str(coord_str).strip())
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def is_bad_entry(notes_str, coords_str):
    """Check if entry should be excluded (marked as bad/discard)."""
    for text in [str(notes_str), str(coords_str)]:
        if pd.notna(text) and any(flag in text.lower() for flag in ['bad match', 'discard', 'bad']):
            return True
    return False

def main():
    print("Computing distances for manual-only Tier-2 analysis...")
    
    # Paths
    base_path = Path('.')
    manual_path = base_path / 'tier2_positional_accuracy/data/stratified100_manual_geocoding.csv'
    spatial_path = base_path / 'data/processed/spatial_validation.csv'
    output_path = base_path / 'tier2_positional_accuracy/results/stratified100_final.csv'
    
    # Create results directory
    output_path.parent.mkdir(exist_ok=True)
    
    # Load manual worksheet
    print(f"Loading manual worksheet: {manual_path}")
    manual_df = pd.read_csv(manual_path)
    print(f"  → {len(manual_df)} total grants")
    
    # Load OSS centroids from the original worksheet files (since spatial_validation.csv doesn't have centroids)
    print(f"Loading OSS centroids from worksheet files...")
    
    # Try to get centroids from the 100 anchor worksheets
    sheet1_path = base_path / 'data/processed/100 anchor worksheet - Sheet1.csv'
    sheet2_path = base_path / 'data/processed/100 anchor worksheet - Sheet2.csv'
    
    centroids_list = []
    for sheet_path in [sheet1_path, sheet2_path]:
        if sheet_path.exists():
            sheet_df = pd.read_csv(sheet_path)
            if all(col in sheet_df.columns for col in ['cp_id', 'oss_id', 'centroid_lat', 'centroid_lon']):
                centroids_list.append(sheet_df[['cp_id', 'oss_id', 'centroid_lat', 'centroid_lon']])
    
    if centroids_list:
        centroids_df = pd.concat(centroids_list, ignore_index=True).drop_duplicates(subset=['cp_id'])
    else:
        # Fallback: try to get from spatial_validation.csv with different column names
        spatial_df = pd.read_csv(spatial_path)
        centroids_df = spatial_df[['cp_id', 'oss_id']].copy() if 'oss_id' in spatial_df.columns else spatial_df[['cp_id']].copy()
        centroids_df['centroid_lat'] = np.nan
        centroids_df['centroid_lon'] = np.nan
    
    # Merge to get OSS centroids
    merged = manual_df.merge(centroids_df, on='cp_id', how='left')
    
    print(f"  → {merged['oss_id'].notna().sum()} grants have OSS matches")
    
    # Parse manual coordinates
    manual_coords = []
    for _, row in merged.iterrows():
        lat, lon = parse_manual_coords(row.get('manual_lat_lon', ''))
        manual_coords.append({'manual_lat': lat, 'manual_lon': lon})
    
    manual_coords_df = pd.DataFrame(manual_coords)
    result = pd.concat([merged, manual_coords_df], axis=1)
    
    # Filter for valid entries
    valid_mask = (
        result['manual_lat'].notna() &
        result['manual_lon'].notna() &
        result['centroid_lat'].notna() &
        result['centroid_lon'].notna() &
        ~result.apply(lambda row: is_bad_entry(row.get('notes', ''), row.get('manual_lat_lon', '')), axis=1)
    )
    
    valid_df = result[valid_mask].copy()
    print(f"  → {len(valid_df)} grants have valid manual coordinates")
    
    if len(valid_df) == 0:
        print("No valid manual coordinates found. Creating empty results file.")
        result['final_d_km'] = np.nan
        result['coord_source'] = 'none'
        result.to_csv(output_path, index=False)
        return
    
    # Compute great-circle distances
    print("Computing great-circle distances...")
    geod = Geod(ellps='WGS84')
    
    distances = []
    for _, row in valid_df.iterrows():
        _, _, dist_m = geod.inv(
            row['manual_lon'], row['manual_lat'],
            row['centroid_lon'], row['centroid_lat']
        )
        distances.append(dist_m / 1000.0)  # Convert to km
    
    valid_df['final_d_km'] = distances
    valid_df['coord_source'] = 'manual'
    
    # Merge back to full dataset
    result = result.merge(
        valid_df[['cp_id', 'final_d_km', 'coord_source']], 
        on='cp_id', 
        how='left'
    )
    
    # Fill missing values
    result['final_d_km'] = result['final_d_km'].fillna(np.nan)
    result['coord_source'] = result['coord_source'].fillna('none')
    
    # Add summary columns
    result['has_manual_coords'] = result['manual_lat'].notna() & result['manual_lon'].notna()
    result['has_oss_match'] = result['oss_id'].notna()
    result['is_valid'] = valid_mask
    
    # Reorder columns
    final_cols = [
        'cp_id', 'cp_county', 'decade', 'oss_id',
        'manual_lat_lon', 'manual_lat', 'manual_lon', 
        'centroid_lat', 'centroid_lon',
        'final_d_km', 'coord_source',
        'has_manual_coords', 'has_oss_match', 'is_valid',
        'notes', 'cp_raw', 'suggested_anchor_phrase'
    ]
    
    # Only include columns that exist
    available_cols = [col for col in final_cols if col in result.columns]
    result = result[available_cols]
    
    # Save results
    print(f"Saving results: {output_path}")
    result.to_csv(output_path, index=False)
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"  Total grants: {len(result)}")
    print(f"  Have manual coordinates: {result['has_manual_coords'].sum()}")
    print(f"  Have OSS matches: {result['has_oss_match'].sum()}")
    print(f"  Valid for distance calc: {result['is_valid'].sum()}")
    
    if result['is_valid'].sum() > 0:
        valid_distances = result[result['is_valid']]['final_d_km']
        print(f"  Distance stats (km):")
        print(f"    Min:    {valid_distances.min():.2f}")
        print(f"    Median: {valid_distances.median():.2f}")
        print(f"    Max:    {valid_distances.max():.2f}")
        print(f"    P90:    {valid_distances.quantile(0.9):.2f}")
        print(f"    Mean:   {valid_distances.mean():.2f}")

if __name__ == '__main__':
    main() 