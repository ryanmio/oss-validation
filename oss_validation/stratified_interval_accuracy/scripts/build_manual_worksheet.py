#!/usr/bin/env python3
"""build_manual_worksheet.py
============================
Create consolidated manual-audit worksheet by:
1. Loading stratified_100_anchor_worksheet.csv (100 grants, seed=42)
2. Merging any manual coordinates from '100 anchor worksheet - Sheet1/2.csv'
3. Output: stratified100_manual_worksheet.csv (ready for manual geolocation)

Run:
    cd /path/to/repo
    python stratified_interval_accuracy/scripts/build_manual_worksheet.py
"""

import pandas as pd
import re
from pathlib import Path

def parse_manual_coords(coord_str):
    """Parse 'lat, lon' string into (lat, lon) tuple or (None, None)."""
    if pd.isna(coord_str) or not str(coord_str).strip():
        return None, None
    
    # Try to match "lat, lon" pattern
    match = re.match(r'^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)', str(coord_str).strip())
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def load_existing_manual(sheet_path):
    """Load existing manual worksheet and extract manual coords."""
    if not Path(sheet_path).exists():
        return pd.DataFrame(columns=['cp_id', 'manual_lat', 'manual_lon', 'notes'])
    
    df = pd.read_csv(sheet_path)
    
    # Standardize column names
    coord_col = None
    for col in ['Manual Latlon', 'Manual LatLon', 'manual_lat_lon']:
        if col in df.columns:
            coord_col = col
            break
    
    notes_col = 'Notes' if 'Notes' in df.columns else 'notes'
    
    if coord_col is None:
        return pd.DataFrame(columns=['cp_id', 'manual_lat', 'manual_lon', 'notes'])
    
    # Parse coordinates
    manual_data = []
    for _, row in df.iterrows():
        cp_id = row['cp_id']
        coord_str = row.get(coord_col, '')
        notes = row.get(notes_col, '')
        
        # Skip if marked as bad
        if 'bad match' in str(notes).lower() or 'discard' in str(notes).lower():
            continue
        if 'bad match' in str(coord_str).lower() or 'discard' in str(coord_str).lower():
            continue
            
        lat, lon = parse_manual_coords(coord_str)
        if lat is not None and lon is not None:
            manual_data.append({
                'cp_id': cp_id,
                'manual_lat': lat,
                'manual_lon': lon,
                'notes': str(notes) if pd.notna(notes) else ''
            })
    
    return pd.DataFrame(manual_data)

def main():
    print("Building consolidated manual worksheet...")
    
    # Paths
    base_path = Path('.')
    stratified_path = base_path / 'stratified_interval_accuracy/data/stratified100_sample_master.csv'
    sheet1_path = base_path / 'data/processed/100 anchor worksheet - Sheet1.csv'
    sheet2_path = base_path / 'data/processed/100 anchor worksheet - Sheet2.csv'
    output_path = base_path / 'stratified_interval_accuracy/data/stratified100_manual_geocoding.csv'
    
    # Load stratified sample
    print(f"Loading stratified sample: {stratified_path}")
    stratified = pd.read_csv(stratified_path)
    print(f"  → {len(stratified)} grants")
    
    # Load existing manual coordinates
    print(f"Loading existing manual coords from Sheet1/2...")
    manual1 = load_existing_manual(sheet1_path)
    manual2 = load_existing_manual(sheet2_path)
    manual_combined = pd.concat([manual1, manual2], ignore_index=True)
    
    # Remove duplicates (keep first)
    manual_combined = manual_combined.drop_duplicates(subset=['cp_id'], keep='first')
    print(f"  → {len(manual_combined)} manual coordinates found")
    
    # Merge with stratified sample
    result = stratified.merge(manual_combined, on='cp_id', how='left')
    
    # Clean up columns for manual audit (handle missing 'notes' column)
    base_cols = ['cp_id', 'cp_county', 'decade', 'cp_raw', 'suggested_anchor_phrase', 'manual_lat', 'manual_lon']
    optional_cols = ['notes', 'initial_20', 'anchor_score']  # anchor_score now optional / legacy
    
    final_cols = base_cols.copy()
    for col in optional_cols:
        if col in result.columns:
            final_cols.append(col)
        else:
            result[col] = '' if col == 'notes' else 0
            final_cols.append(col)
    
    result = result[final_cols]
    
    # Fill NaN values
    result['manual_lat'] = result['manual_lat'].fillna('')
    result['manual_lon'] = result['manual_lon'].fillna('')
    result['notes'] = result['notes'].fillna('')
    
    # Add manual_lat_lon column for easier editing
    def combine_coords(row):
        if row['manual_lat'] and row['manual_lon']:
            return f"{row['manual_lat']}, {row['manual_lon']}"
        return ''
    
    result['manual_lat_lon'] = result.apply(combine_coords, axis=1)
    
    # Reorder columns
    result = result[['cp_id', 'cp_county', 'decade', 'cp_raw', 'suggested_anchor_phrase',
                    'manual_lat_lon', 'notes', 'initial_20', 'anchor_score']]
    
    # Save
    print(f"Saving consolidated worksheet: {output_path}")
    result.to_csv(output_path, index=False)
    
    # Summary
    has_coords = (result['manual_lat_lon'] != '').sum()
    print(f"\nSummary:")
    print(f"  Total grants: {len(result)}")
    print(f"  Pre-filled coordinates: {has_coords}")
    print(f"  Remaining to geocode: {len(result) - has_coords}")
    
    if has_coords > 0:
        print(f"\nPre-filled grants:")
        filled = result[result['manual_lat_lon'] != '']
        for _, row in filled.iterrows():
            print(f"  {row['cp_id']}: {row['manual_lat_lon']} ({row['notes'][:50]}...)")

if __name__ == '__main__':
    main() 