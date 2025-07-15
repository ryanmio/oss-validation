#!/usr/bin/env python3
"""compute_distances_with_fallback.py
=====================================
Calculate great-circle distances between OSS centroid and either:
1. Manually geocoded anchor (`manual_lat_lon`)
2. Fallback = ½ of the maximum centroid-to-centroid distance **within the same county**

Outputs updated `results/stratified100_final.csv`.
Prints summary counts for located vs fallback rows.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from pyproj import Geod
import re
from itertools import combinations

BASE = Path('.')
DATA_DIR = BASE / 'stratified_interval_accuracy' / 'data'
RESULT_DIR = BASE / 'stratified_interval_accuracy' / 'results'
RESULT_DIR.mkdir(parents=True, exist_ok=True)

WORKING_CSV = DATA_DIR / 'stratified100_manual_geocoding.csv'
CENTROID_SOURCE = BASE / 'data' / 'raw' / 'CentralVAPatents_PLY-shp' / 'centralva.geojson'
OUTPUT_CSV = RESULT_DIR / 'stratified100_final.csv'

geod = Geod(ellps='WGS84')

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def parse_coords(coord_str):
    if pd.isna(coord_str):
        return None, None
    coord_str = str(coord_str).strip()
    if coord_str == '-' or coord_str == '':
        return None, None
    m = re.match(r'^\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)', coord_str)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def gc_distance(lat1, lon1, lat2, lon2):
    _, _, dist_m = geod.inv(lon1, lat1, lon2, lat2)
    return dist_m / 1000.0  # km

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
print(f"Loading working file: {WORKING_CSV}")
# Load working and master to attach oss_id
work_df = pd.read_csv(WORKING_CSV)
print(f"  → {len(work_df)} rows")

master_path = DATA_DIR / 'stratified100_sample_master.csv'
master_df = pd.read_csv(master_path)[['cp_id','oss_id']]
work_df = work_df.merge(master_df, on='cp_id', how='left')

# ------------------------------------------------------------------
# Load centroids from GeoJSON (OBJECTID == oss_id)
# ------------------------------------------------------------------
print(f"Loading centroid source (geojson): {CENTROID_SOURCE}")
import geopandas as gpd
gdf = gpd.read_file(CENTROID_SOURCE)[['OBJECTID', 'geometry']]
# Compute centroid coordinates in WGS84
gdf = gdf.to_crs(4326)
gdf['centroid_lon'] = gdf.geometry.centroid.x
gdf['centroid_lat'] = gdf.geometry.centroid.y
centroids_df = gdf[['OBJECTID', 'centroid_lat', 'centroid_lon']].rename(columns={'OBJECTID':'oss_id'})

# Merge centroids
merged = work_df.merge(centroids_df, on='oss_id', how='left')

missing_centroids = merged['centroid_lat'].isna().sum()
if missing_centroids:
    print(f"⚠️  {missing_centroids} rows missing centroid coords – they will be skipped.")

# ---------------------------------------------------------------------
# Compute county widths
# ---------------------------------------------------------------------
county_width_km = {}
for county, grp in merged.groupby('cp_county'):
    coords = grp[['centroid_lat', 'centroid_lon']].dropna().values
    max_dist = 0.0
    if len(coords) > 1:
        for (lat1, lon1), (lat2, lon2) in combinations(coords, 2):
            d = gc_distance(lat1, lon1, lat2, lon2)
            if d > max_dist:
                max_dist = d
    county_width_km[county] = max_dist / 2.0  # fallback is half the width

# If width is zero (single sample), use overall median width/2 as fallback
non_zero = [v for v in county_width_km.values() if v > 0]
overall_fallback = np.median(non_zero) if non_zero else 5.0  # km
for county, val in county_width_km.items():
    if val == 0:
        county_width_km[county] = overall_fallback

# ---------------------------------------------------------------------
# Compute distances / fallbacks
# ---------------------------------------------------------------------
manual_rows = 0
fallback_rows = 0
final_distances = []
coord_source = []

for _, row in merged.iterrows():
    lat_c, lon_c = row['centroid_lat'], row['centroid_lon']
    if pd.isna(lat_c) or pd.isna(lon_c):
        final_distances.append(np.nan)
        coord_source.append('no_centroid')
        continue
    lat_m, lon_m = parse_coords(row.get('manual_lat_lon', ''))
    if lat_m is not None:
        dkm = gc_distance(lat_m, lon_m, lat_c, lon_c)
        manual_rows += 1
        final_distances.append(dkm)
        coord_source.append('manual')
    else:
        fb = county_width_km.get(row['cp_county'], overall_fallback)
        fallback_rows += 1
        final_distances.append(fb)
        coord_source.append('fallback')

merged['final_d_km'] = final_distances
merged['coord_source'] = coord_source
merged['is_valid'] = merged['coord_source'] != 'no_centroid'

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
merged.to_csv(OUTPUT_CSV, index=False)
print(f"Saved results → {OUTPUT_CSV}")

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
print("\nSummary:")
print(f"  Manual anchors:   {manual_rows}")
print(f"  Fallback anchors: {fallback_rows}")
print(f"  Missing centroid: {(merged['coord_source']=='no_centroid').sum()}")
print(f"  Total:            {len(merged)}") 