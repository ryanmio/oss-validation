#!/usr/bin/env python3
"""
Stage 1: Auto-harvest candidate anchors (Demo Version)

This script demonstrates the complete Stage 1 process for auto-harvesting
candidate anchors using mock data to show the core functionality.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely.geometry import Point, box

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=""))

# Constants
WORKSPACE_DIR = Path("/workspace")
DATA_DIR = WORKSPACE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# CRS constants
CRS_LATLON = "EPSG:4326"


def create_mock_gnis_data() -> pd.DataFrame:
    """Create mock GNIS data for demonstration."""
    mock_data = [
        # Courthouse entries
        {
            'FEATURE_ID': '1001',
            'FEATURE_NAME': 'Richmond Courthouse',
            'FEATURE_CLASS': 'Locale',
            'STATE_ALPHA': 'VA',
            'COUNTY_NAME': 'Richmond',
            'PRIM_LAT_DEC': 37.5407,
            'PRIM_LONG_DEC': -78.4766
        },
        {
            'FEATURE_ID': '1002',
            'FEATURE_NAME': 'Charlottesville Courthouse',
            'FEATURE_CLASS': 'Locale',
            'STATE_ALPHA': 'VA',
            'COUNTY_NAME': 'Albemarle',
            'PRIM_LAT_DEC': 38.0293,
            'PRIM_LONG_DEC': -78.8503
        },
        {
            'FEATURE_ID': '1003',
            'FEATURE_NAME': 'Henrico Courthouse',
            'FEATURE_CLASS': 'Locale',
            'STATE_ALPHA': 'VA',
            'COUNTY_NAME': 'Henrico',
            'PRIM_LAT_DEC': 37.5407,
            'PRIM_LONG_DEC': -77.4360
        },
        # Stream mouth entries
        {
            'FEATURE_ID': '2001',
            'FEATURE_NAME': 'Rivanna Mouth',
            'FEATURE_CLASS': 'Stream',
            'STATE_ALPHA': 'VA',
            'COUNTY_NAME': 'Albemarle',
            'PRIM_LAT_DEC': 38.0193,
            'PRIM_LONG_DEC': -78.8403
        },
        {
            'FEATURE_ID': '2002',
            'FEATURE_NAME': 'James Mouth',
            'FEATURE_CLASS': 'Stream',
            'STATE_ALPHA': 'VA',
            'COUNTY_NAME': 'Henrico',
            'PRIM_LAT_DEC': 37.5307,
            'PRIM_LONG_DEC': -77.4260
        },
        {
            'FEATURE_ID': '2003',
            'FEATURE_NAME': 'Appomattox Mouth',
            'FEATURE_CLASS': 'Stream',
            'STATE_ALPHA': 'VA',
            'COUNTY_NAME': 'Prince George',
            'PRIM_LAT_DEC': 37.2331,
            'PRIM_LONG_DEC': -79.1322
        },
        {
            'FEATURE_ID': '2004',
            'FEATURE_NAME': 'North Anna Mouth',
            'FEATURE_CLASS': 'Stream',
            'STATE_ALPHA': 'VA',
            'COUNTY_NAME': 'Hanover',
            'PRIM_LAT_DEC': 37.8943,
            'PRIM_LONG_DEC': -78.4697
        },
        {
            'FEATURE_ID': '2005',
            'FEATURE_NAME': 'Buffalo Mouth',
            'FEATURE_CLASS': 'Stream',
            'STATE_ALPHA': 'VA',
            'COUNTY_NAME': 'Amherst',
            'PRIM_LAT_DEC': 37.3318,
            'PRIM_LONG_DEC': -78.6427
        },
        {
            'FEATURE_ID': '2006',
            'FEATURE_NAME': 'Pamunkey Mouth',
            'FEATURE_CLASS': 'Stream',
            'STATE_ALPHA': 'VA',
            'COUNTY_NAME': 'Hanover',
            'PRIM_LAT_DEC': 37.8716,
            'PRIM_LONG_DEC': -78.0465
        }
    ]
    
    return pd.DataFrame(mock_data)


def create_mock_counties_data() -> gpd.GeoDataFrame:
    """Create mock county boundary data."""
    counties = [
        {
            'NAME': 'Richmond',
            'geometry': box(-78.5, 37.5, -78.4, 37.6)
        },
        {
            'NAME': 'Albemarle',
            'geometry': box(-78.9, 37.9, -78.7, 38.1)
        },
        {
            'NAME': 'Henrico',
            'geometry': box(-77.5, 37.4, -77.3, 37.6)
        },
        {
            'NAME': 'Prince George',
            'geometry': box(-79.2, 37.1, -79.0, 37.3)
        },
        {
            'NAME': 'Hanover',
            'geometry': box(-78.5, 37.8, -78.0, 38.0)
        },
        {
            'NAME': 'Amherst',
            'geometry': box(-78.7, 37.2, -78.5, 37.4)
        }
    ]
    
    return gpd.GeoDataFrame(counties, crs=CRS_LATLON)


def get_county_for_point(lat: float, lon: float, counties_gdf: gpd.GeoDataFrame) -> str:
    """Get the modern county name for a given lat/lon point."""
    point = Point(lon, lat)
    
    # Find which county contains this point
    for idx, county in counties_gdf.iterrows():
        if county.geometry.contains(point):
            return county['NAME']
    
    # If no exact match, find closest county
    counties_gdf_copy = counties_gdf.copy()
    counties_gdf_copy['distance'] = counties_gdf_copy.geometry.distance(point)
    closest_county = counties_gdf_copy.loc[counties_gdf_copy['distance'].idxmin()]
    
    return closest_county['NAME']


def find_courthouse_anchors(counties_gdf: gpd.GeoDataFrame, gnis_df: pd.DataFrame, 
                          parcel_counties: List[str]) -> List[Dict]:
    """Find courthouse anchors for each unique county."""
    courthouse_anchors = []
    
    logger.info(f"Finding courthouse anchors for {len(parcel_counties)} unique counties...")
    
    for county_name in parcel_counties:
        # Find GNIS points with courthouse in name
        courthouse_mask = (
            (gnis_df['FEATURE_NAME'].str.contains(' Courthouse', case=False, na=False)) &
            (gnis_df['FEATURE_CLASS'] == 'Locale') &
            (gnis_df['COUNTY_NAME'].str.upper() == county_name.upper())
        )
        
        courthouse_points = gnis_df[courthouse_mask]
        
        if len(courthouse_points) == 0:
            logger.warning(f"No courthouse found for {county_name}")
            continue
        
        if len(courthouse_points) == 1:
            # Single courthouse - use it
            courthouse = courthouse_points.iloc[0]
        else:
            # Multiple courthouses - pick closest to county centroid
            county_row = counties_gdf[counties_gdf['NAME'] == county_name]
            if len(county_row) == 0:
                logger.warning(f"County {county_name} not found in TIGER data")
                continue
            
            county_centroid = county_row.geometry.centroid.iloc[0]
            
            # Calculate distances
            distances = []
            for idx, courthouse in courthouse_points.iterrows():
                courthouse_point = Point(courthouse['PRIM_LONG_DEC'], courthouse['PRIM_LAT_DEC'])
                distance = county_centroid.distance(courthouse_point)
                distances.append((distance, idx))
            
            # Pick closest
            _, closest_idx = min(distances)
            courthouse = courthouse_points.loc[closest_idx]
        
        courthouse_anchors.append({
            'grant_id': f"auto_courthouse_{county_name}",
            'anchor_lat': courthouse['PRIM_LAT_DEC'],
            'anchor_lon': courthouse['PRIM_LONG_DEC'],
            'sigma_m': 300,
            'anchor_quality': 'auto_courthouse'
        })
        
        logger.info(f"Found courthouse for {county_name}: {courthouse['FEATURE_NAME']}")
    
    return courthouse_anchors


def find_mouth_anchors(manual_anchors_df: pd.DataFrame, gnis_df: pd.DataFrame, 
                      counties_gdf: gpd.GeoDataFrame) -> List[Dict]:
    """Find mouth anchors from manual anchor phrases."""
    mouth_anchors = []
    
    if 'anchor_phrase' not in manual_anchors_df.columns:
        logger.warning("No anchor_phrase column found in manual anchors")
        return mouth_anchors
    
    # Regex pattern for mouth references
    mouth_pattern = r'mouth of ([A-Z][A-Za-z]+ [A-Z][A-Za-z]+) (Creek|River|Branch)$'
    
    unique_streams = set()
    
    for idx, row in manual_anchors_df.iterrows():
        if pd.isna(row['anchor_phrase']):
            continue
            
        match = re.search(mouth_pattern, row['anchor_phrase'])
        if not match:
            continue
        
        stream_name = match.group(1)
        stream_type = match.group(2)
        full_stream = f"{stream_name} {stream_type}"
        
        # Get county for this manual anchor
        county_name = get_county_for_point(
            row['anchor_lat'], 
            row['anchor_lon'], 
            counties_gdf
        )
        
        stream_county_key = (full_stream, county_name)
        
        if stream_county_key in unique_streams:
            continue
        
        unique_streams.add(stream_county_key)
        
        # Look for mouth in GNIS
        mouth_name = f"{stream_name} Mouth"
        
        mouth_mask = (
            (gnis_df['FEATURE_NAME'] == mouth_name) &
            (gnis_df['COUNTY_NAME'].str.upper() == county_name.upper())
        )
        
        mouth_points = gnis_df[mouth_mask]
        
        if len(mouth_points) == 1:
            mouth_point = mouth_points.iloc[0]
            
            mouth_anchors.append({
                'grant_id': f"auto_mouth_{stream_name.replace(' ', '_')}_{county_name}",
                'anchor_lat': mouth_point['PRIM_LAT_DEC'],
                'anchor_lon': mouth_point['PRIM_LONG_DEC'],
                'sigma_m': 500,
                'anchor_quality': 'auto_mouth'
            })
            
            logger.info(f"Found mouth for {full_stream} in {county_name}")
        else:
            logger.warning(f"No unique mouth found for {full_stream} in {county_name}")
        
        # Stop at 15 mouth anchors
        if len(mouth_anchors) >= 15:
            break
    
    return mouth_anchors


def main():
    """Main Stage 1 execution."""
    logger.info("=== Stage 1: Auto-harvest candidate anchors (Demo) ===")
    
    # Check for required input files
    centralva_path = DATA_DIR / "centralva.geojson"
    manual_worksheet_path = DATA_DIR / "manual_anchor_worksheet.csv"
    
    if not centralva_path.exists():
        logger.error(f"Required file not found: {centralva_path}")
        logger.info("Please place centralva.geojson in the data/ directory")
        return
    
    if not manual_worksheet_path.exists():
        logger.error(f"Required file not found: {manual_worksheet_path}")
        logger.info("Please place manual_anchor_worksheet.csv in the data/ directory")
        return
    
    # Step 1: Create mock reference data
    logger.info("Step 1: Creating mock reference data...")
    
    gnis_df = create_mock_gnis_data()
    counties_gdf = create_mock_counties_data()
    
    logger.success(f"Created mock GNIS data with {len(gnis_df)} features")
    logger.success(f"Created mock county data with {len(counties_gdf)} counties")
    
    # Step 2: Load input data
    logger.info("Step 2: Loading input data...")
    
    try:
        # Load parcels
        parcels_gdf = gpd.read_file(centralva_path)
        logger.success(f"Loaded {len(parcels_gdf)} parcels from centralva.geojson")
        
        # Load manual anchors
        manual_anchors_df = pd.read_csv(manual_worksheet_path)
        logger.success(f"Loaded {len(manual_anchors_df)} manual anchors")
        
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return
    
    # Step 3: Determine counties for parcels
    logger.info("Step 3: Determining counties for parcels...")
    
    # Get parcel centroids
    parcels_gdf_4326 = parcels_gdf.to_crs(CRS_LATLON)
    centroids = parcels_gdf_4326.geometry.centroid
    
    # Find unique counties
    unique_counties = set()
    for centroid in centroids:
        county = get_county_for_point(centroid.y, centroid.x, counties_gdf)
        unique_counties.add(county)
    
    unique_counties = sorted(unique_counties)
    logger.success(f"Found {len(unique_counties)} unique counties: {', '.join(unique_counties)}")
    
    # Step 4: Find courthouse anchors
    logger.info("Step 4: Finding courthouse anchors...")
    courthouse_anchors = find_courthouse_anchors(counties_gdf, gnis_df, unique_counties)
    
    # Step 5: Find mouth anchors
    logger.info("Step 5: Finding mouth anchors...")
    mouth_anchors = find_mouth_anchors(manual_anchors_df, gnis_df, counties_gdf)
    
    # Step 6: Combine with manual anchors
    logger.info("Step 6: Combining with manual anchors...")
    
    # Convert manual anchors to standard format
    manual_anchors = []
    for idx, row in manual_anchors_df.iterrows():
        manual_anchors.append({
            'grant_id': row['grant_id'],
            'anchor_lat': row['anchor_lat'],
            'anchor_lon': row['anchor_lon'],
            'sigma_m': 100,  # Default for manual anchors
            'anchor_quality': 'manual'
        })
    
    # Combine all anchors
    all_anchors = manual_anchors + courthouse_anchors + mouth_anchors
    
    # Step 7: Write output
    logger.info("Step 7: Writing candidate anchor set...")
    
    output_path = PROCESSED_DIR / "candidate_anchor_set.csv"
    
    # Create DataFrame and save
    anchors_df = pd.DataFrame(all_anchors)
    anchors_df.to_csv(output_path, index=False)
    
    logger.success(f"Wrote {len(all_anchors)} candidate anchors to {output_path}")
    
    # Step 8: Print summary
    logger.info("=== SUMMARY ===")
    print(f"Total manual anchors: {len(manual_anchors)}")
    print(f"Courthouse anchors added: {len(courthouse_anchors)}")
    print(f"Mouth anchors added: {len(mouth_anchors)}")
    print(f"Total candidate anchors: {len(all_anchors)}")
    
    logger.success("Stage 1 completed successfully!")


if __name__ == "__main__":
    main()