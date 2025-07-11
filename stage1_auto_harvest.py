#!/usr/bin/env python3
"""
Stage 1: Auto-harvest candidate anchors (no solver)

This script implements the complete Stage 1 process for auto-harvesting
candidate anchors from GNIS data and manual anchor worksheets.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from zipfile import ZipFile

import geopandas as gpd
import pandas as pd
import requests
from loguru import logger
from shapely.geometry import Point
from shapely.ops import unary_union

# Configure logging
logger.remove()
logger.add(lambda msg: print(msg, end=""))

# Constants
WORKSPACE_DIR = Path("/workspace")
DATA_DIR = WORKSPACE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
for dir_path in [RAW_DIR, REFERENCE_DIR, PROCESSED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# CRS constants
CRS_LATLON = "EPSG:4326"
CRS_DISTANCE = "EPSG:3857"


def download_file(url: str, out_path: Path, chunk_size: int = 8192) -> None:
    """Download a file from URL to output path."""
    logger.info(f"Downloading {url} → {out_path}")
    
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    
    size_mb = out_path.stat().st_size / (1024 * 1024)
    logger.success(f"Downloaded {size_mb:.1f} MB → {out_path}")


def download_gnis_national_file() -> Path:
    """Download USGS GNIS NationalFile.txt if not present."""
    gnis_path = RAW_DIR / "NationalFile.txt"
    
    if gnis_path.exists():
        logger.info("GNIS NationalFile.txt already exists - skipping download")
        return gnis_path
    
    # Try the direct GNIS file first
    gnis_url = "https://geonames.usgs.gov/docs/NationalFile.txt"
    
    try:
        download_file(gnis_url, gnis_path)
        return gnis_path
    except requests.exceptions.RequestException as e:
        logger.warning(f"Direct GNIS download failed: {e}")
        
        # Fallback to ZIP download
        zip_url = "https://geonames.usgs.gov/docs/federalcodes/NationalFile.zip"
        zip_path = RAW_DIR / "NationalFile.zip"
        
        try:
            download_file(zip_url, zip_path)
            
            # Extract the ZIP file
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(RAW_DIR)
            
            # Find the extracted file
            for candidate in RAW_DIR.glob("*National*"):
                if candidate.is_file() and candidate.suffix in ['.txt', '.csv']:
                    candidate.rename(gnis_path)
                    break
            
            zip_path.unlink()  # Remove ZIP file
            return gnis_path
            
        except Exception as e:
            logger.error(f"GNIS download failed: {e}")
            raise


def download_tiger_counties() -> Path:
    """Download TIGER county boundaries for Virginia."""
    counties_path = REFERENCE_DIR / "tiger_va_counties.gpkg"
    
    if counties_path.exists():
        logger.info("TIGER VA counties already exist - skipping download")
        return counties_path
    
    # Try recent years
    for year in range(2023, 2019, -1):
        url = f"https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/tl_{year}_51_county.zip"
        
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                zip_path = tmp_path / f"counties_{year}.zip"
                
                download_file(url, zip_path)
                
                # Extract ZIP
                with ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(tmp_path)
                
                # Find shapefile
                shp_file = next(tmp_path.glob("*.shp"))
                
                # Read and save as GeoPackage
                gdf = gpd.read_file(shp_file)
                gdf = gdf.to_crs(CRS_LATLON)
                gdf.to_file(counties_path, driver="GPKG")
                
                logger.success(f"TIGER counties {year} saved → {counties_path}")
                return counties_path
                
        except Exception as e:
            logger.warning(f"TIGER counties {year} failed: {e}")
            continue
    
    raise RuntimeError("All TIGER county download attempts failed")


def load_gnis_data() -> pd.DataFrame:
    """Load GNIS data into a pandas DataFrame."""
    gnis_path = RAW_DIR / "NationalFile.txt"
    
    if not gnis_path.exists():
        raise FileNotFoundError(f"GNIS file not found at {gnis_path}")
    
    # GNIS column names (based on USGS documentation)
    gnis_columns = [
        'FEATURE_ID', 'FEATURE_NAME', 'FEATURE_CLASS', 'STATE_ALPHA', 
        'STATE_NUMERIC', 'COUNTY_NAME', 'COUNTY_NUMERIC', 'PRIMARY_LAT_DMS',
        'PRIM_LONG_DMS', 'PRIM_LAT_DEC', 'PRIM_LONG_DEC', 'SOURCE_LAT_DMS',
        'SOURCE_LONG_DMS', 'SOURCE_LAT_DEC', 'SOURCE_LONG_DEC', 'ELEV_IN_M',
        'ELEV_IN_FT', 'MAP_NAME', 'DATE_CREATED', 'DATE_EDITED'
    ]
    
    logger.info("Loading GNIS NationalFile.txt...")
    
    # Read the file with proper parsing
    df = pd.read_csv(gnis_path, sep='\t', names=gnis_columns, dtype=str, low_memory=False)
    
    # Filter to Virginia only
    df = df[df['STATE_ALPHA'] == 'VA'].copy()
    
    # Convert coordinates to numeric
    df['PRIM_LAT_DEC'] = pd.to_numeric(df['PRIM_LAT_DEC'], errors='coerce')
    df['PRIM_LONG_DEC'] = pd.to_numeric(df['PRIM_LONG_DEC'], errors='coerce')
    
    # Remove rows with missing coordinates
    df = df.dropna(subset=['PRIM_LAT_DEC', 'PRIM_LONG_DEC'])
    
    logger.success(f"Loaded {len(df)} Virginia GNIS features")
    return df


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
    logger.info("=== Stage 1: Auto-harvest candidate anchors ===")
    
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
    
    # Step 1: Download reference data
    logger.info("Step 1: Downloading reference data...")
    
    try:
        gnis_path = download_gnis_national_file()
        counties_path = download_tiger_counties()
    except Exception as e:
        logger.error(f"Failed to download reference data: {e}")
        return
    
    # Step 2: Load data
    logger.info("Step 2: Loading data...")
    
    try:
        # Load GNIS data
        gnis_df = load_gnis_data()
        
        # Load county boundaries
        counties_gdf = gpd.read_file(counties_path)
        
        # Load parcels
        parcels_gdf = gpd.read_file(centralva_path)
        logger.success(f"Loaded {len(parcels_gdf)} parcels from centralva.geojson")
        
        # Load manual anchors
        manual_anchors_df = pd.read_csv(manual_worksheet_path)
        logger.success(f"Loaded {len(manual_anchors_df)} manual anchors")
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
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