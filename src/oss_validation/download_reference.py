"""Download and prepare all reference GIS layers required by the OSS validation pipeline.

Run with:
    python -m oss_validation.download_reference

Outputs are written under data/reference/ as defined in `oss_validation.config`.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import osmnx as ox
import requests
from loguru import logger

from . import config

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _download_file(url: str, out_path: Path, chunk: int = 1 << 20, *, verify_ssl: bool = True) -> None:
    """Download *url* to *out_path* with streaming.

    On some macOS Python installs (particularly Homebrew Python 3.13), the
    default certificate store may be missing.  If SSL verification fails,
    re-run with ``verify_ssl=False`` to skip cert checks **temporarily**.
    """
    logger.info(f"Downloading {url} → {out_path} (verify_ssl={verify_ssl}) …")
    headers = {}
    # Support simple resume – if a partial file exists, request the remaining bytes
    if out_path.exists():
        headers["Range"] = f"bytes={out_path.stat().st_size}-"

    # Separate connect/read timeouts: 30 s connect, unlimited read
    with requests.get(
        url,
        stream=True,
        timeout=(30, None),
        verify=verify_ssl,
        headers=headers,
    ) as r:
        r.raise_for_status()
        with out_path.open("ab" if headers else "wb") as f:
            for part in r.iter_content(chunk):
                f.write(part)
    size_mb = out_path.stat().st_size / 1_048_576
    try:
        rel = out_path.relative_to(config.ROOT_DIR)
    except ValueError:
        rel = out_path
    logger.success(f"Saved {size_mb:.1f} MB → {rel}")


def _extract_zip(zip_path: Path, to_dir: Path) -> None:
    logger.info(f"Extracting {zip_path.name} …")
    with ZipFile(zip_path) as z:
        z.extractall(to_dir)


# ---------------------------------------------------------------------------
# Dataset routines
# ---------------------------------------------------------------------------

def download_tiger_counties() -> Path:
    """Download TIGER/Line Virginia counties.

    Iterates from 2023 → 2020 until a valid ZIP is found (the Census often
    reshuffles the hosting structure annually).  Logs the chosen year and
    writes `tiger_va_counties_<year>.gpkg`.
    """
    for year in range(2023, 2009, -1):
        url = (
            f"https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/"
            f"tl_{year}_51_county.zip"
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmpdir = Path(tmp)
                zfile = tmpdir / "va_counties.zip"
                _download_file(url, zfile, verify_ssl=False)
                _extract_zip(zfile, tmpdir)
                shp = next(tmpdir.glob(f"tl_{year}_51_county.shp"))
                gdf = gpd.read_file(shp)
                out = config.REFERENCE_DIR / f"tiger_va_counties_{year}.gpkg"
                gdf.to_file(out, driver="GPKG")
                logger.success(f"TIGER counties {year} downloaded → {out.relative_to(config.ROOT_DIR)}")
                return out
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"TIGER counties {year} failed: {exc}")
            continue

    # ------------------------------------------------------------------
    # Fallback: Cartographic Boundary (generalized) counties, 500k scale
    # ------------------------------------------------------------------
    logger.warning("Falling back to Cartographic Boundary (cb) county files …")
    for year in range(2023, 2009, -1):
        for sub in ("shp", "shape"):
            url = (
                f"https://www2.census.gov/geo/tiger/GENZ{year}/{sub}/"
                f"cb_{year}_51_county_500k.zip"
            )
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    tmpdir = Path(tmp)
                    zfile = tmpdir / "cb_counties.zip"
                    _download_file(url, zfile, verify_ssl=False)
                    _extract_zip(zfile, tmpdir)
                    shp = next(tmpdir.glob("*.shp"))
                    gdf = gpd.read_file(shp)
                    out = config.REFERENCE_DIR / f"cb_va_counties_{year}.gpkg"
                    gdf.to_file(out, driver="GPKG")
                    logger.success(
                        f"Cartographic counties {year} downloaded → {out.relative_to(config.ROOT_DIR)}"
                    )
                    return out
            except Exception as exc:
                logger.warning(f"CB counties {year} ({sub}) failed: {exc}")
                continue

    # ------------------------------------------------------------------
    # Final fallback: national county shapefile, filter to VA (state FIPS 51)
    # ------------------------------------------------------------------
    logger.warning("Attempting national county file fallback …")
    for year in range(2023, 2009, -1):
        url = (
            f"https://www2.census.gov/geo/tiger/TIGER{year}/COUNTY/"
            f"tl_{year}_us_county.zip"
        )
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmpdir = Path(tmp)
                zfile = tmpdir / "us_counties.zip"
                _download_file(url, zfile, verify_ssl=False)
                _extract_zip(zfile, tmpdir)
                shp = next(tmpdir.glob("*.shp"))
                gdf = gpd.read_file(shp)
                gdf = gdf[gdf["STATEFP"] == "51"].copy()
                out = config.REFERENCE_DIR / f"tiger_va_counties_{year}_from_us.gpkg"
                gdf.to_file(out, driver="GPKG")
                logger.success(
                    f"Filtered VA counties ({len(gdf)}) from national TIGER {year} → {out.relative_to(config.ROOT_DIR)}"
                )
                return out
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"National counties {year} failed: {exc}")
            continue

    raise RuntimeError("All county download strategies failed (TIGER state, CB state, TIGER national).")


def download_nhd_flowlines(huc: str) -> Path:
    url = (
        "https://prd-tnm.s3.amazonaws.com/NHD/HU4/HighResolution/GDB/"
        f"NHD_H_{huc}_HU4_GDB.zip"
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            zfile = tmpdir / f"nhd_{huc}.zip"
            _download_file(url, zfile, verify_ssl=False)
            _extract_zip(zfile, tmpdir)
            gdb_path = next(tmpdir.rglob("*.gdb"))
            gdf = gpd.read_file(gdb_path, layer="NHDFlowline")
            out = config.REFERENCE_DIR / f"nhd_flowlines_hu4_{huc}.gpkg"
            gdf.to_file(out, driver="GPKG")
            return out
    except Exception as exc:
        logger.warning(f"HU4 download for {huc} failed: {exc}; falling back to state-level NHD")

    # Next fallback: state-level HighResolution GDB
    state_gdb_url = (
        "https://prd-tnm.s3.amazonaws.com/NHD/State/HighResolution/GDB/"
        "NHD_H_Virginia_State_GDB.zip"
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            zfile = tmpdir / "nhd_va_gdb.zip"
            _download_file(state_gdb_url, zfile, verify_ssl=False)
            _extract_zip(zfile, tmpdir)
            gdb_path = next(tmpdir.rglob("*.gdb"))
            gdf = gpd.read_file(gdb_path, layer="NHDFlowline")
            out = config.REFERENCE_DIR / "nhd_flowlines_va.gpkg"
            gdf.to_file(out, driver="GPKG")
            logger.success(
                f"State-level NHD flowlines (GDB) saved → {out.relative_to(config.ROOT_DIR)}"
            )
            return out
    except Exception as exc:
        logger.warning(f"State-level NHD GDB failed: {exc}; trying SHAPE archive …")

    # Final fallback: state-level HighResolution SHAPE
    state_shape_url = (
        "https://prd-tnm.s3.amazonaws.com/NHD/State/HighResolution/Shape/"
        "NHD_H_Virginia_State_Shape.zip"
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        zfile = tmpdir / "nhd_va_shp.zip"
        _download_file(state_shape_url, zfile, verify_ssl=False)
        _extract_zip(zfile, tmpdir)
        shp_path = next(tmpdir.rglob("*NHDFlowline.shp"))
        gdf = gpd.read_file(shp_path)
        out = config.REFERENCE_DIR / "nhd_flowlines_va.gpkg"
        gdf.to_file(out, driver="GPKG")
        logger.success(
            f"State-level NHD flowlines (SHAPE) saved → {out.relative_to(config.ROOT_DIR)}"
        )
        return out


def download_gnis_features() -> Path:
    out = config.REFERENCE_DIR / "gnis_features_va.gpkg"
    if out.exists():
        logger.info("GNIS features already present – skipping download.")
        return out

    # First try legacy State_Shape ZIP
    legacy_url = "https://geonames.usgs.gov/docs/stategaz/VA_State_Shape.zip"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            zfile = tmpdir / "gnis_va.zip"
            _download_file(legacy_url, zfile, verify_ssl=False)
            _extract_zip(zfile, tmpdir)
            shp = next(tmpdir.rglob("*.shp"))
            gdf = gpd.read_file(shp)
            gdf.to_file(out, driver="GPKG")
            logger.success(f"GNIS legacy shape saved → {out.relative_to(config.ROOT_DIR)}")
            return out
    except Exception as exc:
        logger.warning(f"Legacy GNIS URL failed: {exc}; trying FullModel GPKG …")

    # Fallback to new FullModel GeoPackage
    full_url = (
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/GeographicNames/FullModel/"
        "GPKG/Virginia.gpkg.zip"
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        zfile = tmpdir / "gnis_full.zip"
        _download_file(full_url, zfile, verify_ssl=False)
        _extract_zip(zfile, tmpdir)
        gpkg = next(tmpdir.rglob("*.gpkg"))
        gdf = gpd.read_file(gpkg, layer="Feature")
        gdf.to_file(out, driver="GPKG")
        logger.success(f"GNIS FullModel saved → {out.relative_to(config.ROOT_DIR)}")
        return out


def download_osm_roads() -> Path:
    logger.info("Requesting Virginia road network from OpenStreetMap (this may take several minutes)…")
    graph = ox.graph_from_place("Virginia, USA", network_type="drive")
    gdf, _ = ox.graph_to_gdfs(graph)
    out = config.REFERENCE_DIR / "osm_roads_va.gpkg"
    gdf.to_file(out, driver="GPKG")
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Download reference layers, but skip those that already exist."""
    logger.add(lambda m: print(m, end=""))
    config.REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Counties – skip if any existing county GeoPackage is present.
    # ------------------------------------------------------------------
    county_files = list(config.REFERENCE_DIR.glob("*counties*.gpkg"))
    if county_files:
        logger.info("County layer already present – skipping download.")
    else:
        download_tiger_counties()

    # ------------------------------------------------------------------
    # NHD flowlines – skip if the merged Virginia file already exists.
    # ------------------------------------------------------------------
    nhd_state_path = config.REFERENCE_DIR / "nhd_flowlines_va.gpkg"
    if nhd_state_path.exists():
        logger.info("NHD flowlines already present – skipping download.")
    else:
        for huc in ("0208", "0301"):
            download_nhd_flowlines(huc)

    # ------------------------------------------------------------------
    # GNIS features – function already handles its own skip logic.
    # ------------------------------------------------------------------
    download_gnis_features()

    # ------------------------------------------------------------------
    # OSM roads – skip if previously cached.
    # ------------------------------------------------------------------
    roads_path = config.REFERENCE_DIR / "osm_roads_va.gpkg"
    if roads_path.exists():
        logger.info("OSM roads already present – skipping download.")
    else:
        download_osm_roads()

    logger.success("All reference layers available ✓")


if __name__ == "__main__":
    main() 