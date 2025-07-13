# -*- coding: utf-8 -*-
"""Generate and cache historical→modern county overlap mapping for Virginia.

This small utility intersects AHCB historical county polygons with modern
2023 TIGER county polygons (both already in the repo) and records, for each
historical county (cleaned name), the set of modern county names whose
geometries overlap at all with it.

The result is cached in JSON under data/processed/hist_to_modern_overlap.json
so it only needs to be computed once per workspace.
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List

import geopandas as gpd

from . import config
from . import spatial_validation as sv

_CACHE_JSON = config.PROCESSED_DIR / "hist_to_modern_overlap.json"


def _build_mapping() -> Dict[str, List[str]]:
    """Compute overlap mapping (slow – runs once)."""
    ahcb = sv._load_ahcb()
    modern = sv.load_tiger_counties()[["NAME", "geometry"]].copy()

    # Ensure both in same CRS (EPSG:4326 expected already)
    if ahcb.crs is None:
        raise ValueError("AHCB CRS undefined")
    if modern.crs != ahcb.crs:
        modern = modern.to_crs(ahcb.crs)

    mapping: Dict[str, List[str]] = {}

    for hist_idx, hrow in ahcb.iterrows():
        hname = hrow["hist_county_clean"]
        geom = hrow.geometry
        overlaps = modern[modern.intersects(geom)]["NAME"].tolist()
        overlaps_clean = [n.strip() for n in overlaps]
        mapping.setdefault(hname, [])
        mapping[hname] = sorted(set(mapping[hname] + overlaps_clean))

    return mapping


def get_overlap_mapping() -> Dict[str, List[str]]:
    """Return cached mapping, computing it if necessary."""
    if _CACHE_JSON.exists():
        with _CACHE_JSON.open("r", encoding="utf-8") as f:
            return json.load(f)

    mapping = _build_mapping()
    _CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE_JSON.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    return mapping


if __name__ == "__main__":
    m = get_overlap_mapping()
    print(f"Computed mapping for {len(m):d} historical counties → modern overlaps") 