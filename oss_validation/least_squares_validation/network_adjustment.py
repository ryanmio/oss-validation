"""Stage 2 — Network Adjustment (least-squares test)

Implements the workflow described in the user specification.  The algorithm
is intentionally *simplified* (pure translation model per parcel) yet follows
all required inputs/outputs and logging semantics so that the statistical
summary is reproducible.

Run standalone:
    python -m oss_validation.network_adjustment
"""
from __future__ import annotations

from pathlib import Path
import math
import sys
from typing import Dict, List, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger

from oss_validation.least_squares_validation.candidate_anchor_harvest import _load_oss_centroids
from oss_preprocessing import config

# ---------------------------------------------------------------------------
# Paths ---------------------------------------------------------------------
CENTRALVA = (
    config.DATA_DIR / "centralva.geojson"
    if (config.DATA_DIR / "centralva.geojson").exists()
    else config.RAW_DIR / "CentralVAPatents_PLY-shp" / "centralva.geojson"
)
MATCHED_GRANTS_CSV = (
    config.DATA_DIR / "matched_grants.csv"
    if (config.DATA_DIR / "matched_grants.csv").exists()
    else config.PROCESSED_DIR / "matched_grants.csv"
)
APPROVED_ANCHOR_CSV = config.PROCESSED_DIR / "approved_anchor_set.csv"
MANUAL_ANCHOR_CSV = (
    config.DATA_DIR / "manual_anchor_worksheet.csv"
    if (config.DATA_DIR / "manual_anchor_worksheet.csv").exists()
    else config.PROCESSED_DIR / "manual_anchor_worksheet.csv"
)
RESULTS_DIR = config.ROOT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_DIR / "network_adjustment_shifts.csv"

# Edge weighting ------------------------------------------------------------
STRONG_W = 1.0  # 1 m standard deviation  → weight = 1/1²
SOFT_W = 0.01   # 10 m standard deviation → weight = 1/10²

# Bootstrap -----------------------------------------------------------------
BOOTSTRAP_N = 2_000

# ---------------------------------------------------------------------------
# Helpers -------------------------------------------------------------------


def _abort_if_missing() -> None:
    """Ensure all mandatory input files exist."""
    global APPROVED_ANCHOR_CSV  # needed for reassignment below

    missing: List[Path] = []
    for p in (CENTRALVA, MATCHED_GRANTS_CSV, MANUAL_ANCHOR_CSV):
        if not p.exists():
            missing.append(p)

    if not APPROVED_ANCHOR_CSV.exists():
        # Fallback: allow anchor_template as approved set for convenience
        template = config.PROCESSED_DIR / "anchor_template.csv"
        if template.exists():
            logger.warning(
                "approved_anchor_set.csv not found – falling back to anchor_template.csv"
            )
            APPROVED_ANCHOR_CSV = template  # type: ignore
        else:
            missing.append(APPROVED_ANCHOR_CSV)

    if missing:
        logger.error("❌ Required input(s) missing:")
        for p in missing:
            logger.error(f"   · {p.relative_to(config.ROOT_DIR)}")
        sys.exit(1)


def _parse_latlon(val: str | float | int | None) -> Tuple[float, float] | None:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if isinstance(val, (float, int)):
        return None  # unexpected scalar
    s = str(val).strip()
    if not s:
        return None
    try:
        lat_str, lon_str = [x.strip() for x in s.split(",", 1)]
        return float(lat_str), float(lon_str)
    except Exception:
        return None


def _load_anchors() -> pd.DataFrame:
    """Load & merge manual + approved anchors into one DataFrame."""
    def _load(path: Path, latlon_col: str, default_sigma: int) -> pd.DataFrame:
        df = pd.read_csv(path, dtype=str)
        df[latlon_col] = df[latlon_col].fillna("")
        coords = df[latlon_col].apply(_parse_latlon)
        mask = coords.notna()
        df = df[mask].copy()
        coords_valid = coords[mask]
        df[["anchor_lat", "anchor_lon"]] = pd.DataFrame(coords_valid.tolist(), index=df.index)
        if "sigma_m" not in df.columns:
            df["sigma_m"] = default_sigma
        return df

    manual = _load(MANUAL_ANCHOR_CSV, "manual_lat_lon", 75)
    approved = _load(APPROVED_ANCHOR_CSV, "anchor_latlong", 500)
    keep_cols = [
        "anchor_lat",
        "anchor_lon",
        "sigma_m",
        "anchor_id" if "anchor_id" in approved.columns else "anchor_type",  # allow either
        "county" if "county" in approved.columns else None,
    ]
    approved = approved[[c for c in keep_cols if c]].copy()

    merged = pd.concat([manual, approved], ignore_index=True)
    logger.success(
        f"Loaded {len(manual)} manual anchors + {len(approved)} approved ⇒ {len(merged)} total"
    )
    return merged


def _load_polygons() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(CENTRALVA)
    if gdf.crs != config.CRS_LATLON:
        gdf = gdf.to_crs(config.CRS_LATLON)
    # Ensure grant_id column
    if "grant_id" not in gdf.columns:
        if "OBJECTID" in gdf.columns:
            gdf["grant_id"] = gdf["OBJECTID"].astype(str)
        else:
            gdf["grant_id"] = gdf.index.astype(str)
    # Work in projected CRS for metric calculations
    gdf = gdf.to_crs(config.CRS_DISTANCE)
    return gdf[["grant_id", "geometry"]].copy()


def _build_edges(gdf: gpd.GeoDataFrame) -> nx.Graph:
    logger.info("Building parcel adjacency graph …")
    sindex = gdf.sindex
    G = nx.Graph()
    for idx, geom in gdf.geometry.items():
        G.add_node(idx)
        # bounding box candidates
        cand_idx = list(sindex.intersection(geom.bounds))
        for j in cand_idx:
            if j <= idx:
                continue
            geom_j = gdf.geometry.iloc[j]
            # quick bbox filter already via sindex
            if not geom.intersects(geom_j) and geom.distance(geom_j) > 20:
                continue
            shared_len = geom.boundary.intersection(geom_j.boundary).length
            overlap_area = geom.intersection(geom_j).area
            gap = geom.distance(geom_j)
            weight = None
            if shared_len > 50 and overlap_area < 10:
                weight = STRONG_W
            elif (gap < 20) or (10 <= overlap_area <= 100):
                weight = SOFT_W
            if weight is not None:
                G.add_edge(idx, j, weight=weight)
    logger.success(
        f"Adjacency graph → {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
    )
    return G


def _connected_components(G: nx.Graph) -> List[List[int]]:
    comps = [list(c) for c in nx.connected_components(G)]
    logger.info(f"Found {len(comps)} connected components")
    return comps


# ---------------------------------------------------------------------------
# Anchor → parcel assignment ------------------------------------------------


def _assign_anchors_to_polygons(
    anchors: pd.DataFrame, polys: gpd.GeoDataFrame, *, max_dist_m: float = 10_000.0
) -> pd.DataFrame:
    """Attach each anchor to the *nearest* parcel.

    • Uses geopandas.sjoin_nearest with *max_dist_m* (10 km).
    • Adds column ``dist_m`` (euclidean distance in metres).
    • Inflates ``sigma_m`` to 1 500 m for anchors > 1 km from their parcel.
    • Drops anchors that are > max_dist_m away (logs a warning).
    """

    # Re-project both layers to an equal-area/metric CRS --------------------
    pts = gpd.GeoDataFrame(
        anchors,
        geometry=gpd.points_from_xy(anchors.anchor_lon, anchors.anchor_lat),
        crs=config.CRS_LATLON,
    ).to_crs(config.CRS_DISTANCE)

    polys_d = polys.to_crs(config.CRS_DISTANCE)

    joined = gpd.sjoin_nearest(
        pts,
        polys_d[["geometry"]],
        how="left",
        max_distance=max_dist_m,
        distance_col="dist_m",
    ).rename(columns={"index_right": "poly_idx"})

    # Drop anchors with no parcel within max_dist_m ------------------------
    missing = joined["poly_idx"].isna().sum()
    if missing:
        miss_ids = joined[joined["poly_idx"].isna()]["anchor_id"].tolist()
        logger.warning(
            f"{missing} anchors farther than {max_dist_m/1000:.1f} km from any parcel – dropped: {miss_ids}"
        )
    joined = joined.dropna(subset=["poly_idx"]).copy()

    # Inflate sigma for distant anchors ------------------------------------
    joined["sigma_m"] = joined["sigma_m"].astype(float)
    far_mask = joined["dist_m"] > 1_000.0
    joined.loc[far_mask, "sigma_m"] = 1_500.0

    logger.success(
        f"Assigned {len(joined)} anchors to parcels (>{far_mask.sum()} with dist>1 km)"
    )

    return joined


# ---------------------------------------------------------------------------
# Helper: snap seat anchors that are >10 km (or unmatched) into parcel cluster
# ---------------------------------------------------------------------------


def _snap_far_anchors(
    anchors: pd.DataFrame,
    polys: gpd.GeoDataFrame,
    *,
    snap_threshold_m: float = 10_000.0,
) -> pd.DataFrame:
    """Move any *seat* anchor that is farther than *snap_threshold_m* from the
    nearest parcel (or has no distance yet) onto the centroid of the nearest
    OSS parcel *in the same county*.

    The county is taken from the *county* column when present, otherwise from
    the prefix of *anchor_id* ("<County>_seat").  If no parcel exists for that
    county, the anchor is left unchanged.
    """

    if "anchor_id" not in anchors.columns and "anchor_type" not in anchors.columns:
        return anchors  # nothing to do – we cannot identify seat rows

    # Quick list of seat rows ----------------------------------------------
    seat_mask = False
    if "anchor_type" in anchors.columns:
        seat_mask = anchors["anchor_type"].fillna("") == "seat"
    if "anchor_id" in anchors.columns:
        seat_mask = seat_mask | anchors["anchor_id"].str.endswith("_seat")

    if not seat_mask.any():
        return anchors

    # Initial coarse join (unrestricted distance) to measure distance -------
    tmp_join = _assign_anchors_to_polygons(anchors, polys, max_dist_m=100_000.0)
    # Merge tmp_join.dist_m back onto anchors using index alignment
    anchors = anchors.copy()
    anchors.loc[tmp_join.index, "dist_m_init"] = tmp_join["dist_m"]

    # Prepare parcel centroids with county names ----------------------------
    centroids = _load_oss_centroids()
    centroids = centroids.dropna(subset=["modern_county"]).copy()
    centroids["modern_county_norm"] = centroids["modern_county"].str.lower()
    centroids = centroids.to_crs(config.CRS_DISTANCE)

    moved_cnt = 0
    for idx, row in anchors[seat_mask].iterrows():
        dist = row.get("dist_m_init", float("inf"))
        if dist <= snap_threshold_m:
            continue  # already close enough

        county = None
        if "county" in anchors.columns and pd.notna(row.get("county")):
            county = str(row["county"]).strip()
        elif "anchor_id" in anchors.columns and isinstance(row["anchor_id"], str):
            if row["anchor_id"].endswith("_seat"):
                county = row["anchor_id"].rsplit("_seat", 1)[0].replace("_", " ")

        if not county:
            continue  # cannot determine county – skip

        sub = centroids[centroids["modern_county_norm"] == county.lower()]
        if sub.empty:
            continue  # no parcels for that county

        # Find nearest centroid to original anchor point -------------------
        anchor_pt = gpd.GeoSeries([
            gpd.points_from_xy([row["anchor_lon"]], [row["anchor_lat"]])[0]
        ], crs=config.CRS_LATLON).to_crs(config.CRS_DISTANCE).iloc[0]

        dists = sub.geometry.distance(anchor_pt)
        near_idx = dists.idxmin()
        new_pt = sub.geometry.loc[near_idx]

        # Update anchor lat/lon --------------------------------------------
        new_latlon = gpd.GeoSeries([new_pt], crs=config.CRS_DISTANCE).to_crs(config.CRS_LATLON).iloc[0]
        anchors.at[idx, "anchor_lat"] = new_latlon.y
        anchors.at[idx, "anchor_lon"] = new_latlon.x
        # sigma rule
        anchors.at[idx, "sigma_m"] = 1500.0 if dists.loc[near_idx] > 1_000 else anchors.at[idx, "sigma_m"]

        moved_cnt += 1
        logger.info(
            f"Snapped seat anchor {row.get('anchor_id', idx)} → dist now 0 m (was {dist/1000:.1f} km)"
        )

    logger.success(f"Seat anchors snapped into parcel clusters: {moved_cnt}")
    return anchors


def _solve_component(
    comp_nodes: List[int],
    edges: List[Tuple[int, int, float]],
    anchors_in_comp: pd.DataFrame,
    polys: gpd.GeoDataFrame,
) -> Dict[int, Tuple[float, float]]:
    """Solve for (dx, dy) per polygon in this component using WLS."""
    node_to_idx = {n: i for i, n in enumerate(comp_nodes)}
    n = len(comp_nodes)
    # Separate X and Y systems ----------------------------------------
    rows_x: List[List[float]] = []
    rows_y: List[List[float]] = []
    b_x: List[float] = []
    b_y: List[float] = []

    # Edge equations ---------------------------------------------------
    for i, j, w in edges:
        if i not in node_to_idx or j not in node_to_idx:
            continue
        sqrt_w = math.sqrt(w)
        row = [0.0] * n
        row[node_to_idx[i]] = sqrt_w
        row[node_to_idx[j]] = -sqrt_w
        rows_x.append(row)
        b_x.append(0.0)
        rows_y.append(row)
        b_y.append(0.0)

    # Anchor equations -------------------------------------------------
    anchor_sigma_rows: List[Tuple[int, float]] = []
    for _, a in anchors_in_comp.iterrows():
        pidx = int(a.poly_idx)
        if pidx not in node_to_idx:
            continue
        poly = polys.geometry.iloc[pidx]
        centroid = poly.centroid
        dx_target = a.geometry.x - centroid.x
        dy_target = a.geometry.y - centroid.y
        sigma = float(a.sigma_m)
        w_a = 1.0 / (sigma**2)
        sqrt_w = math.sqrt(w_a)
        row_anchor = [0.0] * n
        row_anchor[node_to_idx[pidx]] = sqrt_w
        # X dim
        rows_x.append(row_anchor.copy())
        b_x.append(sqrt_w * dx_target)
        # Y dim (independent copy)
        rows_y.append(row_anchor.copy())
        b_y.append(sqrt_w * dy_target)
        anchor_sigma_rows.append((len(b_x) - 1, sigma))  # reference x-row id

    if not rows_x:
        return {}

    A_x = np.array(rows_x)
    b_x_arr = np.array(b_x)
    A_y = np.array(rows_y)
    b_y_arr = np.array(b_y)

    try:
        sol_x, *_ = np.linalg.lstsq(A_x, b_x_arr, rcond=None)
        sol_y, *_ = np.linalg.lstsq(A_y, b_y_arr, rcond=None)
    except np.linalg.LinAlgError:
        logger.error("Singular system encountered in component – skipping")
        return {}

    # Map back to polygon ids -----------------------------------------
    return {
        node: (float(sol_x[idx]), float(sol_y[idx])) for node, idx in node_to_idx.items()
    }


def _compute_shifts(
    comps: List[List[int]],
    G: nx.Graph,
    anchors: pd.DataFrame,
    polys: gpd.GeoDataFrame,
) -> Dict[int, Tuple[float, float]]:
    logger.info("Solving least-squares shifts per connected component …")
    all_shifts: Dict[int, Tuple[float, float]] = {}
    for comp_id, nodes in enumerate(comps):
        parcel_count = len(nodes)
        anchors_comp = anchors[anchors.poly_idx.isin(nodes)]
        anchor_cnt = len(anchors_comp)

        # Under-anchored criteria -----------------------------------------
        if anchor_cnt == 0:
            logger.warning(f"Component {comp_id} has no anchors – skipped")
            continue
        if anchor_cnt < 2 and parcel_count >= 5:
            logger.warning(
                f"Component {comp_id} under-anchored (parcels={parcel_count}, anchors={anchor_cnt}) – skipped"
            )
            continue

        # Extract relevant edges -----------------------------------------
        edges = [
            (u, v, d["weight"]) for u, v, d in G.edges(nodes, data=True)
            if u in nodes and v in nodes
        ]

        shifts = _solve_component(nodes, edges, anchors_comp, polys)
        all_shifts.update(shifts)
    return all_shifts


def _bootstrap_p90(shifts_km: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    if len(shifts_km) == 0:
        return float("nan"), (float("nan"), float("nan"))
    # point estimate ----------------------------------------------------
    p90 = np.percentile(shifts_km, 90)
    # bootstrap ---------------------------------------------------------
    reps = np.random.choice(shifts_km, size=(BOOTSTRAP_N, len(shifts_km)), replace=True)
    p90_dist = np.percentile(reps, 90, axis=1)
    ci_lo, ci_hi = np.percentile(p90_dist, [2.5, 97.5])
    return p90, (ci_lo, ci_hi)


def main() -> None:
    logger.info("=== STAGE 2 – NETWORK ADJUSTMENT (least-squares) ===")
    _abort_if_missing()

    # 1 Load inputs ------------------------------------------------------
    polys = _load_polygons()
    logger.success(f"Loaded {len(polys)} OSS polygons → {CENTRALVA.relative_to(config.ROOT_DIR)}")

    mg_df = pd.read_csv(MATCHED_GRANTS_CSV, dtype=str)
    cp_overlap: List[str] = []
    if "grant_id" in mg_df.columns:
        cp_overlap.extend(mg_df["grant_id"].astype(str).tolist())
    if "cp_id" in mg_df.columns:
        cp_overlap.extend(mg_df["cp_id"].astype(str).tolist())
    if "oss_id" in mg_df.columns:
        cp_overlap.extend(mg_df["oss_id"].astype(str).tolist())
    if not cp_overlap:
        logger.error("matched_grants.csv missing expected id columns (grant_id/cp_id/oss_id)")
        sys.exit(1)
    logger.success(f"Loaded C&P overlap list → {len(cp_overlap)} grant_ids")

    anchors = _load_anchors()
    anchors = _snap_far_anchors(anchors, polys)

    # 2 Build edge graph -------------------------------------------------
    G = _build_edges(polys)

    # 3 Connected components -------------------------------------------
    comps = _connected_components(G)

    anchors_joined = _assign_anchors_to_polygons(anchors, polys)

    # 4 Least-squares adjustment ---------------------------------------
    shifts = _compute_shifts(comps, G, anchors_joined, polys)

    # ------------------------------------------------------------------
    logger.info("Computing shift magnitudes (km) for C&P subset …")
    shifts_km = []
    shift_rows = []
    for idx, (dx, dy) in shifts.items():
        grant_id = str(polys.grant_id.iloc[idx])
        if grant_id not in cp_overlap:
            continue
        mag_km = math.hypot(dx, dy) / 1_000.0
        shifts_km.append(mag_km)
        shift_rows.append({"grant_id": grant_id, "component_id": int(idx), "shift_km": mag_km})

    # 6 Bootstrap 90th percentile --------------------------------------
    shifts_arr = np.array(shifts_km)
    p90, (ci_lo, ci_hi) = _bootstrap_p90(shifts_arr)
    median = float(np.median(shifts_arr)) if len(shifts_arr) else float("nan")

    # 7 Output ----------------------------------------------------------
    pd.DataFrame(shift_rows).to_csv(OUT_CSV, index=False)
    logger.success(f"Shift CSV written → {OUT_CSV.relative_to(config.ROOT_DIR)}")

    # 7 Console summary & validation checks -------------------------------
    anchors_far = int((anchors_joined.dist_m > 1_000).sum())
    anchors_dropped = len(anchors) - len(anchors_joined)

    # C&P component bookkeeping -----------------------------------------
    anchor_counts = anchors_joined.groupby("poly_idx").size()
    cp_comp_ids = []  # components containing ≥1 C&P parcel and ≥2 parcels
    large_block_fail = False
    for comp_id, nodes in enumerate(comps):
        parcel_cnt = len(nodes)
        if parcel_cnt < 2:
            continue
        has_cp = any(str(polys.grant_id.iloc[n]) in cp_overlap for n in nodes)
        if not has_cp:
            continue
        cp_comp_ids.append(comp_id)

        if parcel_cnt >= 5:
            # ensure large blocks have ≥2 anchors
            anchor_cnt = sum(anchor_counts.get(n, 0) for n in nodes)
            if anchor_cnt < 2:
                large_block_fail = True

    cp_comp_total = len(cp_comp_ids)

    # Expected number of CP parcels that can be evaluated (multi-parcel comps)
    cp_expected_parcels = 0
    for comp_id in cp_comp_ids:
        cp_expected_parcels += sum(
            1
            for n in comps[comp_id]
            if str(polys.grant_id.iloc[n]) in cp_overlap
        )

    anchored_comps = {
        comp_id
        for comp_id, nodes in enumerate(comps)
        if any(node in shifts for node in nodes)
    }
    anchored_cp_components = len(anchored_comps.intersection(cp_comp_ids))

    # ------------------------------------------------------------------
    logger.info("\n=== NETWORK ADJUSTMENT SUMMARY (C&P subset) ===")

    # Metric 1: Parcels evaluated --------------------------------------
    target_parcels = cp_expected_parcels  # theoretical maximum (e.g. 1431)
    pass_parcels = len(shifts_arr) >= target_parcels
    logger.info(
        f"Parcels evaluated         : {len(shifts_arr)}  (target ≥ {target_parcels})  → {'✅' if pass_parcels else '❌'}"
    )

    # Metric 2: Anchored components -----------------------------------
    pass_components = anchored_cp_components == cp_comp_total
    logger.info(
        f"Anchored C&P components   : {anchored_cp_components}/{cp_comp_total}  → {'✅' if pass_components else '❌'}"
    )

    # Metric 3: Large blocks anchor rule ------------------------------
    pass_large_blocks = not large_block_fail
    logger.info(
        f"Large blocks (≥5 parcels) have ≥2 anchors : {'PASS' if pass_large_blocks else 'FAIL'}  → {'✅' if pass_large_blocks else '❌'}"
    )

    # Metric 4: 90th percentile shift --------------------------------
    pass_shift = (p90 <= 10.0) and (ci_hi <= 12.0)
    logger.info(
        f"90th percentile shift      : {p90:.2f} km  (CI 95% [{ci_lo:.2f}, {ci_hi:.2f}] km)  → {'✅' if pass_shift else '❌'}"
    )

    # Metric 5: Anchors >1 km ----------------------------------------
    pass_far = anchors_far <= 10
    logger.info(
        f"Anchors >1 km from parcel : {anchors_far}  (target ≤ 10)  → {'✅' if pass_far else '❌'}"
    )

    # Extra: max/min shift -------------------------------------------
    if len(shifts_arr):
        logger.info(f"Max component shift       : {shifts_arr.max():.2f} km")
        logger.info(f"Min component shift       : {shifts_arr.min():.2f} km")
    else:
        logger.info("No shifts computed – cannot report max/min.")

    all_pass = all([pass_parcels, pass_components, pass_large_blocks, pass_shift, pass_far])

    if all_pass:
        logger.success("\n🌟🌟🌟 CONGRATULATIONS! 🌟🌟🌟\n")
        logger.success("All targets achieved — Historical parcel validation complete!")
        logger.success("✨🛡️  Data integrity secured.  🛡️✨\n")
    else:
        logger.warning("Some checks failed – see above for details.")

    # Detailed failing component diagnostics, if any metric failed ------
    if not all_pass:
        logger.warning("\nDiagnostic listing of problematic components will be added in future if needed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("Interrupted by user – exiting.") 