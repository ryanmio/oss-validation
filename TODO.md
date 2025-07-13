# OSS Independent Validation TODO

This TODO outlines steps to properly validate the positional accuracy of OSS polygons using independent anchors, building on prior discussions.

1. **Prepare Data and Environment**
   - Review `data/processed/matched_grants.csv` (the 1,496 C&P-matched OSS grants) and ensure it includes deed text summaries or links for anchor extraction. If deed texts are missing, add a column by pulling from C&P volumes or a database.
   - Review `data/processed/manual_anchor_worksheet.csv` (your original text-based manual anchors). Salvage any truly independent entries (e.g., those not influenced by OSS positions) as a starting point—aim for 20–30 to bootstrap the process.
   - Install/update dependencies if needed (e.g., `pandas`, `geopandas`, `scipy` for stats, `shapely` for geometry). Update `pyproject.toml` accordingly.
   - Create a new script skeleton in `src/oss_validation/` (e.g., `independent_validation.py`) to house the new workflow logic.

2. **Implement Stratified Sampling**
   - Rewrite or copy/adapt the archived `stratified_sampling.py` to draw a sample of 50–100 grants from `matched_grants.csv`. Stratify by historical county (`cp_county`) and decade (derived from grant year) to ensure diversity.
     - Target: Proportional allocation (e.g., more samples from populous counties/decades).
     - Output: A CSV like `data/processed/independent_sample.csv` with sampled grant IDs, counties, decades, and deed text excerpts.
   - Goal: This sample represents the 720 components and 1,436 benchmark parcels statistically, providing power for extrapolation (e.g., binomial CI for the 90% claim).

3. **Manually Geolocate Independent Anchors**
   - For each sampled grant, parse the deed text to identify 1–2 locatable features (e.g., "mouth of X creek", "at Y's corner").
   - Geolocate them independently using sources like GNIS, historical maps (e.g., USGS topo archives), or Google Maps—**do not open the OSS GeoJSON or QGIS layer**.
     - Assign uncertainty (σ) based on feature type (e.g., 500 m for confluences, 1 km for seats).
     - Record in a new CSV: `data/processed/independent_anchors.csv` with columns for grant ID, feature type, lat/lon, σ, and notes.
   - If a grant's text is ambiguous, skip or flag it (aim for 80%+ success rate). Use the salvaged entries from `manual_anchor_worksheet.csv` to fill gaps where possible.
   - Time estimate: 1–2 hours per 10 grants; total ~5–10 hours for 50 grants.

4. **Compute Absolute Positional Errors**
   - Load OSS polygons from `data/raw/CentralVAPatents_PLY-shp/centralva.geojson` and compute centroids for the sampled grants.
   - For each sample, calculate great-circle distances (in km) from the OSS centroid to the independent anchor(s). If multiple anchors per grant, average or use the minimum.
     - Adapt logic from the archived `tier2_distance.py` (great-circle formula) and `tier2_positional_accuracy.py` (centroid loading).
   - Output: Update `data/processed/independent_sample.csv` with a `distance_km` column, plus any residuals or confidence metrics (e.g., incorporating σ).

5. **Analyze and Validate the Claim**
   - Compute key statistics on the `distance_km` values:
     - Median, 90th percentile, max error.
     - Proportion of samples ≤10 km.
     - Bootstrap 95% CI (n=1,000 iterations) for the 90th percentile and proportion (to confirm if ≥90% hold with statistical confidence).
   - If components are involved (e.g., for blocks with multiple samples), adapt a simplified version of the archived network adjustment logic (e.g., from `network_adjustment.py`) to estimate block-level shifts using only independent anchors— but keep it optional for this sample-based approach.
   - Check for biases: Group stats by county/decade; test for systematic errors (e.g., via histogram or t-tests).
   - Output: A report CSV/JSON in `results/` (e.g., `independent_validation_stats.csv`) and plots in a new `plots/` (e.g., error histogram, map of errors).

6. **Refine and Extrapolate**
   - If the sample's 90th-percentile error >10 km or CI is wide, iterate: Increase sample size, refine geolocation (e.g., add a second reviewer for 20% of samples), or adjust the claim (e.g., to 8 km or 85%).
   - Extrapolate to full dataset: Use the sample proportion to estimate overall compliance (with CI bounds). If >90% of samples pass, claim it's defensible for the full set (noting sampling assumptions).
   - Edge cases: Handle unanchored components by sampling a few manually; document any exclusions.

7. **Document and Test**
   - Update `README.md` with the new workflow, assumptions (e.g., independence of anchors, σ values), and results.
   - Add tests in `tests/` (e.g., for sampling, distance calcs, stats).
   - Commit changes to the `independent-validation` branch and push to origin if ready.
   - Run the full pipeline and review results—aim for a defensible claim like "Based on independent sampling, 90% of centroids are within X km (95% CI: Y–Z)". 