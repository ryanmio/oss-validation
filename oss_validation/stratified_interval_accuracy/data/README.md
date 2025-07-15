# Data Folder: Manual-Only Tier-2 Positional Accuracy

This folder contains the **stratified sample and the working file** for the manual-only positional-accuracy audit.

## Files

| File | Purpose |
|------|---------|
| `stratified100_sample_master.csv` | Original, untouched stratified sample of 100 grants (seed = 42). Includes `oss_id` for convenience. **Never edit this file.** |
| `stratified100_manual_geocoding.csv` | Working file where you enter `manual_lat_lon` and `notes` for each grant you can geolocate. This is the only file the analysis scripts read. |

## Column Descriptions

### Columns present in **both** files
- `cp_id` – grant identifier
- `oss_id` (**master only**) – OSS polygon id matching the grant (joined from `spatial_validation.csv`)
- `cp_county` – county
- `decade` – decade of grant date
- `cp_raw` – full deed text
- `suggested_anchor_phrase` – regex-extracted phrase hinting at a geographic feature
- `anchor_score` – **(removed)** legacy heuristic; no longer included in data

### Columns **only in `stratified100_manual_geocoding.csv`**
- `manual_lat_lon` – **edit me** – “lat, lon” (decimal degrees) of the geolocated anchor
- `notes` – notes on confidence, sources, or why the grant can’t be geolocated

### Columns removed
- `initial_20` – legacy timing flag (no longer needed)
- `anchor_score` – legacy heuristic score (removed to avoid confusion)
- `manual_lat_lon` in the master file (removed to avoid accidental edits)

## Workflow
1. **Reference** `stratified100_sample_master.csv` for the definitive list of grants.
2. **Edit** `stratified100_manual_geocoding.csv` as you manually geolocate features.
3. Run analysis scripts (distance & CI) – they read only the working file.

## Provenance
Both files were generated from the same stratified sampling script (seed = 42). The master file is untouched, while the working file is meant for iterative editing. 