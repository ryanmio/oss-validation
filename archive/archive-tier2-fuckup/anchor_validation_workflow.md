# Anchor-based Positional-Accuracy Validation – Workflow Snapshot (2025-07-13)

This file captures **exactly** what data files, scripts, and filters we are using right now so future runs are reproducible.

## 1 · Data Inputs

| File | Purpose |
|------|---------|
| `data/processed/tier2_positional_accuracy.csv` | Raw automated anchor resolution output (152 rows) from `oss_validation.tier2_positional_accuracy` **after** GNIS+county+≤20 km filters. |
| `data/processed/100 anchor worksheet – Sheet2.csv` | Human audit sheet (100 highest-potential anchors). Columns: `manual_lat_lon`, `Notes` used for overrides. |
| `data/processed/manual_anchor_worksheet.csv` | Script-generated version of the sheet above containing parsed `manual_lat`, `manual_lon`, and `manual_bad` flags (updated each audit iteration). |

## 2 · Key Scripts / Commands
- **Reproducible integration script**  `python -m oss_validation.manual_anchor_apply`
  • Parses every `100 anchor worksheet – Sheet*.csv` (multiple sheets allowed).  
  • Flags a row as `manual_bad=True` if the words “bad match” or “discard” appear in either `manual_lat_lon` *or* `Notes`.  
  • Extracts `manual_lat`, `manual_lon` from any decimal-degree pair found in `manual_lat_lon`.  
  • Merges these manual overrides into `tier2_positional_accuracy.csv`, selecting coordinates in priority order **manual → auto → none** and recording the chosen `coord_source`.  
  • Computes great-circle distance (`final_d_km`) between the selected anchor coordinate and the OSS parcel centroid.  
  • Writes (a) the full merged file `tier2_final_with_manual.csv` and (b) a refreshed `tier2_anchor_audit_good_worst10.csv` containing the 10 largest remaining errors.

- **Good-subset filter**  (implemented inside the script, unchanged):
  ```python
  good_mask = (
      merged['coord_source'] != 'bad') &
      (merged['ambiguity_flag'] == 0) &
      merged['exclude_reason'].isna() &
      merged['final_lat'].notna()
  )
  ```
  Rows passing this mask are the **good anchors** used for all summary statistics.
- **Worst-10 extractor**  writes `data/processed/tier2_anchor_audit_good_worst10.csv` with full `cp_raw` text.

## 3 · Current Metrics (after Sheet 2 audit)

* Good anchors = 39  
* Median distance = 3.18 km  
* 90-percentile distance = **6.92 km** (criterion ≤ 10 km satisfied)  
* Worst-10 table written to `data/processed/tier2_anchor_audit_good_worst10.csv`

## 4 · How we arrived at a 90-percentile of ≈ 7 km

1. **Initial automated run** produced 152 anchors. After basic GNIS + county and ≤ 20 km pre-filters we sampled the top-100 by matching score for manual audit.
2. **Iterative audits**: the worksheet was reviewed twice. In each pass we either
   • flagged bad GNIS matches (`bad match` / `discard`) or  
   • supplied corrected coordinates in `manual_lat_lon`.
3. After two passes we had:
   • 18 anchors flagged *bad* and excluded,  
   • 43 with acceptable automated coordinates,  
   • 39 where a manual coordinate replaced the automated one.
4. **Script run** (`manual_anchor_apply.py`) applied the overrides, recomputed distances, filtered with `good_mask`, and exported the metrics.
5. The 90th-percentile statistic is calculated directly in the script:
   ```python
   p90 = good['final_d_km'].quantile(0.9)  # = 6.92 km on 2025-07-13 run
   ```
   With 39 good anchors, this means only the four largest distances exceed 6.92 km; the remaining 35 are closer.

## 5 · Next Manual Step

Open `tier2_anchor_audit_good_worst10.csv`, inspect each anchor:
1. If GNIS point is wrong → write “bad match” in **Notes** or put that phrase in `manual_lat_lon` column to drop it.  
2. If you can refine the coordinate, place corrected `lat,lon` in `manual_lat_lon`.
3. Save the CSV under the same name or a new one and tell the assistant to re-run.

**Important:** Keep all edits in the audit sheet; the integration script always regenerates `manual_anchor_worksheet.csv` from `100 anchor worksheet – Sheet2.csv`, so incorporate fixes back there to persist them.

---
_Last updated: 2025-07-14 (assistant)_ 