# Summary of Previous OSS Validation Workflow

This document summarizes the results and validity of the previous positional accuracy validation workflow (now archived). The goal was to assess the claim: **90% of parcel centroids are within 10 km of their true 18th-century ground position.** While the workflow provided some useful insights, it had significant limitations due to biased anchor selection, making absolute accuracy claims indefensible.

## Key Results from Latest Run (with ~260 Anchors)
- **Parcels Evaluated**: 1,436 (96% of the 1,496 C&P-matched benchmark parcels).
- **Anchored Components**: 66/66 multi-parcel connected components (blocks).
- **Median Shift**: Not explicitly reported, but implied to be low based on percentiles.
- **90th-Percentile Shift**: 0.94 km (95% CI: 0.92–0.95 km, via 1,000 bootstrap iterations).
- **Maximum Block Shift**: 4.18 km.
- **Other Metrics**: All shifts were well under 10 km, with tight confidence intervals suggesting high precision in the computed adjustments.

These results were generated using a least-squares solver (in scripts like `network_adjustment.py` and `rigorous_network_adjustment.py`) that minimized weighted residuals to anchors while enforcing edge rigidity in an adjacency graph.

## What Was Valid
- **Internal Consistency**: The small shifts (e.g., 90th percentile ~1 km) indicate that the OSS polygons have strong *relative* accuracy within connected components. Parcels in the same block are well-aligned to each other, with minimal distortion needed to fit the anchors. This is useful for detecting local errors or inconsistencies in digitization.
- **Coverage and Methodology**: The workflow successfully anchored all relevant components, covering 96% of the benchmark. The bootstrap analysis provided reliable uncertainty estimates for the relative shifts, and the edge-rigidity constraint (min boundary length 50 m) effectively modeled parcel relationships.
- **Potential for Diagnostics**: The results highlight areas with larger relative errors (e.g., the 4.18 km max), which could flag specific blocks for manual review, even if not absolute.

## What Was Not Valid (Limitations and Biases)
- **Lack of Independence**: Anchors were selected by identifying features (e.g., creek mouths) *inside* OSS parcel clusters using QGIS and Google Maps. This made anchors dependent on OSS positions, leading to circular validation: residuals were small because anchors were chosen to align with OSS, not independent truth. It couldn't detect systematic errors (e.g., entire blocks misplaced by >10 km).
- **No Absolute Accuracy**: The shifts measured *adjustments to biased anchors*, not distances to true historical positions. The 90th-percentile shift of 0.94 km proves consistency but not the ≤10 km claim against ground truth.
- **Statistical Power Misdirected**: Tight CIs are for relative metrics only; without independent inputs, the analysis lacks power to certify the absolute claim. Potential for confirmation bias, where the method inherently produces optimistic (low) errors.
- **Other Issues**: Anchors from `anchor_template.csv` weren't purely text-based (unlike the partial set in `manual_anchor_worksheet.csv`), and the workflow didn't account for historical changes in features (though σ uncertainties helped somewhat).

## Recommendations
- Treat these results as a baseline for *relative* quality, not absolute validation.
- For the new independent workflow, compare sample-based absolute errors against these relative shifts to quantify bias (e.g., if new errors are much larger, it confirms the old method's optimism).
- Archive reference: All old scripts, data (e.g., `anchor_template.csv`, `network_adjustment_shifts.csv`), and plots are in `archive/`.

Overall, the previous data showed excellent internal OSS consistency but nothing conclusive about absolute positional accuracy. Proceed with the independent validation TODO to address this. 