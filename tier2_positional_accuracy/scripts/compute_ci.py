#!/usr/bin/env python3
"""compute_ci.py
================
Bootstrap the 90th-percentile distance error and its 95% confidence interval
for the manual-anchor subset from the stratified 100-grant sample.

Reads:
    tier2_positional_accuracy/results/stratified100_final.csv

Prints to stdout (and can be redirected to a file):
    Sample size, median error, 90th-percentile error, and 95% CI.

Run:
    cd /path/to/repo
    python tier2_positional_accuracy/scripts/compute_ci.py > tier2_positional_accuracy/results/ci_summary.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path

def main():
    # Paths
    base_path = Path('.')
    results_path = base_path / 'tier2_positional_accuracy/results/stratified100_final.csv'
    
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        print("Run compute_distances_manual.py first.")
        return
    
    # Load results
    df = pd.read_csv(results_path)
    
    # Filter for valid distances (is_valid=True and final_d_km is not null)
    valid_mask = (df.get('is_valid', False) == True) & df['final_d_km'].notna()
    valid_df = df[valid_mask]
    
    if len(valid_df) == 0:
        print("No valid manual anchor distances found.")
        print(f"Total grants: {len(df)}")
        print(f"With manual coordinates: {df.get('has_manual_coords', pd.Series(False)).sum()}")
        print(f"With OSS matches: {df.get('has_oss_match', pd.Series(False)).sum()}")
        return
    
    # Extract distances
    distances = valid_df['final_d_km'].values
    n = len(distances)
    
    print(f"Manual-Only Tier-2 Positional Accuracy Analysis")
    print(f"=" * 50)
    print(f"Sample size: {n}")
    
    if n < 3:
        print(f"WARNING: Sample size too small for meaningful CI (n={n})")
        print(f"Basic statistics:")
        print(f"  Min:    {distances.min():.2f} km")
        print(f"  Median: {np.median(distances):.2f} km")
        print(f"  Max:    {distances.max():.2f} km")
        if n >= 2:
            print(f"  Mean:   {distances.mean():.2f} km")
        return
    
    # Basic statistics
    median = np.median(distances)
    p90 = np.percentile(distances, 90)
    
    print(f"Median: {median:.2f} km")
    print(f"90th percentile: {p90:.2f} km")
    
    # Bootstrap CI for 90th percentile
    print(f"\nBootstrap 95% CI for 90th percentile:")
    
    rng = np.random.default_rng(42)  # Fixed seed for reproducibility
    n_bootstrap = 10000
    
    bootstrap_p90s = []
    for _ in range(n_bootstrap):
        bootstrap_sample = rng.choice(distances, size=n, replace=True)
        bootstrap_p90s.append(np.percentile(bootstrap_sample, 90))
    
    ci_low, ci_hi = np.percentile(bootstrap_p90s, [2.5, 97.5])
    
    print(f"  90th percentile: {p90:.2f} km (95% CI {ci_low:.2f}–{ci_hi:.2f} km)")
    
    # Assessment against 10 km threshold
    print(f"\nAssessment:")
    if ci_hi <= 10.0:
        print(f"  ✓ Upper CI bound ({ci_hi:.2f} km) ≤ 10 km threshold")
        print(f"  The 90% confidence claim is statistically defensible.")
    else:
        print(f"  ✗ Upper CI bound ({ci_hi:.2f} km) > 10 km threshold")
        print(f"  More manual anchors needed to strengthen the confidence interval.")
        
        # Estimate required sample size
        target_p90 = p90  # Assume true P90 similar to observed
        # Rough estimate: CI width ∝ 1/√n, so n_new ≈ n_current * (ci_width_current / ci_width_target)²
        current_width = ci_hi - ci_low
        target_ci_hi = 10.0
        target_width = target_ci_hi - ci_low  # Assume same lower bound
        
        if target_width > 0 and current_width > 0:
            estimated_n = n * (current_width / target_width) ** 2
            additional_needed = max(0, int(estimated_n - n))
            print(f"  Estimated additional manual anchors needed: ~{additional_needed}")
    
    # Additional details
    print(f"\nFull sample statistics:")
    print(f"  Min:    {distances.min():.2f} km")
    print(f"  Q25:    {np.percentile(distances, 25):.2f} km")
    print(f"  Median: {median:.2f} km")
    print(f"  Q75:    {np.percentile(distances, 75):.2f} km")
    print(f"  Max:    {distances.max():.2f} km")
    print(f"  Mean:   {distances.mean():.2f} km")
    print(f"  Std:    {distances.std():.2f} km")

if __name__ == '__main__':
    main() 