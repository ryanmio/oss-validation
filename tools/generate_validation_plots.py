from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""Utility to generate plots for OSS validation paper.
Run with Python 3 in an environment where matplotlib is available.
Outputs PNGs to reports/plots/ directory (created if missing).
"""

def ensure_outdir() -> Path:
    out = Path('reports/plots')
    out.mkdir(parents=True, exist_ok=True)
    return out

def area_error_plots():
    out = ensure_outdir()
    df = pd.read_csv('oss_validation/area_accuracy_validation/area_validation.csv')
    vals = np.sort(df['pct_diff'].abs())

    # ECDF
    ecdf = np.arange(1, len(vals) + 1) / len(vals)
    plt.figure(figsize=(6, 4))
    plt.step(vals, ecdf, where='post')
    plt.xlabel('|% acreage error|')
    plt.ylabel('Empirical CDF')
    plt.title(f'ECDF of acreage % error (N={len(vals)})')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(out / 'area_error_ecdf.png', dpi=300)
    plt.close()

    # Histogram
    plt.figure(figsize=(6, 4))
    bins = np.logspace(np.log10(0.1), np.log10(vals.max() + 1), 50)
    plt.hist(vals, bins=bins, color='steelblue', edgecolor='black')
    plt.xscale('log')
    plt.xlabel('|% acreage error| (log scale)')
    plt.ylabel('Frequency')
    plt.title('Histogram of acreage % error')
    plt.tight_layout()
    plt.savefig(out / 'area_error_hist.png', dpi=300)
    plt.close()


def tier2_anchor_ecdf():
    out = ensure_outdir()
    anchor_path = Path('archive/tier2_final_with_manual_filtered.csv')
    df = pd.read_csv(anchor_path)
    mask = (df['exclude_reason'].isna()) | (df['exclude_reason'] == '')
    if 'coord_source' in df.columns:
        mask &= df['coord_source'] != 'none'
    if 'manual_bad' in df.columns:
        mask &= ~(df['manual_bad'].fillna(False))
    dists = df.loc[mask, 'final_d_km'].dropna().values
    dists = np.sort(dists)
    ecdf = np.arange(1, len(dists) + 1) / len(dists)

    plt.figure(figsize=(6, 4))
    plt.step(dists, ecdf, where='post')
    plt.xlabel('Great-circle error (km)')
    plt.ylabel('Empirical CDF')
    plt.title(f'ECDF Tier-2 anchor error (N={len(dists)})')
    plt.tight_layout()
    plt.savefig(out / 'tier2_anchor_ecdf.png', dpi=300)
    plt.close()


def main():
    area_error_plots()
    tier2_anchor_ecdf()
    print('Plots saved in reports/plots/')

if __name__ == '__main__':
    main() 