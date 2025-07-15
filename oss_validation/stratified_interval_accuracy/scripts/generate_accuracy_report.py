#!/usr/bin/env python3
"""generate_accuracy_report.py
================================
Create a reproducible Markdown report summarising positional-accuracy statistics
for the Tier-2 validation.

Outputs `stratified_interval_accuracy/results/accuracy_report.md` containing:
  • Descriptive stats for exact manual distances (n≈60)
  • Interval-censored log-normal fit for full 100-grant sample
  • 90th-percentile estimates with 95 % confidence intervals

Run:
    python stratified_interval_accuracy/scripts/generate_accuracy_report.py
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import numpy as _np
import pandas as _pd
from scipy.optimize import minimize as _minimize
from scipy.stats import lognorm as _lognorm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE = Path('.')
RES_DIR = BASE / 'stratified_interval_accuracy' / 'results'
CSV_PATH = RES_DIR / 'stratified100_final.csv'
REPORT_PATH = RES_DIR / 'accuracy_report.md'
B_BOOT = 10_000  # bootstrap replicates for manual subset
B_BOOT_CENS = 1_000  # bootstrap replicates for censored model (slower)
RNG = _np.random.default_rng(42)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _bootstrap_percentile(distances: _np.ndarray, p: float = 90.0, b: int = 10_000) -> tuple[float, float]:
    """Return (lower, upper) 95 % CI via percentile bootstrap for the p-th percentile."""
    boot_stats = [_np.percentile(RNG.choice(distances, size=len(distances), replace=True), p)
                  for _ in range(b)]
    return tuple(_np.percentile(boot_stats, [2.5, 97.5]))


def _neg_loglik(params, exact_obs: _np.ndarray, cens_u: _np.ndarray):
    """Negative log-likelihood for interval-censored log-normal."""
    sigma, mu = params
    if sigma <= 0:
        return _np.inf
    ll_exact = _lognorm.logpdf(exact_obs, s=sigma, scale=_np.exp(mu)).sum()
    ll_cens = _np.log(_lognorm.cdf(cens_u, s=sigma, scale=_np.exp(mu))).sum()
    return -(ll_exact + ll_cens)


def _fit_interval_lognormal(exact: _np.ndarray, cens_u: _np.ndarray):
    """Fit log-normal with interval censoring and return (sigma, mu)."""
    mu0 = _np.mean(_np.log(exact))
    sigma0 = _np.std(_np.log(exact), ddof=1)
    res = _minimize(_neg_loglik, x0=[sigma0, mu0], args=(exact, cens_u))
    if not res.success:
        raise RuntimeError('MLE failed: ' + res.message)
    return tuple(res.x)


def _bootstrap_interval_p90(exact: _np.ndarray, cens_u: _np.ndarray, sigma_hat: float, mu_hat: float,
                            b: int = 1_000) -> tuple[float, float]:
    boot_p90 = []
    for _ in range(b):
        ex_b = RNG.choice(exact, size=len(exact), replace=True)
        cen_b = RNG.choice(cens_u, size=len(cens_u), replace=True)
        try:
            s_b, m_b = _fit_interval_lognormal(ex_b, cen_b)
            boot_p90.append(_lognorm.ppf(0.90, s=s_b, scale=_np.exp(m_b)))
        except RuntimeError:
            pass  # skip failed fits
    return tuple(_np.percentile(boot_p90, [2.5, 97.5]))

# -----------------------------------------------------------------------------
# Main computation
# -----------------------------------------------------------------------------

def main() -> None:
    # Load dataset
    if not CSV_PATH.exists():
        raise SystemExit(f"Results file not found: {CSV_PATH}\nRun the distance-computation script first.")

    df = _pd.read_csv(CSV_PATH)

    # ------------------------------------------------------------------
    # Manual subset stats
    # ------------------------------------------------------------------
    exact = df.query("coord_source == 'manual' and final_d_km.notna()")['final_d_km'].values
    if len(exact) == 0:
        raise SystemExit('No manual distances available – cannot compute statistics.')

    stats_manual = {
        'n': len(exact),
        'min': float(exact.min()),
        'q25': float(_np.percentile(exact, 25)),
        'median': float(_np.median(exact)),
        'mean': float(exact.mean()),
        'q75': float(_np.percentile(exact, 75)),
        'max': float(exact.max()),
        'std': float(exact.std(ddof=1)),
        'p90': float(_np.percentile(exact, 90)),
    }
    stats_manual['p90_ci'] = _bootstrap_percentile(exact, 90, B_BOOT)

    # ------------------------------------------------------------------
    # Interval-censored model stats (full sample)
    # ------------------------------------------------------------------
    cens_u = df.query("coord_source == 'fallback' and final_d_km.notna()")['final_d_km'].values
    sigma_hat, mu_hat = _fit_interval_lognormal(exact, cens_u)
    p90_hat = float(_lognorm.ppf(0.90, s=sigma_hat, scale=_np.exp(mu_hat)))
    ci_low, ci_hi = _bootstrap_interval_p90(exact, cens_u, sigma_hat, mu_hat, B_BOOT_CENS)

    stats_cens = {
        'n_exact': len(exact),
        'n_censored': len(cens_u),
        'sigma': sigma_hat,
        'mu': mu_hat,
        'p90': p90_hat,
        'p90_ci': (ci_low, ci_hi),
    }

    # ------------------------------------------------------------------
    # Build Markdown report
    # ------------------------------------------------------------------
    ts = _dt.datetime.now().strftime('%Y-%m-%d %H:%M')
    md_lines = [
        '# Tier-2 Positional Accuracy Report',
        '',
        f'*Generated: {ts}*',
        '',
        '## Manual Anchors (exact distances)',
        '',
        f'- Sample size: **{stats_manual['n']}**',
        '',
        '| Metric | Value (km) |',
        '|--------|------------|',
        f"| Min | {stats_manual['min']:.2f} |",
        f"| Q25 | {stats_manual['q25']:.2f} |",
        f"| Median | {stats_manual['median']:.2f} |",
        f"| Mean | {stats_manual['mean']:.2f} |",
        f"| Q75 | {stats_manual['q75']:.2f} |",
        f"| Max | {stats_manual['max']:.2f} |",
        f"| Std-dev | {stats_manual['std']:.2f} |",
        f"| P90 | {stats_manual['p90']:.2f} |",
        '',
        f"**P90 95 % CI:** {stats_manual['p90_ci'][0]:.2f} – {stats_manual['p90_ci'][1]:.2f} km",
        '',
        '---',
        '',
        '## Interval-Censored Model (exact + censored)',
        '',
        f"Exact observations: {stats_cens['n_exact']}  ",
        f"Censored observations: {stats_cens['n_censored']}",
        '',
        f"MLE parameters: σ = {stats_cens['sigma']:.3f}, μ = {stats_cens['mu']:.3f}",
        '',
        f"**90th percentile:** {stats_cens['p90']:.2f} km", 
        f"**95 % CI:** {stats_cens['p90_ci'][0]:.2f} – {stats_cens['p90_ci'][1]:.2f} km",
        '',
    ]

    REPORT_PATH.write_text('\n'.join(md_lines))
    print(f'Report written → {REPORT_PATH}')


if __name__ == '__main__':
    main() 