#!/usr/bin/env python3
"""compute_interval_ci.py
=================================
Estimate the 90th-percentile positional error (and 95 % CI) using
interval-censored maximum-likelihood on the stratified 100-grant sample.

Exact observations  : rows where `coord_source == 'manual'`  → distance value
Censored observations: rows where `coord_source == 'fallback'` → interval (0, U]
where U is the county half-width already stored in `final_d_km`.

Outputs `results/ci_interval_summary.txt`.
"""

from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import lognorm
from scipy.optimize import minimize

BASE = Path('.')
RES_DIR = BASE / 'stratified_interval_accuracy' / 'results'
CSV_PATH = RES_DIR / 'stratified100_final.csv'
OUT_PATH = RES_DIR / 'ci_interval_summary.txt'

B_BOOT = 1000
RNG = np.random.default_rng(42)

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
print(f"Loading {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

exact = df.query("coord_source == 'manual' and final_d_km.notna()")['final_d_km'].values
censU = df.query("coord_source == 'fallback' and final_d_km.notna()")['final_d_km'].values

print(f"Exact distances:   {len(exact)}")
print(f"Censored (0,U] :   {len(censU)}")

if len(exact) < 5:
    raise SystemExit("Not enough exact observations to fit a distribution.")

# -------------------------------------------------------------------
# Log-likelihood (negative) for interval-censored log-normal
# -------------------------------------------------------------------

def neg_loglik(params, exact_obs, cens_U):
    sigma, mu = params
    if sigma <= 0:
        return np.inf
    ll_exact = lognorm.logpdf(exact_obs, s=sigma, scale=np.exp(mu)).sum()
    ll_cens = np.log(lognorm.cdf(cens_U, s=sigma, scale=np.exp(mu))).sum()
    return -(ll_exact + ll_cens)

# Initial guesses
mu0 = np.mean(np.log(exact))
sigma0 = np.std(np.log(exact), ddof=1)
res = minimize(neg_loglik, x0=[sigma0, mu0], args=(exact, censU))
if not res.success:
    raise RuntimeError("MLE failed: " + res.message)

sigma_hat, mu_hat = res.x
p90_hat = lognorm.ppf(0.90, s=sigma_hat, scale=np.exp(mu_hat))

print(f"MLE fit: sigma={sigma_hat:.3f}, mu={mu_hat:.3f}, P90={p90_hat:.2f} km")

# -------------------------------------------------------------------
# Parametric bootstrap
# -------------------------------------------------------------------
print(f"Bootstrapping ({B_BOOT} replicates)…")
boot_p90 = []
for _ in range(B_BOOT):
    # Resample exact and censored rows with replacement
    ex_b = RNG.choice(exact, size=len(exact), replace=True)
    cen_b = RNG.choice(censU, size=len(censU), replace=True)
    # Re-fit
    res_b = minimize(neg_loglik, x0=[sigma_hat, mu_hat], args=(ex_b, cen_b))
    if res_b.success:
        s_b, m_b = res_b.x
        boot_p90.append(lognorm.ppf(0.90, s=s_b, scale=np.exp(m_b)))

ci_low, ci_hi = np.percentile(boot_p90, [2.5, 97.5])

# -------------------------------------------------------------------
# Save summary
# -------------------------------------------------------------------
summary = (
    f"Interval-censored log-normal fit (exact + censored)\n"
    f"Exact n = {len(exact)}, censored n = {len(censU)}\n"
    f"90th percentile = {p90_hat:.2f} km\n"
    f"95% CI = {ci_low:.2f} – {ci_hi:.2f} km  (B={len(boot_p90)} bootstraps)\n"
)
print(summary)
OUT_PATH.write_text(summary)
print(f"Summary written to {OUT_PATH}") 