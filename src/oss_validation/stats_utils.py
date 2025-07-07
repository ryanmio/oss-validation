# -*- coding: utf-8 -*-
"""Statistical helper functions (CIs, bootstrap) for validation reports."""
from __future__ import annotations

import math
from typing import Callable, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Wilson score confidence interval for a binomial proportion
# ---------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Return Wilson score interval (lower, upper) for a proportion.

    Parameters
    ----------
    successes : int
        Number of positive outcomes.
    n : int
        Sample size.
    alpha : float, default 0.05
        1 − confidence level. alpha=0.05 → 95 % CI.
    """
    if n == 0:
        return (float("nan"), float("nan"))

    # z critical value (two-tailed); hard-code common levels to avoid scipy dependency
    if abs(alpha - 0.05) < 1e-6:
        z = 1.959963984540054
    elif abs(alpha - 0.01) < 1e-6:
        z = 2.5758293035489004
    elif abs(alpha - 0.10) < 1e-6:
        z = 1.6448536269514722
    else:
        # Approximate using scipy if available else normal quantile via sampling
        try:
            from math import erf
            # Use inverse error function approximation via numpy's erfinv if present
            import numpy as np
            z = math.sqrt(2) * np.erfinv(1 - alpha)
        except Exception:
            # fallback to Monte Carlo (less precise but acceptable for rare alphas)
            z = abs(np.quantile(np.random.standard_normal(1_000_000), 1 - alpha / 2))

    phat = successes / n
    denom = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    adj = z * math.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    lower = (centre - adj) / denom
    upper = (centre + adj) / denom
    return (max(0.0, lower), min(1.0, upper))


# ---------------------------------------------------------------------------
# Bootstrap confidence interval for an arbitrary statistic
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: Sequence[float],
    func: Callable[[np.ndarray], float] | None = None,
    reps: int = 5000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Return percentile bootstrap CI (lower, upper) for statistic *func*.

    Parameters
    ----------
    data : sequence of floats
    func : statistic function – defaults to np.mean
    reps : int, default 5000
        Number of bootstrap resamples.
    alpha : float, default 0.05
        1 − confidence level.
    """
    if func is None:
        func = np.mean

    arr = np.asarray(data, dtype=float)
    n = len(arr)
    if n == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(42)
    stats = np.empty(reps)
    for i in range(reps):
        sample = rng.choice(arr, size=n, replace=True)
        stats[i] = func(sample)

    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return (lower, upper)
