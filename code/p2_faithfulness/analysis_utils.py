"""
Shared analysis utilities for P2 faithfulness track (Leïla Benjelloun).

bootstrap_ci      — 95% CI on nanmean via percentile bootstrap
get_valid_mask    — boolean mask of non-NaN baseline NLI scores
locked_nli_stats  — mean + CI restricted to that locked set

The locked-denominator approach fixes the comparison problem in the prompt
sweep: baseline NLI skips 21 answers too short to score (NaN); stricter
prompts produce longer answers and score all 100, shifting the denominator.
By restricting every condition to the same 79 samples where baseline NLI is
non-NaN, the means are directly comparable.
"""

import numpy as np


def bootstrap_ci(scores, n_bootstrap: int = 2000, ci: float = 0.95, seed: int = 0):
    """
    Percentile bootstrap CI for the nanmean of `scores`.

    NaN values are excluded from resampling (treated as missing, not zero).
    Returns (mean, lower, upper).  All three are NaN if no valid scores exist.
    """
    arr = np.asarray(scores, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        nan = float("nan")
        return nan, nan, nan
    rng = np.random.default_rng(seed)
    boot_means = np.fromiter(
        (np.mean(rng.choice(valid, size=len(valid), replace=True))
         for _ in range(n_bootstrap)),
        dtype=float,
        count=n_bootstrap,
    )
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_means, 100 * alpha))
    hi = float(np.percentile(boot_means, 100 * (1.0 - alpha)))
    return float(np.nanmean(arr)), lo, hi


def get_valid_mask(baseline_nli_scores):
    """
    Return a boolean array marking samples where baseline NLI is non-NaN.
    Pass this mask to locked_nli_stats() to fix the denominator for all
    cross-condition comparisons.
    """
    return ~np.isnan(np.asarray(baseline_nli_scores, dtype=float))


def locked_nli_stats(scores, valid_mask, n_bootstrap: int = 2000,
                     ci: float = 0.95, seed: int = 0):
    """
    Compute mean + bootstrap CI restricted to the locked valid set.

    Args:
        scores:     per-sample NLI scores for this condition (may contain NaN)
        valid_mask: boolean mask from get_valid_mask() — same 79 baseline indices
        n_bootstrap, ci, seed: forwarded to bootstrap_ci()

    Returns:
        (mean, lower_ci, upper_ci) on the locked subset only
    """
    arr = np.asarray(scores, dtype=float)
    locked = arr[valid_mask]
    return bootstrap_ci(locked, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
