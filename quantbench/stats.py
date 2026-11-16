"""Is the accuracy loss of a quantized model real, or just noise?

All comparisons are paired: every variant is evaluated on the same questions and the
same text windows, so we resample items, not variants independently.
"""

from math import comb

import numpy as np


def bootstrap_mean_ci(values, n_boot=10_000, alpha=0.05, seed=0):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)


def paired_accuracy_diff(base_correct, quant_correct, n_boot=10_000, alpha=0.05, seed=0):
    """Accuracy difference (quantized minus baseline) with a paired bootstrap CI."""
    diff = np.asarray(quant_correct, float) - np.asarray(base_correct, float)
    return bootstrap_mean_ci(diff, n_boot, alpha, seed)


def mcnemar_exact(base_correct, quant_correct):
    """Exact McNemar test on the discordant pairs (two sided p value)."""
    base = np.asarray(base_correct, bool)
    quant = np.asarray(quant_correct, bool)
    b = int(np.sum(base & ~quant))   # baseline right, quantized wrong
    c = int(np.sum(~base & quant))   # baseline wrong, quantized right
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, p)), b, c


def perplexity_ratio_ci(base_nll, quant_nll, ntok, n_boot=10_000, alpha=0.05, seed=0):
    """Ratio PPL_quant / PPL_base with a paired bootstrap over text windows."""
    base_nll, quant_nll, ntok = map(lambda a: np.asarray(a, float), (base_nll, quant_nll, ntok))
    point = float(np.exp((quant_nll.sum() - base_nll.sum()) / ntok.sum()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ntok), size=(n_boot, len(ntok)))
    ratios = np.exp((quant_nll[idx].sum(1) - base_nll[idx].sum(1)) / ntok[idx].sum(1))
    lo, hi = np.quantile(ratios, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
