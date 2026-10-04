"""Small, dependency-light statistics used across the audits."""
from math import comb, log, ceil, sqrt

import numpy as np

Z95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def binom_cdf(c: int, n: int, p: float) -> float:
    """P(X <= c) for X ~ Binomial(n, p)."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(c + 1))


def accept_prob(n: int, c: int, defect_rate: float) -> float:
    """Probability that a batch passes a check of n items allowing at most c defects."""
    return binom_cdf(c, n, defect_rate)


def zero_defect_n(max_rate: float, confidence: float = 0.95) -> int:
    """Items to check, all clean, to show the defect rate is below max_rate at the given confidence."""
    return ceil(log(1 - confidence) / log(1 - max_rate))


def find_plan(good_rate: float, bad_rate: float, producer_risk: float = 0.05,
              consumer_risk: float = 0.10, n_max: int = 2000) -> tuple[int, int]:
    """Smallest (n, c) that accepts a good batch with prob >= 1 - producer_risk
    and accepts a bad batch with prob <= consumer_risk."""
    for n in range(1, n_max + 1):
        for c in range(0, n + 1):
            if accept_prob(n, c, bad_rate) > consumer_risk:
                break
            if accept_prob(n, c, good_rate) >= 1 - producer_risk:
                return n, c
    raise ValueError("no plan found within n_max")


def krippendorff_alpha_nominal(matrix: np.ndarray) -> float:
    """Krippendorff's alpha for nominal labels. matrix: items x raters, np.nan for missing."""
    values = np.unique(matrix[~np.isnan(matrix)])
    index = {v: i for i, v in enumerate(values)}
    o = np.zeros((len(values), len(values)))
    for row in matrix:
        r = row[~np.isnan(row)]
        m = len(r)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[index[r[a]], index[r[b]]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    d_o = n - np.trace(o)
    d_e = (n * n - (n_c**2).sum()) / (n - 1)
    return 1.0 - d_o / d_e if d_e > 0 else 1.0


def interval(values) -> tuple[float, float, float]:
    """Mean and central 95% range of repeated simulation results."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
