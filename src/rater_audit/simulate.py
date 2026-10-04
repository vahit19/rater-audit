"""Rating panels with a known answer, so every audit can be checked against the truth.

Binary task: each item has a true label (acceptable / not acceptable). Most items are easy. On hard items a
plausible wrong answer pulls typical raters the same way, while a few deep experts usually get it right.
"""
from dataclasses import dataclass

import numpy as np

from .stats import krippendorff_alpha_nominal


@dataclass
class Panel:
    labels: np.ndarray      # items x raters, np.nan where a rater did not rate the item
    truth: np.ndarray       # true label per item
    hard: np.ndarray        # bool per item
    deep: np.ndarray        # bool per rater


def simulate_panel(rng, n_items=1000, hard_frac=0.2, n_raters=10, raters_per_item=3, n_deep=2,
                   p_easy=0.96, p_hard_deep=0.85, p_hard_typical=0.30, p_hard_equal=None, hard=None) -> Panel:
    """p_hard_equal: if set, every rater has this accuracy on hard items (the no-skill-difference control).
    hard: optional bool mask of hard items; overrides hard_frac."""
    truth = rng.integers(0, 2, n_items)
    if hard is None:
        hard = np.zeros(n_items, bool)
        hard[rng.choice(n_items, int(round(hard_frac * n_items)), replace=False)] = True
    deep = np.zeros(n_raters, bool)
    deep[:n_deep] = True
    p_hard = np.where(deep, p_hard_deep, p_hard_typical) if p_hard_equal is None else np.full(n_raters, p_hard_equal)
    p_correct = np.where(hard[:, None], p_hard[None, :], p_easy)
    correct = rng.random((n_items, n_raters)) < p_correct
    labels = np.where(correct, truth[:, None], 1 - truth[:, None]).astype(float)
    mask = np.zeros((n_items, n_raters), bool)
    for i in range(n_items):
        mask[i, rng.choice(n_raters, raters_per_item, replace=False)] = True
    labels[~mask] = np.nan
    return Panel(labels, truth, hard, deep)


def majority_labels(labels: np.ndarray) -> np.ndarray:
    """Majority of binary labels per item; NaN on a tie or an empty row."""
    ones = np.nansum(labels, axis=1)
    n = np.sum(~np.isnan(labels), axis=1)
    out = np.where(ones * 2 > n, 1.0, np.where(ones * 2 < n, 0.0, np.nan))
    out[n == 0] = np.nan
    return out


def agreement_rates(labels: np.ndarray) -> np.ndarray:
    """Per rater: share of items where the rater matches the majority of the other raters (ties skipped)."""
    n_items, n_raters = labels.shape
    agree = np.zeros(n_raters)
    total = np.zeros(n_raters)
    for j in range(n_raters):
        rated = ~np.isnan(labels[:, j])
        others = labels[rated].copy()
        others[:, j] = np.nan
        m = majority_labels(others)
        ok = ~np.isnan(m)
        agree[j] = np.sum(labels[rated, j][ok] == m[ok])
        total[j] = ok.sum()
    return np.divide(agree, total, out=np.full(n_raters, np.nan), where=total > 0)


def label_accuracy(labels: np.ndarray, truth: np.ndarray, rows: np.ndarray, raters=None) -> float:
    """Share of individual labels that are correct, on the given items and raters."""
    sub = labels[rows] if raters is None else labels[rows][:, raters]
    t = np.broadcast_to(truth[rows][:, None], sub.shape)
    ok = ~np.isnan(sub)
    return float(np.mean(sub[ok] == t[ok]))


def alpha(labels: np.ndarray) -> float:
    return krippendorff_alpha_nominal(labels)


def pairwise_agreement(labels: np.ndarray) -> float:
    """Raw share of rater pairs on the same item that gave the same label."""
    agree = pairs = 0
    for row in labels:
        r = row[~np.isnan(row)]
        m = len(r)
        if m < 2:
            continue
        ones = r.sum()
        agree += ones * (ones - 1) / 2 + (m - ones) * (m - ones - 1) / 2
        pairs += m * (m - 1) / 2
    return float(agree / pairs)
