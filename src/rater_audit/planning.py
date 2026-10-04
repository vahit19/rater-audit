"""Sizing the controls before a project starts: gold items per rater, canary items per rater, acceptance plan."""
from math import ceil

from .stats import Z95, wilson_ci, binom_cdf, find_plan, accept_prob


def gold_per_rater(half_width: float = 0.05, expected_accuracy: float = 0.90) -> int:
    """Gold items each rater must answer to know their accuracy within +/- half_width (95%)."""
    p = expected_accuracy
    return ceil(Z95 * Z95 * p * (1 - p) / (half_width * half_width))


def flag_probability(n: int, match_rate: float, baseline: float, margin: float) -> float:
    """Chance a rater with the given true match rate is flagged on n canaries
    (flag when the Wilson lower bound of k/n exceeds baseline + margin)."""
    flag_ks = [k for k in range(n + 1) if wilson_ci(k, n)[0] > baseline + margin]
    if not flag_ks:
        return 0.0
    return 1.0 - binom_cdf(min(flag_ks) - 1, n, match_rate) if min(flag_ks) > 0 else 1.0


def canaries_per_rater(copy_rate: float = 0.5, human_match: float = 0.10, margin: float = 0.10,
                       power: float = 0.90, n_max: int = 400) -> dict:
    """Smallest number of canary items per rater that flags a rater copying a model on copy_rate of items
    with at least the given power, and the chance an honest rater is flagged at that size."""
    copier = copy_rate + (1 - copy_rate) * human_match
    for n in range(5, n_max + 1):
        if flag_probability(n, copier, human_match, margin) >= power:
            return {"canaries": n, "power": flag_probability(n, copier, human_match, margin),
                    "honest_false_flag": flag_probability(n, human_match, human_match, margin)}
    raise ValueError("no size found within n_max")


def plan(n_raters: int, items: int, ratings_per_item: int = 3, half_width: float = 0.05, expected_accuracy: float = 0.90,
         copy_rate: float = 0.5, max_defect: float = 0.05, good: float = 0.02, bad: float = 0.08) -> dict:
    """A project-level quality plan: how much of the work is spent on controls."""
    g = gold_per_rater(half_width, expected_accuracy)
    c = canaries_per_rater(copy_rate)
    n, k = find_plan(good, bad)
    work = items * ratings_per_item
    overhead = n_raters * (g + c["canaries"])
    return {"gold_per_rater": g, "canaries_per_rater": c["canaries"], "canary_power": c["power"],
            "canary_false_flag": c["honest_false_flag"], "acceptance_review": n, "acceptance_max_defects": k,
            "accept_prob_good": accept_prob(n, k, good), "accept_prob_bad": accept_prob(n, k, bad),
            "control_ratings": overhead, "production_ratings": work, "control_share": overhead / (overhead + work),
            "max_defect": max_defect}
