"""Experiment 4: batch acceptance. How often does a bad batch pass, and a good one fail?

Stage: client acceptance testing. The contract says the defect rate must be at most 5%. A batch with 2% defects
should pass; a batch with 8% should not. Common quick checks are compared with a plan sized for both risks
(at most 5% chance to reject a 2% batch, at most 10% chance to accept an 8% batch).
Control: the exact binomial numbers are checked against 20,000 simulated batches per point.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rater_audit.stats import accept_prob, find_plan, zero_defect_n, wilson_ci  # noqa: E402
from rater_audit.plotting import plt, save, write_json, BLUE, RED, GRAY, ORANGE, GREEN  # noqa: E402

GOOD, BAD = 0.02, 0.08
plan = find_plan(GOOD, BAD, producer_risk=0.05, consumer_risk=0.10)
plans = {"check 20, allow 0": (20, 0), "check 20, allow 1": (20, 1), "check 50, allow 2": (50, 2),
         f"check {plan[0]}, allow {plan[1]} (sized)": plan}
rates = np.linspace(0, 0.20, 81)

out = {"good_rate": GOOD, "bad_rate": BAD, "sized_plan": plan, "zero_defect_n_for_5pct": zero_defect_n(0.05),
       "plans": {}}
rng = np.random.default_rng(0)
max_gap = 0.0
for name, (n, c) in plans.items():
    exact_bad, exact_good = accept_prob(n, c, BAD), accept_prob(n, c, GOOD)
    sim_bad = float(np.mean(rng.binomial(n, BAD, 20000) <= c))
    sim_good = float(np.mean(rng.binomial(n, GOOD, 20000) <= c))
    max_gap = max(max_gap, abs(sim_bad - exact_bad), abs(sim_good - exact_good))
    out["plans"][name] = dict(n=n, c=c, bad_batch_passes=exact_bad, good_batch_fails=1 - exact_good,
                              pass_10pct_batch=accept_prob(n, c, 0.10))
out["control_max_gap_exact_vs_simulated"] = max_gap
out["example_2_of_100"] = wilson_ci(2, 100)
write_json("exp4_acceptance", out)

fig, ax = plt.subplots(figsize=(6.6, 4.0))
colors = [RED, ORANGE, GRAY, BLUE]
for (name, (n, c)), col in zip(plans.items(), colors):
    ax.plot(rates * 100, [accept_prob(n, c, r) for r in rates], color=col, lw=2.2 if col == BLUE else 1.6, label=name)
ax.axvspan(0, GOOD * 100, color=GREEN, alpha=0.08)
ax.axvspan(BAD * 100, 20, color=RED, alpha=0.06)
ax.axvline(5, color="k", lw=0.8, ls=":")
ax.text(5.2, 0.95, "contract: 5%", fontsize=8.5)
ax.set_xlabel("true defect rate of the batch (%)")
ax.set_ylabel("chance the batch is accepted")
ax.set_title("Small checks pass bad batches")
ax.legend(fontsize=8.5)
save(fig, "fig4_acceptance")
print(plan, {k: {m: round(v, 3) for m, v in d.items() if isinstance(v, float)} for k, d in out["plans"].items()}, max_gap)
