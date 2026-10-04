"""Experiment 1: high agreement, wrong majority.

Treatment: 20% hard items where a plausible wrong answer pulls typical raters the same way (2 of 10 raters are
deep experts). Control: the same panel with no hard items. Truth is known, so agreement can be set against accuracy.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rater_audit.simulate import simulate_panel, majority_labels, alpha, pairwise_agreement  # noqa: E402
from rater_audit.stats import interval  # noqa: E402
from rater_audit.plotting import plt, save, write_json, BLUE, RED, GRAY  # noqa: E402

SEEDS, N_ITEMS = 300, 2000


def run(seed, **kw):
    p = simulate_panel(np.random.default_rng(seed), n_items=N_ITEMS, **kw)
    m = majority_labels(p.labels)
    acc = lambda rows: float(np.nanmean(m[rows] == p.truth[rows])) if rows.any() else float("nan")  # noqa: E731
    return dict(alpha=alpha(p.labels), raw_agreement=pairwise_agreement(p.labels),
                majority_acc_all=acc(np.ones(len(m), bool)), majority_acc_easy=acc(~p.hard),
                majority_acc_hard=acc(p.hard))


def summarize(runs):
    return {k: interval([r[k] for r in runs]) for k in runs[0]}


arms = {"treatment_20pct_hard": {}, "control_no_hard_items": {"hard_frac": 0.0}}
out = {name: summarize([run(s, **kw) for s in range(SEEDS)]) for name, kw in arms.items()}

fracs = [0.0, 0.1, 0.2, 0.3, 0.4]
sweep = {f: summarize([run(s, hard_frac=f) for s in range(100)]) for f in fracs}
out["sweep_hard_fraction"] = {str(f): sweep[f] for f in fracs}
write_json("exp1_agreement_vs_accuracy", out)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
t = out["treatment_20pct_hard"]
names = ["Raw agreement\n(what a dashboard shows)", "Majority correct,\neasy items", "Majority correct,\nhard items"]
vals = [t["raw_agreement"], t["majority_acc_easy"], t["majority_acc_hard"]]
ax[0].bar(names, [v[0] for v in vals], color=[GRAY, BLUE, RED],
          yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], capsize=4)
for i, v in enumerate(vals):
    ax[0].text(i, v[0] + 0.03, f"{v[0]:.0%}", ha="center", fontweight="bold")
ax[0].set_ylim(0, 1.1)
ax[0].set_title("Experts agree; on hard items the majority is wrong")
ax[1].plot(fracs, [sweep[f]["raw_agreement"][0] for f in fracs], "o-", color=GRAY, label="raw agreement")
ax[1].plot(fracs, [sweep[f]["alpha"][0] for f in fracs], "s--", color=GRAY, label="Krippendorff's alpha")
ax[1].plot(fracs[1:], [sweep[f]["majority_acc_hard"][0] for f in fracs[1:]], "o-", color=RED, label="majority correct, hard items")
ax[1].plot(fracs, [sweep[f]["majority_acc_all"][0] for f in fracs], "o-", color=BLUE, label="majority correct, all items")
ax[1].set_xlabel("share of hard items in the batch")
ax[1].set_ylim(0, 1.05)
ax[1].set_title("Agreement stays high while hard items fail")
ax[1].legend(fontsize=8.5, loc="lower left")
save(fig, "fig1_agreement_vs_accuracy")
print({k: {m: round(v[0], 3) for m, v in d.items()} for k, d in out.items() if k != "sweep_hard_fraction"})
