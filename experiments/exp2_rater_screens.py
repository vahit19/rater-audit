"""Experiment 2: which raters does a screen remove?

Stage: production monitoring. A common screen drops the raters who agree least with the panel. The alternative
drops the raters with the lowest accuracy on gold items (items with a known answer, mixed into the queue).
Treatment: 2 of 10 raters are deep experts who are right on hard items where the rest go wrong.
Control: every rater has the same skill, so no screen can find a better or worse rater.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rater_audit.simulate import simulate_panel, agreement_rates, label_accuracy, alpha  # noqa: E402
from rater_audit.stats import interval  # noqa: E402
from rater_audit.plotting import plt, save, write_json, BLUE, RED, GRAY, GREEN  # noqa: E402

SEEDS, N_ITEMS, DROP, GOLD_FRAC = 300, 2000, 2, 0.10


def run(seed, **kw):
    rng = np.random.default_rng(seed)
    p = simulate_panel(rng, n_items=N_ITEMS, **kw)
    n_r = p.labels.shape[1]
    gold = np.zeros(N_ITEMS, bool)          # gold items: a random 10% with a known answer, same difficulty mix
    gold[rng.choice(N_ITEMS, int(GOLD_FRAC * N_ITEMS), replace=False)] = True
    agree = agreement_rates(p.labels)
    gold_acc = np.array([label_accuracy(p.labels, p.truth, gold, [j]) for j in range(n_r)])
    true_hard = np.array([label_accuracy(p.labels, p.truth, p.hard, [j]) for j in range(n_r)]) if p.hard.any() else None
    res = {}
    for name, removed in {"agreement_screen": np.argsort(agree)[:DROP], "gold_accuracy_screen": np.argsort(gold_acc)[:DROP]}.items():
        kept = np.setdiff1d(np.arange(n_r), removed)
        lab = p.labels.copy()
        lab[:, removed] = np.nan
        res[name] = dict(
            removed_a_deep_expert=float(p.deep[removed].any()),
            alpha_before=alpha(p.labels), alpha_after=alpha(lab),
            hard_label_acc_before=label_accuracy(p.labels, p.truth, p.hard),
            hard_label_acc_after=label_accuracy(p.labels, p.truth, p.hard, kept),
            removed_true_hard_acc=float(true_hard[removed].mean()), kept_true_hard_acc=float(true_hard[kept].mean()))
    return res, agree, gold_acc, true_hard, p.deep


def summarize(runs):
    return {s: {k: interval([r[s][k] for r in runs]) for k in runs[0][s]} for s in runs[0]}


out = {}
example = None
for arm, kw in {"treatment_two_deep_experts": {}, "control_equal_skill": {"p_hard_equal": 0.41}}.items():
    runs = []
    for s in range(SEEDS):
        r, agree, gacc, th, deep = run(s, **kw)
        runs.append(r)
        if s == 0 and arm.startswith("treatment"):
            example = (agree, gacc, th, deep)
    out[arm] = summarize(runs)
write_json("exp2_rater_screens", out)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
agree, gacc, th, deep = example
ax[0].scatter(agree[~deep], th[~deep], color=GRAY, s=45, label="typical raters")
ax[0].scatter(agree[deep], th[deep], color=GREEN, s=70, marker="*", label="deep experts")
low = np.argsort(agree)[:DROP]
ax[0].scatter(agree[low], th[low], facecolors="none", edgecolors=RED, s=180, linewidths=1.6, label="dropped by agreement screen")
ax[0].set_xlabel("agreement with the rest of the panel")
ax[0].set_ylabel("true accuracy on hard items")
ax[0].set_title("The screen drops the raters who are right")
ax[0].legend(fontsize=8.5, loc="center left")
t = out["treatment_two_deep_experts"]
labels = ["agreement\nscreen", "gold-accuracy\nscreen"]
before = t["agreement_screen"]["hard_label_acc_before"][0]
after = [t["agreement_screen"]["hard_label_acc_after"][0], t["gold_accuracy_screen"]["hard_label_acc_after"][0]]
ax[1].axhline(before, color=GRAY, ls="--", label=f"before any screen ({before:.0%})")
ax[1].bar(labels, after, color=[RED, BLUE])
for i, v in enumerate(after):
    ax[1].text(i, v + 0.01, f"{v:.0%}", ha="center", fontweight="bold")
ax[1].set_ylim(0, 0.6)
ax[1].set_ylabel("correct labels on hard items, after the screen")
ax[1].set_title("Screen on gold accuracy, not on agreement")
ax[1].legend(fontsize=8.5, loc="upper left")
save(fig, "fig2_rater_screens")
for arm, d in out.items():
    print(arm, {s: {k: round(v[0], 3) for k, v in m.items()} for s, m in d.items()})
