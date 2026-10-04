"""Experiment 6 (real data): what a ranking from expert votes can support, and how many votes it needs.

Stage: evaluation sprint / benchmark validation. Six models are ranked from 3,355 expert pairwise votes on 80
MT-Bench questions, and again from GPT-4's 2,400 votes. Questions are resampled (bootstrap) to put an interval
on each win rate and to split the ranking into tiers that hold in at least 95% of resamples. Then the sprint is
shrunk: with only 10, 20 or 40 questions, how often does it recover the full-data ranking?
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rater_audit.mtbench import load, win_rates, bootstrap_win_rates, tiers  # noqa: E402
from rater_audit.plotting import plt, save, write_json, BLUE, RED, GRAY  # noqa: E402

human, gpt4 = load("human"), load("gpt4_pair")
out = {}
boots = {}
for name, v in {"experts": human, "gpt4_judge": gpt4}.items():
    wr = win_rates(v)
    b = bootstrap_win_rates(v, reps=2000, seed=1)
    boots[name] = (wr, b)
    order = list(wr.index)
    out[name] = {"order": order, "win_rate": {m: float(wr[m]) for m in order},
                 "ci": {m: [float(b[m].quantile(0.025)), float(b[m].quantile(0.975))] for m in order},
                 "tiers": tiers(b, order)}

full_order = out["experts"]["order"]
rng = np.random.default_rng(2)
qs = human["question_id"].unique()
groups = {q: d for q, d in human.groupby("question_id")}


def kendall(a, b):
    pos = {m: i for i, m in enumerate(b)}
    c = d = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if pos[a[i]] < pos[a[j]]:
                c += 1
            else:
                d += 1
    return (c - d) / (c + d)


import pandas as pd  # noqa: E402
shrink = {}
for k in [10, 20, 40, 80]:
    top1 = exact = 0
    taus = []
    for _ in range(500):
        pick = rng.choice(qs, k, replace=k == 80)
        order = list(win_rates(pd.concat([groups[q] for q in pick])).index)
        top1 += order[0] == full_order[0]
        exact += order == full_order
        taus.append(kendall(order, full_order))
    shrink[k] = {"top_model_recovered": top1 / 500, "full_order_recovered": exact / 500, "mean_kendall_tau": float(np.mean(taus))}
out["sprint_size"] = shrink
write_json("exp6_model_ranking", out)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
order = full_order
y = np.arange(len(order))
for off, (name, col, lab) in zip([-0.15, 0.15], [("experts", BLUE, "expert votes"), ("gpt4_judge", RED, "GPT-4 judge votes")]):
    wr = [out[name]["win_rate"][m] for m in order]
    lo = [out[name]["ci"][m][0] for m in order]
    hi = [out[name]["ci"][m][1] for m in order]
    ax[0].errorbar(wr, y + off, xerr=[np.subtract(wr, lo), np.subtract(hi, wr)], fmt="o", color=col, capsize=3, label=lab)
ax[0].set_yticks(y, order)
ax[0].invert_yaxis()
ax[0].set_xlabel("win rate (95% interval over questions)")
ax[0].set_title("Six models, " + str(len(out["experts"]["tiers"])) + " tiers the expert votes can separate")
ax[0].legend(fontsize=8.5, loc="lower right")
ks = list(shrink)
ax[1].plot(ks, [shrink[k]["top_model_recovered"] for k in ks], "o-", color=BLUE, label="top model recovered")
ax[1].plot(ks, [shrink[k]["full_order_recovered"] for k in ks], "o-", color=RED, label="full ranking recovered")
ax[1].set_xticks(ks, ["10", "20", "40", "80 (all," + chr(10) + "resampled)"])
ax[1].set_xlabel("questions in the evaluation sprint")
ax[1].set_ylabel("share of resampled sprints")
ax[1].set_ylim(0, 1.05)
ax[1].set_title("Small sprints do not support a full ranking")
ax[1].legend(fontsize=8.5)
save(fig, "fig6_model_ranking")
print({n: (out[n]["order"], out[n]["tiers"], {m: round(v, 3) for m, v in out[n]["win_rate"].items()}) for n in ("experts", "gpt4_judge")})
print(shrink)
