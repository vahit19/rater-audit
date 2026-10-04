"""Experiment 7 (real data): can the agreement numbers be trusted, and does a rater screen find real outliers?

1. Reproduce the published claim: GPT-4 and experts agree over 80% (ties excluded), the same level as experts
   with each other (Zheng et al., 2023, abstract).
2. Position bias: share of GPT-4 verdicts that changed when the two answers swapped places.
3. Rater screen on real experts. Each expert's agreement with other experts on shared items; a two-sigma screen
   flags the low ones. Control (null): if every expert had the same true agreement rate, how many would the screen
   flag from sampling noise alone, given each expert's number of shared items?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rater_audit.mtbench import load, judge_pairs, agreement  # noqa: E402
from rater_audit.stats import wilson_ci  # noqa: E402
from rater_audit.plotting import plt, save, write_json, BLUE, RED, GRAY  # noqa: E402

MIN_PAIRS, NULL_REPS = 20, 5000
human, gpt4 = load("human"), load("gpt4_pair")
out = {}

hp = judge_pairs(human)
out["expert_vs_expert"] = {"with_ties": agreement(hp.v1, hp.v2, False), "ties_excluded": agreement(hp.v1, hp.v2, True)}
m = human.merge(gpt4[["unit", "vote"]].rename(columns={"vote": "g"}), on="unit")
out["gpt4_vs_expert"] = {"with_ties": agreement(m.vote, m.g, False), "ties_excluded": agreement(m.vote, m.g, True)}

inc = int((gpt4["raw_winner"] == "tie (inconsistent)").sum())
out["gpt4_position_flip"] = {"k": inc, "n": len(gpt4), "rate": inc / len(gpt4), "ci": wilson_ci(inc, len(gpt4))}

long = pd.concat([hp.rename(columns={"j1": "judge"}).assign(agree=hp.v1 == hp.v2)[["judge", "agree"]],
                  hp.rename(columns={"j2": "judge"}).assign(agree=hp.v1 == hp.v2)[["judge", "agree"]]])
per = long.groupby("judge")["agree"].agg(n="size", k="sum").reset_index()
per = per[per.n >= MIN_PAIRS].copy()
per["rate"] = per.k / per.n
ci = [wilson_ci(int(k), int(n)) for k, n in zip(per.k, per.n)]
per["lo"], per["hi"] = [c[0] for c in ci], [c[1] for c in ci]
mean, sd = per.rate.mean(), per.rate.std(ddof=1)
flagged = per[per.rate < mean - 2 * sd]
pooled = per.k.sum() / per.n.sum()
rng = np.random.default_rng(3)
null_flags = []
for _ in range(NULL_REPS):
    r = rng.binomial(per.n.to_numpy(), pooled) / per.n.to_numpy()
    null_flags.append(int(np.sum(r < r.mean() - 2 * r.std(ddof=1))))
obs_var = per.rate.var(ddof=1)
null_var = float(np.mean(pooled * (1 - pooled) / per.n))
out["screen"] = {"experts_with_min_pairs": len(per), "min_pairs": MIN_PAIRS, "pooled_agreement": float(pooled),
                 "flagged": flagged.judge.tolist(), "flagged_n_pairs": flagged.n.tolist(),
                 "null_expected_flags": float(np.mean(null_flags)),
                 "null_share_runs_with_any_flag": float(np.mean(np.array(null_flags) > 0)),
                 "share_of_spread_explained_by_sampling_noise": min(1.0, null_var / obs_var)}
write_json("exp7_judges_real_data", out)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
p = per.sort_values("rate").reset_index(drop=True)
x = np.arange(len(p))
col = [RED if j in set(flagged.judge) else GRAY for j in p.judge]
ax[0].vlines(x, p.lo, p.hi, color=col, lw=1.4)
ax[0].scatter(x, p.rate, color=col, s=16, zorder=3)
ax[0].axhline(mean - 2 * sd, color=RED, ls="--", lw=1, label="two-sigma screen")
ax[0].axhline(pooled, color=BLUE, lw=1, label=f"all experts pooled ({pooled:.0%})")
ax[0].set_xticks([])
ax[0].set_xlabel(f"{len(p)} experts with at least {MIN_PAIRS} shared votes, sorted")
ax[0].set_ylabel("agreement with other experts (95% interval)")
ax[0].set_title("Rater screens act on wide intervals")
ax[0].legend(fontsize=8.5, loc="lower right")
names = ["expert vs\nexpert", "GPT-4 vs\nexpert"]
for i, key in enumerate(["expert_vs_expert", "gpt4_vs_expert"]):
    ax[1].bar(i - 0.18, out[key]["with_ties"][0], 0.36, color=GRAY, label="ties counted" if i == 0 else None)
    ax[1].bar(i + 0.18, out[key]["ties_excluded"][0], 0.36, color=BLUE, label="ties excluded" if i == 0 else None)
    ax[1].text(i - 0.18, out[key]["with_ties"][0] + 0.02, f"{out[key]['with_ties'][0]:.0%}", ha="center", fontsize=9)
    ax[1].text(i + 0.18, out[key]["ties_excluded"][0] + 0.02, f"{out[key]['ties_excluded'][0]:.0%}", ha="center", fontsize=9)
ax[1].bar(2, out["gpt4_position_flip"]["rate"], 0.36, color=RED)
ax[1].text(2, out["gpt4_position_flip"]["rate"] + 0.02, f"{out['gpt4_position_flip']['rate']:.0%}", ha="center", fontsize=9)
ax[1].set_xticks([0, 1, 2], names + ["GPT-4 verdict flips\nwhen answers swap"])
ax[1].set_ylim(0, 1)
ax[1].set_title("Agreement depends on how ties are counted")
ax[1].legend(fontsize=8.5, loc="upper right")
save(fig, "fig7_judges_real_data")
print(out)
