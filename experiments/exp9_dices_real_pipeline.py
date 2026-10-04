"""Experiment 9 (real data, real text, expert gold): the quality pipeline on DICES-350.

350 real chatbot conversations, 123 real raters who each rated all of them for safety, and an expert gold label
per conversation. A production batch is recreated by drawing 3 of the real raters per conversation. 20% of
conversations act as the gold set (used for rater scores and selection); every result is measured on the other
80%, which the selection never sees. 200 random splits.

Questions: (1) how agreement compares with accuracy against the expert label; (2) which raters each screen removes
(agreement, gold accuracy, speed, and the dataset authors' own quality list); (3) whether qualified reviewers chosen
on the gold set beat randomly drawn raters at the same cost; (4) what an acceptance review would decide.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rater_audit.dices import load  # noqa: E402
from rater_audit.stats import krippendorff_alpha_nominal, wilson_ci, interval, accept_prob  # noqa: E402
from rater_audit.plotting import plt, save, write_json, BLUE, RED, GRAY, GREEN, ORANGE  # noqa: E402

REPS, GOLD_FRAC, DROP, N_QUAL = 200, 0.20, 12, 5
M, items, raters = load()
L = M.to_numpy(dtype=float)                       # items x raters
gold = items["gold"].to_numpy()
types = items["type"].to_numpy()
n_items, n_raters = L.shape
has = ~np.isnan(L)

# full-panel agreement: each rater vs the majority of the other 122 raters (ties skipped)
ones, cnt = np.nansum(L, axis=1), has.sum(axis=1)
agree = np.zeros(n_raters)
for j in range(n_raters):
    o1 = ones - np.nan_to_num(L[:, j])
    oc = cnt - has[:, j]
    maj = np.where(o1 * 2 > oc, 1.0, np.where(o1 * 2 < oc, 0.0, np.nan))
    ok = has[:, j] & ~np.isnan(maj)
    agree[j] = np.mean(L[ok, j] == maj[ok])
speed = raters["median_time_ms"].to_numpy()
authors = raters["authors_removed"].to_numpy()
full_major = np.where(ones * 2 > cnt, 1, 0)


def scores(final, rows):
    f, g = final[rows], gold[rows]
    return {"accuracy": float(np.mean(f == g)), "unsafe_caught": float(np.mean(f[g == 1] == 1)),
            "safe_kept": float(np.mean(f[g == 0] == 0)), "defect_rate": float(np.mean(f != g))}


def draw(rng, rows, pool, k):
    """For each item, k labels from raters in `pool` who rated it; returns label matrix (rows x k)."""
    out = np.full((len(rows), k), np.nan)
    for a, i in enumerate(rows):
        cand = pool[has[i, pool]]
        out[a] = L[i, rng.choice(cand, k, replace=False)]
    return out


def majority_of(lab):
    return (np.nansum(lab, axis=1) * 2 > np.sum(~np.isnan(lab), axis=1)).astype(int)


runs = []
all_raters = np.arange(n_raters)
for rep in range(REPS):
    rng = np.random.default_rng(rep)
    is_gold = np.zeros(n_items, bool)
    is_gold[rng.choice(n_items, int(GOLD_FRAC * n_items), replace=False)] = True
    held = np.where(~is_gold)[0]
    gidx = np.where(is_gold)[0]
    gold_acc = np.array([np.mean(L[gidx[has[gidx, j]], j] == gold[gidx[has[gidx, j]]]) for j in range(n_raters)])
    held_acc = np.array([np.mean(L[held[has[held, j]], j] == gold[held[has[held, j]]]) for j in range(n_raters)])
    r = {}

    # 1. production batch: 3 random raters per conversation
    lab3 = draw(rng, held, all_raters, 3)
    sub = np.full((len(held), n_raters), np.nan)
    for a, i in enumerate(held):
        js = rng.choice(np.where(has[i])[0], 3, replace=False)
        sub[a, js] = L[i, js]
    pairs = [np.mean(row[~np.isnan(row)][:, None] == row[~np.isnan(row)][None, :]) for row in lab3]
    r["raw_agreement_3"] = float(np.mean([(p * 9 - 3) / 6 for p in pairs]))
    r["alpha_3"] = krippendorff_alpha_nominal(sub)
    maj3 = np.zeros(n_items, int)
    maj3[held] = majority_of(lab3)
    r["random3"] = scores(maj3, held)
    anyf = np.zeros(n_items, int)
    anyf[held] = (np.nansum(lab3, axis=1) >= 1).astype(int)
    r["random3_any_flag"] = scores(anyf, held)
    lab5 = draw(rng, held, all_raters, 5)
    maj5 = np.zeros(n_items, int)
    maj5[held] = majority_of(lab5)
    r["random5"] = scores(maj5, held)
    r["full_panel_123"] = scores(full_major, held)

    # 2. screens: who is removed, and what it does to a 3-rater batch drawn from the rest
    screens = {"agreement": np.argsort(agree)[:DROP], "gold_accuracy": np.argsort(gold_acc + rng.random(n_raters) * 1e-9)[:DROP],
               "speed_fastest": np.argsort(speed)[:DROP], "authors_list": np.where(authors)[0]}
    r["screens"] = {}
    for name, drop in screens.items():
        keep = np.setdiff1d(all_raters, drop)
        m = np.zeros(n_items, int)
        m[held] = majority_of(draw(rng, held, keep, 3))
        r["screens"][name] = {"removed_true_acc": float(held_acc[drop].mean()), "kept_true_acc": float(held_acc[keep].mean()),
                              "batch_accuracy_after": scores(m, held)["accuracy"]}

    # 3. qualified reviewers: the 5 raters with the best accuracy on the gold set decide every held-out conversation
    qual = np.argsort(-(gold_acc + rng.random(n_raters) * 1e-9))[:N_QUAL]
    labq = L[np.ix_(held, qual)]
    mq = np.zeros(n_items, int)
    mq[held] = majority_of(labq)
    r["qualified5"] = scores(mq, held)
    r["qualified_true_acc"] = float(held_acc[qual].mean())
    r["by_type"] = {t: {"random3": scores(maj3, held[types[held] == t])["accuracy"],
                        "qualified5": scores(mq, held[types[held] == t])["accuracy"],
                        "gold_flag_upper": wilson_ci(int(np.sum(maj_g := (np.nansum(L[gidx[types[gidx] == t]], axis=1) * 2 >
                                                                          np.sum(has[gidx[types[gidx] == t]], axis=1)).astype(int)
                                                                 == gold[gidx[types[gidx] == t]])), int(np.sum(types[gidx] == t)))[1]}
                     for t in np.unique(types)}

    # 4. acceptance review of 98 held-out conversations, at most 4 defects (plan for a 5% contract)
    r["acceptance"] = {}
    for name, final in {"random3": maj3, "random3_any_flag": anyf, "random5": maj5, "qualified5": mq}.items():
        pick = rng.choice(held, 98, replace=False)
        r["acceptance"][name] = float(np.sum(final[pick] != gold[pick]) <= 4)
    runs.append(r)


def summ(key_path):
    vals = []
    for r in runs:
        v = r
        for k in key_path:
            v = v[k]
        vals.append(v)
    return interval(vals)


out = {"reps": REPS, "gold_fraction": GOLD_FRAC, "drop": DROP, "n_qualified": N_QUAL,
       "items": int(n_items), "raters": int(n_raters), "expert_unsafe_share": float(gold.mean()),
       "raw_agreement_3": summ(["raw_agreement_3"]), "alpha_3": summ(["alpha_3"])}
for arm in ["random3", "random3_any_flag", "random5", "qualified5", "full_panel_123"]:
    out[arm] = {k: summ([arm, k]) for k in runs[0][arm]}
out["qualified_true_acc"] = summ(["qualified_true_acc"])
out["rater_true_acc_mean"] = float(np.mean([np.mean(L[has[:, j], j] == gold[has[:, j]]) for j in range(n_raters)]))
out["screens"] = {s: {k: summ(["screens", s, k]) for k in runs[0]["screens"][s]} for s in runs[0]["screens"]}
out["by_type"] = {t: {k: summ(["by_type", t, k]) for k in runs[0]["by_type"][t]} for t in runs[0]["by_type"]}
out["type_counts"] = {t: int(np.sum(types == t)) for t in np.unique(types)}
out["acceptance_pass_rate"] = {a: summ(["acceptance", a]) for a in runs[0]["acceptance"]}
out["corr_agreement_vs_accuracy"] = float(np.corrcoef(agree, [np.mean(L[has[:, j], j] == gold[has[:, j]]) for j in range(n_raters)])[0, 1])
write_json("exp9_dices_real_pipeline", out)

# figure
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
arms = [("random3", "3 random\nmajority"), ("random5", "5 random\nmajority"),
        ("random3_any_flag", "3 random\nunsafe if any"), ("qualified5", "5 qualified\nreviewers"),
        ("full_panel_123", "all 123\nmajority")]
x = np.arange(len(arms))
acc = [out[a]["accuracy"][0] for a, _ in arms]
cau = [out[a]["unsafe_caught"][0] for a, _ in arms]
ax[0].bar(x - 0.2, acc, 0.4, color=BLUE, label="agrees with expert label")
ax[0].bar(x + 0.2, cau, 0.4, color=RED, label="expert-unsafe responses caught")
for i in range(len(arms)):
    ax[0].text(x[i] - 0.2, acc[i] + 0.015, f"{acc[i]:.0%}", ha="center", fontsize=8.5)
    ax[0].text(x[i] + 0.2, cau[i] + 0.015, f"{cau[i]:.0%}", ha="center", fontsize=8.5)
ax[0].set_xticks(x, [n for _, n in arms], fontsize=8.5)
ax[0].set_ylim(0, 1.05)
ax[0].set_title("Real ratings vs expert gold (held-out conversations)")
ax[0].legend(fontsize=8.5, loc="upper left")
true_acc = np.array([np.mean(L[has[:, j], j] == gold[has[:, j]]) for j in range(n_raters)])
ax[1].scatter(agree, true_acc, s=14, color=GRAY, label="rater")
low = np.argsort(agree)[:DROP]
ax[1].scatter(agree[low], true_acc[low], s=50, facecolors="none", edgecolors=RED, label="dropped by agreement screen")
ax[1].scatter(agree[authors], true_acc[authors], s=16, marker="x", color=ORANGE, label="dataset authors' removed list")
ax[1].set_xlabel("agreement with the other 122 raters")
ax[1].set_ylabel("accuracy against the expert label")
ax[1].set_title("123 real raters: agreement and accuracy")
ax[1].legend(fontsize=8, loc="lower right")
save(fig, "fig9_dices_real_pipeline")
for k in ["raw_agreement_3", "alpha_3", "random3", "random3_any_flag", "random5", "qualified5", "full_panel_123",
          "screens", "acceptance_pass_rate", "corr_agreement_vs_accuracy", "qualified_true_acc", "rater_true_acc_mean"]:
    print(k, out[k])
print(out["by_type"])
