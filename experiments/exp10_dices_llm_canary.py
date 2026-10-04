"""Experiment 10 (real data, real model outputs): an LLM as safety rater, and canary items built from its real errors.

GPT-4o-mini rates all 350 DICES conversations with a safety prompt (cached in data/llm_cache.jsonl).
1. The model as a judge: agreement with the expert label, next to 3 random real raters.
2. Canary items: on the gold set, the conversations where the model's verdict differs from the expert label are its
   known wrong answers. Each rater is scored on how often they match the model on up to 20 of these.
   Control (real): the 123 real raters, none of whom used the model here; any flag is a false alarm.
   Treatment: one real rater's labels replaced by the model's real verdicts on a share of conversations.
500 random gold splits.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rater_audit.dices import load  # noqa: E402
from rater_audit.llm import unsafe_verdict  # noqa: E402
from rater_audit.stats import wilson_ci, interval  # noqa: E402
from rater_audit.plotting import plt, save, write_json, BLUE, RED, GRAY, ORANGE  # noqa: E402

MODEL, REPS, GOLD_FRAC, N_CANARY, MARGIN = "openai/gpt-4o-mini", 500, 0.20, 20, 0.10
M, items, raters = load()
L = M.to_numpy(dtype=float)
gold = items["gold"].to_numpy()
has = ~np.isnan(L)
n_items, n_raters = L.shape
llm = np.array([unsafe_verdict(MODEL, c, r) for c, r in zip(items["context"], items["response"])], dtype=float)


def sc(pred, rows):
    p, g = pred[rows], gold[rows]
    return {"accuracy": float(np.mean(p == g)), "unsafe_caught": float(np.mean(p[g == 1] == 1)),
            "safe_kept": float(np.mean(p[g == 0] == 0))}


out = {"model": MODEL, "llm_vs_expert": sc(llm, np.arange(n_items))}
rng = np.random.default_rng(0)
draws = []
for _ in range(200):                                   # average over 200 random draws of 3 real raters per item
    maj3 = np.array([int(np.nansum(L[i, rng.choice(np.where(has[i])[0], 3, replace=False)]) >= 2) for i in range(n_items)])
    draws.append(sc(maj3, np.arange(n_items)))
out["random3_vs_expert"] = {k: float(np.mean([d[k] for d in draws])) for k in draws[0]}

ones, cnt = np.nansum(L, axis=1), has.sum(axis=1)


def agreement(labels_j, mask_j):
    o1, oc = ones, cnt
    maj = np.where(o1 * 2 > oc, 1.0, np.where(o1 * 2 < oc, 0.0, np.nan))
    ok = mask_j & ~np.isnan(maj)
    return float(np.mean(labels_j[ok] == maj[ok]))


real_agree = np.array([agreement(L[:, j], has[:, j]) for j in range(n_raters)])
res = {q: {"flagged": [], "agree_pct": []} for q in (0.3, 0.5, 0.7)}
false_rater, false_panel, n_can = [], [], []
for rep in range(REPS):
    rng = np.random.default_rng(1000 + rep)
    g = rng.choice(n_items, int(GOLD_FRAC * n_items), replace=False)
    wrong = g[llm[g] != gold[g]]
    can = rng.choice(wrong, min(N_CANARY, len(wrong)), replace=False)
    n_can.append(len(can))

    def flags(Lm):
        k = np.array([np.sum(Lm[can, j] == llm[can]) for j in range(Lm.shape[1])])
        n = np.array([np.sum(~np.isnan(Lm[can, j])) for j in range(Lm.shape[1])])
        rate = k / np.maximum(n, 1)
        lo = np.array([wilson_ci(int(a), int(b))[0] for a, b in zip(k, n)])
        return lo > np.median(rate) + MARGIN

    f = flags(L)
    false_rater.append(float(f.mean()))
    false_panel.append(float(f.any()))
    for q in res:
        j = rng.integers(n_raters)
        Lc = L.copy()
        copy = rng.random(n_items) < q
        Lc[copy, j] = llm[copy]
        res[q]["flagged"].append(float(flags(Lc)[j]))
        a = agreement(Lc[:, j], ~np.isnan(Lc[:, j]))
        res[q]["agree_pct"].append(float(np.mean(real_agree < a)))

out["canaries_per_split"] = interval(n_can)
out["control_real_raters_false_flag_rate"] = interval(false_rater)
out["control_any_real_rater_flagged"] = interval(false_panel)
out["copier"] = {str(q): {"flagged": interval(v["flagged"]), "agreement_percentile_among_real_raters": interval(v["agree_pct"])}
                 for q, v in res.items()}
write_json("exp10_dices_llm_canary", out)

fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
names = ["GPT-4o-mini\nas rater", "3 random real\nraters (majority)"]
for i, key in enumerate(["llm_vs_expert", "random3_vs_expert"]):
    v = out[key]
    for k, (m, col) in enumerate([("accuracy", BLUE), ("unsafe_caught", RED), ("safe_kept", GRAY)]):
        ax[0].bar(i + (k - 1) * 0.25, v[m], 0.25, color=col, label=m.replace("_", " ") if i == 0 else None)
        ax[0].text(i + (k - 1) * 0.25, v[m] + 0.015, f"{v[m]:.0%}", ha="center", fontsize=8.5)
ax[0].set_xticks([0, 1], names)
ax[0].set_ylim(0, 1.05)
ax[0].set_title("Against the expert safety label")
ax[0].legend(fontsize=8.5, loc="upper right")
qs = list(res)
ax[1].bar([f"copies {q:.0%}" for q in qs], [out["copier"][str(q)]["flagged"][0] for q in qs], color=[GRAY, ORANGE, RED])
ax[1].axhline(out["control_real_raters_false_flag_rate"][0], color=BLUE, ls="--",
              label=f"control: real raters flagged ({out['control_real_raters_false_flag_rate'][0]:.1%})")
for i, q in enumerate(qs):
    ax[1].text(i, out["copier"][str(q)]["flagged"][0] + 0.02, f"{out['copier'][str(q)]['flagged'][0]:.0%}", ha="center", fontsize=9)
ax[1].set_ylim(0, 1.05)
ax[1].set_ylabel("chance the copying rater is flagged")
ax[1].set_title(f"{N_CANARY} canaries from the model's real errors")
ax[1].legend(fontsize=8.5, loc="upper left")
save(fig, "fig10_dices_llm_canary")
print(out)
