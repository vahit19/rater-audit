"""Experiment 8: do the findings depend on the assumed skill levels?

Each assumption of the simulation is moved one at a time around its default, and three findings are recomputed:
(1) majority correct on hard items, (2) chance that the agreement screen drops a deep expert, (3) gain on hard items
from routing the weak item type to the two best raters on a 30-item qualification test. The point is to show where
each finding holds and where it stops holding, not to pick favourable values.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rater_audit.simulate import simulate_panel, majority_labels, agreement_rates  # noqa: E402
from rater_audit.plotting import plt, save, write_json, BLUE, RED, GREEN, GRAY  # noqa: E402

SEEDS, N_ITEMS, QUAL_N = 100, 2000, 30
DEFAULT = dict(p_easy=0.96, p_hard_deep=0.85, p_hard_typical=0.30, n_deep=2, raters_per_item=3, hard_frac=0.20)
SWEEP = {"p_hard_typical": [0.15, 0.30, 0.45, 0.55], "p_hard_deep": [0.65, 0.75, 0.85, 0.95],
         "n_deep": [1, 2, 3, 4], "raters_per_item": [3, 5], "p_easy": [0.90, 0.96, 0.99],
         "hard_frac": [0.10, 0.20, 0.30, 0.40]}
LABEL = {"p_hard_typical": "typical rater right on hard items", "p_hard_deep": "deep expert right on hard items",
         "n_deep": "deep experts in a panel of 10", "raters_per_item": "raters per item",
         "p_easy": "everyone right on easy items", "hard_frac": "share of hard items"}


def one(seed, cfg):
    rng = np.random.default_rng(seed)
    p = simulate_panel(rng, n_items=N_ITEMS, **cfg)
    m = majority_labels(p.labels)
    hard_acc = float(np.nanmean(m[p.hard] == p.truth[p.hard]))
    agree = agreement_rates(p.labels)
    dropped_deep = float(p.deep[np.argsort(agree)[:2]].any())
    # routing: all hard items go to the two raters with the best score on a 30-item hard qualification test
    p_hard = np.where(p.deep, cfg["p_hard_deep"], cfg["p_hard_typical"])
    score = (rng.random((QUAL_N, len(p_hard))) < p_hard[None, :]).sum(axis=0) + rng.random(len(p_hard)) * 1e-6
    rev = np.argsort(-score)[:2]
    rows = np.where(p.hard)[0]
    ok = rng.random((len(rows), 2)) < p_hard[rev][None, :]
    lab = np.where(ok, p.truth[rows][:, None], 1 - p.truth[rows][:, None])
    routed_acc = float(np.mean(lab[:, 0] == p.truth[rows]))      # split votes go to the top reviewer
    return hard_acc, dropped_deep, routed_acc - hard_acc


out = {"default": DEFAULT, "seeds": SEEDS, "sweeps": {}}
for name, values in SWEEP.items():
    rows = []
    for v in values:
        cfg = dict(DEFAULT, **{name: v})
        if cfg["n_deep"] >= 10:
            continue
        r = np.array([one(s, cfg) for s in range(SEEDS)])
        rows.append({"value": v, "majority_hard_acc": float(r[:, 0].mean()),
                     "screen_drops_deep_expert": float(r[:, 1].mean()), "routing_gain_hard": float(r[:, 2].mean())})
    out["sweeps"][name] = rows
    print(name, [(x["value"], round(x["majority_hard_acc"], 2), round(x["screen_drops_deep_expert"], 2),
                  round(x["routing_gain_hard"], 2)) for x in rows], flush=True)
write_json("exp8_sensitivity", out)

fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharey=True)
for ax, (name, rows) in zip(axes.ravel(), out["sweeps"].items()):
    x = [r["value"] for r in rows]
    ax.plot(x, [r["majority_hard_acc"] for r in rows], "o-", color=RED, label="majority right on hard items")
    ax.plot(x, [r["screen_drops_deep_expert"] for r in rows], "s-", color=GRAY, label="screen drops a deep expert")
    ax.plot(x, [r["routing_gain_hard"] for r in rows], "^-", color=GREEN, label="gain from routing (hard items)")
    ax.axvline(DEFAULT[name], color=BLUE, lw=0.8, ls=":")
    ax.set_title(LABEL[name], fontsize=10)
    ax.set_ylim(-0.05, 1.05)
axes[0, 0].legend(fontsize=8, loc="lower left")
fig.suptitle("Each assumption moved one at a time (dotted line: default)", fontsize=11.5)
save(fig, "fig8_sensitivity")
