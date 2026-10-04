"""Experiment 3: the fix. Find the weak item type with gold, qualify experts on it, route it to them.

Stages: gold set (2), qualification (3), multi-layer review and adjudication (6).
Items come in two types. Type A is mostly easy; type B holds most hard items (in practice a task category,
e.g. multi-step clinical questions). Step 1: panel accuracy on gold items, per type; a type is routed when the
upper end of its interval is below the 90% target. Step 2: every rater takes a 30-item qualification test on
type-B gold; the top two become reviewers. Step 3: routed items are decided by those two reviewers.
Control: no hard items, so no type should be routed.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rater_audit.simulate import simulate_panel, majority_labels  # noqa: E402
from rater_audit.stats import wilson_ci, interval  # noqa: E402
from rater_audit.plotting import plt, save, write_json, BLUE, RED, GRAY, GREEN  # noqa: E402

SEEDS, N_ITEMS, GOLD_FRAC, TARGET, QUAL_N, N_REVIEWERS = 300, 2000, 0.10, 0.90, 30, 2
P_EASY, P_HARD_DEEP, P_HARD_TYP = 0.96, 0.85, 0.30


def run(seed, control=False):
    rng = np.random.default_rng(seed)
    typ_b = rng.random(N_ITEMS) < 0.25
    hard = np.zeros(N_ITEMS, bool) if control else rng.random(N_ITEMS) < np.where(typ_b, 0.65, 0.05)
    p = simulate_panel(rng, n_items=N_ITEMS, hard=hard)
    n_r = p.labels.shape[1]
    final = majority_labels(p.labels)
    gold = rng.random(N_ITEMS) < GOLD_FRAC

    routed_types = []
    for name, rows in {"A": ~typ_b, "B": typ_b}.items():
        g = gold & rows
        k = int(np.sum(final[g] == p.truth[g]))
        if wilson_ci(k, int(g.sum()))[1] < TARGET:
            routed_types.append(name)

    out = dict(routed_any=float(bool(routed_types)), routed_B=float("B" in routed_types),
               routed_A=float("A" in routed_types), reviewers_are_deep=float("nan"), extra_labels_per_item=0.0)
    decided = final.copy()
    if "B" in routed_types:
        # qualification test: QUAL_N type-B gold items, mostly hard in the treatment, rated by every rater
        q_hard = rng.random(QUAL_N) < 0.65
        p_q = np.where(q_hard[:, None], np.where(p.deep, P_HARD_DEEP, P_HARD_TYP)[None, :], P_EASY)
        score = (rng.random((QUAL_N, n_r)) < p_q).sum(axis=0) + rng.random(n_r) * 1e-6
        reviewers = np.argsort(-score)[:N_REVIEWERS]
        out["reviewers_are_deep"] = float(p.deep[reviewers].all())
        rows = np.where(typ_b)[0]
        p_rev = np.where(p.hard[rows][:, None], np.where(p.deep[reviewers], P_HARD_DEEP, P_HARD_TYP)[None, :], P_EASY)
        correct = rng.random(p_rev.shape) < p_rev
        lab = np.where(correct, p.truth[rows][:, None], 1 - p.truth[rows][:, None])
        best = lab[:, 0]                                   # reviewers agree -> their label; split -> top reviewer
        decided[rows] = np.where(lab[:, 0] == lab[:, 1], lab[:, 0], best)
        out["extra_labels_per_item"] = N_REVIEWERS * typ_b.mean()
    acc = lambda d, r: float(np.nanmean(d[r] == p.truth[r])) if r.any() else float("nan")  # noqa: E731
    out.update(before_all=acc(final, np.ones(N_ITEMS, bool)), after_all=acc(decided, np.ones(N_ITEMS, bool)),
               before_hard=acc(final, p.hard), after_hard=acc(decided, p.hard),
               before_B=acc(final, typ_b), after_B=acc(decided, typ_b))
    return out


res = {}
for arm, control in {"treatment_hard_items_in_type_B": False, "control_no_hard_items": True}.items():
    runs = [run(s, control) for s in range(SEEDS)]
    res[arm] = {k: interval([r[k] for r in runs if not np.isnan(r[k])]) if any(not np.isnan(r[k]) for r in runs) else None
                for k in runs[0]}
write_json("exp3_route_hard_items", res)

t = res["treatment_hard_items_in_type_B"]
fig, ax = plt.subplots(figsize=(6.4, 3.9))
groups = ["all items", "type-B items", "hard items"]
b = [t["before_all"][0], t["before_B"][0], t["before_hard"][0]]
a = [t["after_all"][0], t["after_B"][0], t["after_hard"][0]]
x = np.arange(3)
ax.bar(x - 0.18, b, 0.36, color=GRAY, label="majority vote of the panel")
ax.bar(x + 0.18, a, 0.36, color=GREEN, label="weak type routed to qualified reviewers")
for i in range(3):
    ax.text(x[i] - 0.18, b[i] + 0.02, f"{b[i]:.0%}", ha="center", fontsize=9)
    ax.text(x[i] + 0.18, a[i] + 0.02, f"{a[i]:.0%}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x, groups)
ax.set_ylim(0, 1.12)
ax.set_ylabel("final labels correct")
ax.set_title(f"Route the weak item type: +{t['extra_labels_per_item'][0]:.1f} labels per item")
ax.legend(fontsize=8.5, loc="lower left")
save(fig, "fig3_route_hard_items")
for arm, d in res.items():
    print(arm, {k: (round(v[0], 3) if v else None) for k, v in d.items()})
