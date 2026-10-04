"""Experiment 5: a verified expert who lets a language model write the answers.

Stage: production monitoring. Identity checks confirm who rated, not who wrote the rating. Canary items are
planted in every rater's queue: items where a language model is known to give a specific wrong answer and
people usually get it right. A rater is flagged when the lower end of their match-with-the-model interval is
more than 10 points above the panel's median match rate.
Treatment: one of 10 raters copies the model's answer on a share q of items. Control: no one copies; every flag
is a false accusation. The copier's agreement with the panel on ordinary items is also measured.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rater_audit.stats import wilson_ci, interval  # noqa: E402
from rater_audit.plotting import plt, save, write_json, BLUE, RED, GRAY, ORANGE  # noqa: E402

SEEDS, N_RATERS, HUMAN_MATCH, MARGIN = 500, 10, 0.10, 0.10
N_CANARIES, COPY_RATES = [10, 20, 40, 80], [0.3, 0.5, 0.7]
P_EASY_MODEL, P_EASY_HUMAN, N_ORDINARY = 0.97, 0.96, 300


def flags(n_canary, q, rng):
    """Returns flag per rater (rater 0 is the copier when q > 0) and the copier's ordinary-item agreement."""
    human = rng.random((n_canary, N_RATERS)) < HUMAN_MATCH          # honest raters match the model's wrong answer rarely
    copied = rng.random(n_canary) < q
    human[:, 0] = np.where(copied, True, human[:, 0])
    k = human.sum(axis=0)
    lo = np.array([wilson_ci(int(x), n_canary)[0] for x in k])
    base = np.median(k / n_canary)
    flagged = lo > base + MARGIN
    # ordinary easy items: copier = model when copying, else human; agreement with the honest majority
    truth_ok_model = rng.random(N_ORDINARY) < P_EASY_MODEL
    ok_copier = np.where(rng.random(N_ORDINARY) < q, truth_ok_model, rng.random(N_ORDINARY) < P_EASY_HUMAN)
    ok_others = rng.random((N_ORDINARY, 2)) < P_EASY_HUMAN
    maj_ok = ok_others.sum(axis=1) == 2
    agree = float(np.mean(ok_copier[maj_ok]))
    return flagged, agree


out = {"power": {}, "control_false_flag_per_rater": {}, "control_any_false_flag_per_panel": {}, "copier_agreement": {}}
for n in N_CANARIES:
    rng = np.random.default_rng(n)
    ctrl = [flags(n, 0.0, rng)[0] for _ in range(SEEDS)]
    out["control_false_flag_per_rater"][n] = float(np.mean([f.mean() for f in ctrl]))
    out["control_any_false_flag_per_panel"][n] = float(np.mean([f.any() for f in ctrl]))
    for q in COPY_RATES:
        runs = [flags(n, q, rng) for _ in range(SEEDS)]
        out["power"][f"{n}|{q}"] = float(np.mean([f[0] for f, _ in runs]))
        out["copier_agreement"][f"{n}|{q}"] = interval([a for _, a in runs])
write_json("exp5_llm_written_ratings", out)

fig, ax = plt.subplots(figsize=(6.6, 4.0))
for q, col in zip(COPY_RATES, [GRAY, ORANGE, RED]):
    ax.plot(N_CANARIES, [out["power"][f"{n}|{q}"] for n in N_CANARIES], "o-", color=col,
            label=f"rater copies the model on {q:.0%} of items")
ax.plot(N_CANARIES, [out["control_false_flag_per_rater"][n] for n in N_CANARIES], "s--", color=BLUE,
        label="control: honest rater wrongly flagged")
ax.set_xscale("log", base=2)
ax.set_xticks(N_CANARIES, [str(n) for n in N_CANARIES])
ax.set_xlabel("canary items in each rater's queue")
ax.set_ylabel("chance of being flagged")
ax.set_ylim(-0.02, 1.05)
ax.set_title("Canary items catch model-written ratings")
ax.legend(fontsize=8.5, loc="center right")
save(fig, "fig5_llm_written_ratings")
print(out["power"], out["control_false_flag_per_rater"], out["control_any_false_flag_per_panel"],
      {k: round(v[0], 3) for k, v in out["copier_agreement"].items()})
