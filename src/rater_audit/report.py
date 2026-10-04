"""QA summary for one delivered batch.

    python -m rater_audit.report ratings.csv --gold gold.csv [--canary canary.csv] [--review review.csv]
        [--max-defect 0.05 --good 0.02 --bad 0.08] [--out report.md]

ratings.csv  item, rater, label[, type]   one row per rating; 'type' is the item category (optional)
gold.csv     item, label                  items with a known, adjudicated answer, mixed into the queue
canary.csv   item, model_label            items where a language model gives a known wrong answer (optional)
review.csv   item, defect                 the client-style acceptance review: 1 if the delivered label is wrong (optional)
"""
import argparse
from pathlib import Path

import pandas as pd

from .panel import majority, alpha, rater_accuracy, agreement_with_panel, bottom_screen, canary_check
from .stats import wilson_ci, find_plan, accept_prob


def pct(x):
    return f"{100 * x:.1f}%"


def build(ratings, gold, canary=None, review=None, max_defect=0.05, good=0.02, bad=0.08, accuracy_floor=0.80):
    lines = []
    final = majority(ratings)
    g = gold.set_index("item")["label"]
    n_items, n_raters = ratings["item"].nunique(), ratings["rater"].nunique()
    lines += ["# Batch QA summary", "",
              f"{n_items} items, {len(ratings)} ratings, {n_raters} raters, {len(g)} gold items.", ""]

    # 1. Agreement vs accuracy
    on_gold = final.reindex(g.index).dropna()
    k = int((on_gold == g.loc[on_gold.index]).sum())
    lo, hi = wilson_ci(k, len(on_gold))
    lines += ["## 1. Agreement and accuracy", "",
              "| measure | value |", "|---|---|",
              f"| Krippendorff's alpha (agreement) | {alpha(ratings):.3f} |",
              f"| final label correct on gold items | {pct(k / len(on_gold))} (95% CI {pct(lo)} to {pct(hi)}, n={len(on_gold)}) |", ""]
    weak_types = []
    if "type" in ratings.columns:
        types = ratings.drop_duplicates("item").set_index("item")["type"]
        lines += ["By item type:", "", "| type | gold items | final label correct | 95% CI | below target? |", "|---|---|---|---|---|"]
        for t, items in types.groupby(types):
            gi = on_gold.index.intersection(items.index)
            if len(gi) == 0:
                continue
            kt = int((on_gold.loc[gi] == g.loc[gi]).sum())
            l2, h2 = wilson_ci(kt, len(gi))
            below = h2 < 1 - max_defect
            weak_types += [t] if below else []
            lines.append(f"| {t} | {len(gi)} | {pct(kt / len(gi))} | {pct(l2)} to {pct(h2)} | {'YES: route to qualified reviewers' if below else 'no'} |")
        lines.append("")

    # 2. Raters: gold accuracy, not agreement
    acc = rater_accuracy(ratings, g)
    agr = agreement_with_panel(ratings)
    r = acc.merge(agr[["rater", "agreement"]], on="rater", how="left")
    drop_by_agreement = set(bottom_screen(agr, max(1, n_raters // 10)))
    lines += ["## 2. Raters", "", "Judged on gold accuracy. An agreement screen is shown for comparison only.", "",
              "| rater | gold items | gold accuracy | 95% CI | agreement with panel | action |", "|---|---|---|---|---|---|"]
    for _, row in r.sort_values("accuracy").iterrows():
        if row.hi < accuracy_floor:
            action = "retrain or remove (clearly below floor)"
        elif row.rater in drop_by_agreement and row.lo >= accuracy_floor:
            action = "KEEP: low agreement but accurate; an agreement screen would drop this rater"
        elif row.lo < accuracy_floor:
            action = "watch: interval too wide to decide"
        else:
            action = "ok"
        lines.append(f"| {row.rater} | {int(row.n)} | {pct(row.accuracy)} | {pct(row.lo)} to {pct(row.hi)} | "
                     f"{pct(row.agreement) if pd.notna(row.agreement) else '-'} | {action} |")
    lines.append("")

    # 3. Model-written ratings
    if canary is not None:
        c = canary.set_index("item")["model_label"]
        cc = canary_check(ratings, c, human_rate=0.0)
        base = float(cc["match_rate"].median())
        cc["flagged"] = cc["lo"] > base + 0.10
        lines += ["## 3. Canary items (model-written ratings)", "",
                  f"Panel median match with the model's known wrong answer: {pct(base)}. Flag: lower interval end above median + 10 points.", "",
                  "| rater | canaries | match with model | flagged |", "|---|---|---|---|"]
        for _, row in cc.sort_values("match_rate", ascending=False).iterrows():
            lines.append(f"| {row.rater} | {int(row.n)} | {pct(row.match_rate)} | {'YES: review this rater' if row.flagged else 'no'} |")
        lines.append("")

    # 4. Acceptance
    n_plan, c_plan = find_plan(good, bad)
    lines += ["## 4. Acceptance test", "",
              f"Contract limit {pct(max_defect)}. Plan sized so a {pct(good)} batch fails at most 5% of the time and a "
              f"{pct(bad)} batch passes at most 10% of the time: **review {n_plan} items, accept with at most {c_plan} defects**.", ""]
    if review is not None:
        kd, nd = int(review["defect"].sum()), len(review)
        l3, h3 = wilson_ci(kd, nd)
        verdict = "ACCEPT" if (nd >= n_plan and kd <= c_plan) else ("REJECT" if kd > c_plan else "NOT ENOUGH ITEMS REVIEWED")
        lines += [f"Review: {kd} defects in {nd} items, defect rate {pct(kd / nd)} (95% CI {pct(l3)} to {pct(h3)}). "
                  f"Decision under the plan: **{verdict}**.", ""]
    else:
        lines += [f"For comparison, a quick check of 20 items allowing 1 defect passes a {pct(bad)} batch "
                  f"{pct(accept_prob(20, 1, bad))} of the time.", ""]

    # 5. Next batch
    lines += ["## 5. For the next batch", ""]
    lines += [f"- Route type **{t}** to reviewers qualified on that type's gold items." for t in weak_types]
    lines += ["- Keep scoring raters on gold accuracy; do not drop raters for low agreement alone.",
              "- Keep gold items stratified by difficulty; an easy gold set passes everyone.", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ratings")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--canary")
    ap.add_argument("--review")
    ap.add_argument("--max-defect", type=float, default=0.05)
    ap.add_argument("--good", type=float, default=0.02)
    ap.add_argument("--bad", type=float, default=0.08)
    ap.add_argument("--out")
    a = ap.parse_args()
    md = build(pd.read_csv(a.ratings), pd.read_csv(a.gold),
               pd.read_csv(a.canary) if a.canary else None, pd.read_csv(a.review) if a.review else None,
               a.max_defect, a.good, a.bad)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
