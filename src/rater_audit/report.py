"""Batch analysis and the QA summary.

analyze() turns one delivered batch into a structured result (the input for dashboards, alerts and the quality
gate); render() turns that result into the Markdown QA summary. build() does both.

ratings  item, rater, label[, type]   one row per rating; 'type' is the item category (optional)
gold     item, label                  items with a known, adjudicated answer, mixed into the queue
canary   item, model_label            items where a language model gives a known wrong answer (optional)
review   item, defect                 the acceptance review: 1 if the delivered label is wrong (optional)
"""
import pandas as pd

from .panel import majority, alpha, rater_accuracy, agreement_with_panel, bottom_screen, canary_check
from .stats import wilson_ci, find_plan, accept_prob

DEFAULTS = {"max_defect": 0.05, "good": 0.02, "bad": 0.08, "accuracy_floor": 0.80, "canary_margin": 0.10}


def pct(x):
    return f"{100 * x:.1f}%"


def analyze(ratings, gold, canary=None, review=None, **cfg):
    c = {**DEFAULTS, **cfg}
    final = majority(ratings)
    g = gold.set_index("item")["label"]
    n_raters = ratings["rater"].nunique()
    res = {"config": c, "batch": {"items": int(ratings["item"].nunique()), "ratings": int(len(ratings)),
                                  "raters": int(n_raters), "gold_items": int(len(g))}}

    on_gold = final.reindex(g.index).dropna()
    k = int((on_gold == g.loc[on_gold.index]).sum())
    lo, hi = wilson_ci(k, len(on_gold))
    res["agreement_alpha"] = float(alpha(ratings))
    res["accuracy_on_gold"] = {"k": k, "n": int(len(on_gold)), "rate": k / max(1, len(on_gold)), "lo": lo, "hi": hi}

    res["by_type"] = []
    if "type" in ratings.columns:
        types = ratings.drop_duplicates("item").set_index("item")["type"]
        for t, items in types.groupby(types):
            gi = on_gold.index.intersection(items.index)
            if len(gi) == 0:
                continue
            kt = int((on_gold.loc[gi] == g.loc[gi]).sum())
            l2, h2 = wilson_ci(kt, len(gi))
            res["by_type"].append({"type": str(t), "n": int(len(gi)), "rate": kt / len(gi), "lo": l2, "hi": h2,
                                   "route": bool(h2 < 1 - c["max_defect"])})

    acc = rater_accuracy(ratings, g)
    agr = agreement_with_panel(ratings)
    r = acc.merge(agr[["rater", "agreement"]], on="rater", how="left")
    drop_by_agreement = set(bottom_screen(agr, max(1, n_raters // 10)))
    res["raters"] = []
    for _, row in r.sort_values("accuracy").iterrows():
        if row.hi < c["accuracy_floor"]:
            action = "retrain_or_remove"
        elif row.rater in drop_by_agreement and row.lo >= c["accuracy_floor"]:
            action = "keep_despite_low_agreement"
        elif row.lo < c["accuracy_floor"]:
            action = "watch"
        else:
            action = "ok"
        res["raters"].append({"rater": str(row.rater), "n": int(row.n), "accuracy": float(row.accuracy),
                              "lo": float(row.lo), "hi": float(row.hi),
                              "agreement": None if pd.isna(row.agreement) else float(row.agreement), "action": action})

    res["canary"] = None
    if canary is not None:
        cc = canary_check(ratings, canary.set_index("item")["model_label"], human_rate=0.0)
        base = float(cc["match_rate"].median())
        cc["flagged"] = cc["lo"] > base + c["canary_margin"]
        res["canary"] = {"panel_median_match": base,
                         "raters": [{"rater": str(x.rater), "n": int(x.n), "match_rate": float(x.match_rate),
                                     "flagged": bool(x.flagged)} for x in cc.sort_values("match_rate", ascending=False).itertuples()]}

    n_plan, c_plan = find_plan(c["good"], c["bad"])
    res["acceptance"] = {"plan_n": n_plan, "plan_c": c_plan, "quick_check_pass_bad": accept_prob(20, 1, c["bad"]), "review": None,
                         "decision": "NO_REVIEW"}
    if review is not None:
        kd, nd = int(review["defect"].sum()), int(len(review))
        l3, h3 = wilson_ci(kd, nd)
        decision = "ACCEPT" if (nd >= n_plan and kd <= c_plan) else ("REJECT" if kd > c_plan else "NOT_ENOUGH_REVIEWED")
        res["acceptance"].update(review={"k": kd, "n": nd, "rate": kd / nd, "lo": l3, "hi": h3}, decision=decision)

    reasons = []
    if res["acceptance"]["decision"] == "REJECT":
        reasons.append("acceptance review over the defect limit")
    reasons += [f"route item type {t['type']} to qualified reviewers" for t in res["by_type"] if t["route"]]
    reasons += [f"review rater {x['rater']} (canary match)" for x in (res["canary"] or {"raters": []})["raters"] if x["flagged"]]
    reasons += [f"retrain or remove rater {x['rater']}" for x in res["raters"] if x["action"] == "retrain_or_remove"]
    res["gate"] = {"status": "FAIL" if res["acceptance"]["decision"] == "REJECT" else ("ACTION" if reasons else "PASS"),
                   "reasons": reasons}
    return res


ACTION_TEXT = {"retrain_or_remove": "retrain or remove (clearly below floor)",
               "keep_despite_low_agreement": "KEEP: low agreement but accurate; an agreement screen would drop this rater",
               "watch": "watch: interval too wide to decide", "ok": "ok"}


def render(res):
    c, b, a = res["config"], res["batch"], res["accuracy_on_gold"]
    L = ["# Batch QA summary", "",
         f"{b['items']} items, {b['ratings']} ratings, {b['raters']} raters, {b['gold_items']} gold items. "
         f"Quality gate: **{res['gate']['status']}**.", "",
         "## 1. Agreement and accuracy", "", "| measure | value |", "|---|---|",
         f"| Krippendorff's alpha (agreement) | {res['agreement_alpha']:.3f} |",
         f"| final label correct on gold items | {pct(a['rate'])} (95% CI {pct(a['lo'])} to {pct(a['hi'])}, n={a['n']}) |", ""]
    if res["by_type"]:
        L += ["By item type:", "", "| type | gold items | final label correct | 95% CI | below target? |", "|---|---|---|---|---|"]
        L += [f"| {t['type']} | {t['n']} | {pct(t['rate'])} | {pct(t['lo'])} to {pct(t['hi'])} | "
              f"{'YES: route to qualified reviewers' if t['route'] else 'no'} |" for t in res["by_type"]]
        L.append("")
    L += ["## 2. Raters", "", "Judged on gold accuracy. An agreement screen is shown for comparison only.", "",
          "| rater | gold items | gold accuracy | 95% CI | agreement with panel | action |", "|---|---|---|---|---|---|"]
    L += [f"| {x['rater']} | {x['n']} | {pct(x['accuracy'])} | {pct(x['lo'])} to {pct(x['hi'])} | "
          f"{pct(x['agreement']) if x['agreement'] is not None else '-'} | {ACTION_TEXT[x['action']]} |" for x in res["raters"]]
    L.append("")
    if res["canary"]:
        L += ["## 3. Canary items (model-written ratings)", "",
              f"Panel median match with the model's known wrong answer: {pct(res['canary']['panel_median_match'])}. "
              f"Flag: lower interval end above median + {int(100 * c['canary_margin'])} points.", "",
              "| rater | canaries | match with model | flagged |", "|---|---|---|---|"]
        L += [f"| {x['rater']} | {x['n']} | {pct(x['match_rate'])} | {'YES: review this rater' if x['flagged'] else 'no'} |"
              for x in res["canary"]["raters"]]
        L.append("")
    acc = res["acceptance"]
    L += ["## 4. Acceptance test", "",
          f"Contract limit {pct(c['max_defect'])}. Plan sized so a {pct(c['good'])} batch fails at most 5% of the time and a "
          f"{pct(c['bad'])} batch passes at most 10% of the time: **review {acc['plan_n']} items, accept with at most {acc['plan_c']} defects**.", ""]
    if acc["review"]:
        rv = acc["review"]
        verdict = {"ACCEPT": "ACCEPT", "REJECT": "REJECT", "NOT_ENOUGH_REVIEWED": "NOT ENOUGH ITEMS REVIEWED"}[acc["decision"]]
        L += [f"Review: {rv['k']} defects in {rv['n']} items, defect rate {pct(rv['rate'])} (95% CI {pct(rv['lo'])} to {pct(rv['hi'])}). "
              f"Decision under the plan: **{verdict}**.", ""]
    else:
        L += [f"For comparison, a quick check of 20 items allowing 1 defect passes a {pct(c['bad'])} batch "
              f"{pct(acc['quick_check_pass_bad'])} of the time.", ""]
    L += ["## 5. For the next batch", ""]
    L += [f"- Route type **{t['type']}** to reviewers qualified on that type's gold items." for t in res["by_type"] if t["route"]]
    L += ["- Keep scoring raters on gold accuracy; do not drop raters for low agreement alone.",
          "- Keep gold items stratified by difficulty; an easy gold set passes everyone.", ""]
    return "\n".join(L)


def build(ratings, gold, canary=None, review=None, max_defect=0.05, good=0.02, bad=0.08, accuracy_floor=0.80):
    return render(analyze(ratings, gold, canary, review, max_defect=max_defect, good=good, bad=bad,
                          accuracy_floor=accuracy_floor))
