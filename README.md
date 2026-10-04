# rater-audit

**Your experts agree with each other. That does not mean they are right.**

Expert ratings train and evaluate AI models. They are checked with agreement scores, rater screens and small
acceptance samples, and each of those can pass bad work. This repo audits every stage of an expert-data quality
pipeline, shows what goes wrong, measures a fix, and turns it into a QA summary for each delivered batch.
Every simulated result has a control arm where nothing should go wrong; two results use 3,355 real expert votes.

Video Overview

https://github.com/user-attachments/assets/092f2336-4422-493e-8f45-41cf73e24b4a

![Agreement stays high while the majority is wrong on hard items](figures/fig1_agreement_vs_accuracy.png)

![The agreement screen drops the expert who is right; screening on gold accuracy does not](figures/fig2_rater_screens.png)

| Stage | What looks fine | What is true | Control |
|---|---|---|---|
| Production | experts agree on 84% of ratings | on hard items the majority is right 36% of the time | no hard items: 99.5% right |
| Rater screen | dropping the least-agreeing raters raises alpha (0.68 to 0.71) | it drops a deep expert in every run; hard-item accuracy falls from 41% to 32% | equal skill: no change |
| **Fix: route by type** | find the weak item type on gold, qualify reviewers, route it | hard items 36% to **76%**, all items 87% to **94.5%**, for 0.5 extra labels per item | no hard items: nothing routed |
| Acceptance | "check 20 items, allow 1 defect" | passes a batch with 8% defects 52% of the time; a sized plan (98 items, 4 defects) cuts it to 10% | exact maths matches 20,000 simulated batches |
| Model-written ratings | a verified expert who lets a model write ratings agrees with the panel 96% of the time | 20 canary items catch one who copies half the time in 92% of runs | honest rater wrongly flagged 0.08% |
| Ranking (real votes) | a GPT-4 judge separates Claude-v1 from GPT-3.5 (71% vs 61% win rate) | expert votes cannot (65.0% vs 65.2%): five tiers, not six ranks; a 10-question sprint finds the top model 77% of the time | 2,000 question resamples |
| Judges and screens (real votes) | GPT-4 agrees with experts 86% (ties excluded), reproducing the published "over 80%" | GPT-4 changes 16% of verdicts when the two answers swap places; a two-sigma screen flags 2 experts | noise alone would flag 1.5; 59% of the spread between experts is sampling noise |

## QA summary for a delivered batch

```bash
uv sync
uv run python examples/make_example_batch.py          # a synthetic batch with a known answer
uv run python -m rater_audit.report examples/batch/ratings.csv --gold examples/batch/gold.csv \
    --canary examples/batch/canary.csv --review examples/batch/review.csv --out qa_summary.md
```

Inputs are plain CSV files: ratings (`item, rater, label, type`), gold items (`item, label`), canary items
(`item, model_label`) and an acceptance review (`item, defect`). The summary reports agreement next to accuracy
with intervals, the weak item types to route, raters judged on gold accuracy (and which ones an agreement screen
would wrongly drop), canary flags, the sized acceptance plan with a decision, and what to change for the next batch.
Example output: [examples/qa_summary.md](examples/qa_summary.md).

<details>
<summary>The pipeline, stage by stage, and all figures</summary>

| # | Stage | Check in this repo |
|---|---|---|
| 1 | Scope and quality bar | acceptance plan from the contract limit and both risks (exp 4) |
| 2 | Guidelines and gold set | gold items stratified by item type and difficulty (exp 2, 3) |
| 3 | Rater qualification | 30-item qualification test on the weak type (exp 3) |
| 4 | Pilot batch and calibration | final-label accuracy on gold, per item type, with intervals (exp 3, report) |
| 5 | Production monitoring | agreement vs accuracy (exp 1), rater screens (exp 2, 7), canary items (exp 5) |
| 6 | Review and adjudication | routing the weak type to qualified reviewers (exp 3) |
| 7 | Client acceptance | sized sampling plan, defect rate with an interval (exp 4) |
| 8 | Delivery | QA summary per batch (`rater_audit.report`) |
| 9 | Post-mortem | "for the next batch" section of the summary |
| — | Evaluation sprints on top of the data | ranking tiers, judge bias, sprint size on real votes (exp 6, 7) |

| | |
|---|---|
| ![](figures/fig2_rater_screens.png) | ![](figures/fig3_route_hard_items.png) |
| ![](figures/fig4_acceptance.png) | ![](figures/fig5_llm_written_ratings.png) |
| ![](figures/fig6_model_ranking.png) | ![](figures/fig7_judges_real_data.png) |

Simulation set-up: 2,000 items, 10 raters, 3 ratings per item, 20% hard items on which typical raters are right
30% of the time and two deep experts 85%; 300 runs per arm (500 for canaries). Gold items are a random 10%.
</details>

## Reproduce

```bash
uv sync && uv run pytest -q                                   # 10 tests
for e in experiments/exp*.py; do uv run python "$e"; done     # results/*.json and figures/*.png
```

**Limits.** Experiments 1 to 5 are simulations with assumed skill levels; they show the mechanism and its size under
those assumptions, not rates in any particular company. Experiments 6 and 7 use one public dataset of pairwise votes
on six 2023 models. Canary items assume a model answer that is known to be wrong.

Data: MT-Bench human judgments (Zheng et al., 2023, [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)), CC-BY-4.0.
Code: MIT. Author: Vahit FERYAD &lt;vahit.feryat@gmail.com&gt;
