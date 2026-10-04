# Batch QA summary

2030 items, 6330 ratings, 11 raters, 200 gold items. Quality gate: **FAIL**.

## 1. Agreement and accuracy

| measure | value |
|---|---|
| Krippendorff's alpha (agreement) | 0.689 |
| final label correct on gold items | 89.5% (95% CI 84.5% to 93.0%, n=200) |

By item type:

| type | gold items | final label correct | 95% CI | below target? |
|---|---|---|---|---|
| A | 154 | 98.1% | 94.4% to 99.3% | no |
| B | 46 | 60.9% | 46.5% to 73.6% | YES: route to qualified reviewers |

## 2. Raters

Judged on gold accuracy. An agreement screen is shown for comparison only.

| rater | gold items | gold accuracy | 95% CI | agreement with panel | action |
|---|---|---|---|---|---|
| r04 | 71 | 81.7% | 71.2% to 89.0% | 90.6% | watch: interval too wide to decide |
| r06 | 52 | 82.7% | 70.3% to 90.6% | 90.5% | watch: interval too wide to decide |
| r10 | 51 | 84.3% | 72.0% to 91.8% | 93.2% | watch: interval too wide to decide |
| r09 | 54 | 85.2% | 73.4% to 92.3% | 91.6% | watch: interval too wide to decide |
| r08 | 57 | 86.0% | 74.7% to 92.7% | 91.8% | watch: interval too wide to decide |
| r07 | 58 | 87.9% | 77.1% to 94.0% | 92.7% | watch: interval too wide to decide |
| r11 | 50 | 90.0% | 78.6% to 95.7% | 88.4% | watch: interval too wide to decide |
| r03 | 52 | 90.4% | 79.4% to 95.8% | 92.1% | watch: interval too wide to decide |
| r05 | 53 | 90.6% | 79.7% to 95.9% | 90.8% | watch: interval too wide to decide |
| r01 | 51 | 92.2% | 81.5% to 96.9% | 89.1% | ok |
| r02 | 51 | 98.0% | 89.7% to 99.7% | 88.1% | KEEP: low agreement but accurate; an agreement screen would drop this rater |

## 3. Canary items (model-written ratings)

Panel median match with the model's known wrong answer: 16.7%. Flag: lower interval end above median + 10 points.

| rater | canaries | match with model | flagged |
|---|---|---|---|
| r11 | 30 | 63.3% | YES: review this rater |
| r09 | 30 | 23.3% | no |
| r04 | 30 | 20.0% | no |
| r06 | 30 | 20.0% | no |
| r05 | 30 | 20.0% | no |
| r01 | 30 | 16.7% | no |
| r08 | 30 | 13.3% | no |
| r02 | 30 | 10.0% | no |
| r03 | 30 | 6.7% | no |
| r07 | 30 | 6.7% | no |
| r10 | 30 | 6.7% | no |

## 4. Acceptance test

Contract limit 5.0%. Plan sized so a 2.0% batch fails at most 5% of the time and a 8.0% batch passes at most 10% of the time: **review 98 items, accept with at most 4 defects**.

Review: 14 defects in 98 items, defect rate 14.3% (95% CI 8.7% to 22.6%). Decision under the plan: **REJECT**.

## 5. For the next batch

- Route type **B** to reviewers qualified on that type's gold items.
- Keep scoring raters on gold accuracy; do not drop raters for low agreement alone.
- Keep gold items stratified by difficulty; an easy gold set passes everyone.
