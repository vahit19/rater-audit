# From tool to production system

rater-audit is built to run next to an existing rating platform, not to replace it. The platform keeps the task
queue, the experts and the routing; rater-audit reads each delivered batch, decides whether it passes, and sends
back what to change.

![System](../figures/fig11_system.png)

## Components

| Component | Code | Input | Output |
|---|---|---|---|
| Planner | `rater_audit.planning`, `python -m rater_audit plan` | contract limits, panel size, project size | gold items and canaries per rater, acceptance review size, share of work spent on controls |
| Data contract | `rater_audit.pipeline.load_batch` | batch export (CSV) | validated tables, or a contract error (exit 3) |
| Measurement | `rater_audit.report.analyze` | ratings, gold | agreement next to accuracy with intervals, accuracy per item type |
| Rater scoring | `analyze` | ratings, gold, canaries | per-rater gold accuracy, keep / watch / retrain, canary flags |
| Routing | `analyze` | accuracy per type on gold | item types to send to qualified reviewers |
| Acceptance | `analyze` | review sample | defect rate with an interval, decision under the sized plan |
| Gate and report | `python -m rater_audit run` | all of the above | `results.json`, `qa_summary.md`, exit code 0 PASS, 2 ACTION, 1 FAIL |

## Data contract (one batch = one directory)

| File | Columns | Required |
|---|---|---|
| `ratings.csv` | `item, rater, label[, type]` | yes |
| `gold.csv` | `item, label` (adjudicated answer) | yes |
| `canary.csv` | `item, model_label` (a model's known wrong answer) | no |
| `review.csv` | `item, defect` (0/1, from the acceptance review) | no |

The run rejects a batch with missing columns, empty labels, duplicate (item, rater) rows, gold items nobody rated,
or items with fewer than two ratings.

## Build plan

| Phase | Time | What is built | Done when |
|---|---|---|---|
| 0. Pilot | 2 weeks | Run on one past batch exported from the platform: accuracy against gold by item type, rater scoring, acceptance plan, QA summary | before-and-after numbers agreed with the delivery team |
| 1. Integration | 2-4 weeks | Batch export job from the platform; gold and canary items planned per project and mixed into queues; the gate runs on every batch | every batch ships with a QA summary and a gate result |
| 2. Production | 2-4 weeks | Routing decisions and rater actions written back to the platform; dashboard and alerts on `results.json`; per-client configs | routing and rater actions happen without manual steps |
| 3. Data value | optional | Fine-tune a small open model on clean and noisy versions of a batch to show clients what the quality is worth; a small model to pre-screen bad or model-written answers | a measured model gain per client |

## Operating notes

- Runs inside the client's environment; ratings and gold items never leave it.
- Every threshold (defect limit, accuracy floor, canary margin) is a business decision in the TOML config.
- Each run is reproducible from the batch files and the config; model calls, if any, are cached.
- Known limit: accuracy is only as good as the gold set. Gold items need adjudication and a difficulty mix, and
  they are reviewed after each batch.
