import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rater_audit import stats, simulate, panel, mtbench  # noqa: E402
from rater_audit.report import build  # noqa: E402


def test_wilson_matches_worked_example():
    lo, hi = stats.wilson_ci(2, 100)
    assert lo == pytest.approx(0.0055, abs=5e-4) and hi == pytest.approx(0.0700, abs=5e-4)


def test_zero_defect_rule():
    assert stats.zero_defect_n(0.05) == 59
    assert (1 - 0.05) ** 59 <= 0.05 < (1 - 0.05) ** 58


def test_sized_plan_meets_both_risks():
    n, c = stats.find_plan(0.02, 0.08)
    assert (n, c) == (98, 4)
    assert stats.accept_prob(n, c, 0.02) >= 0.95 and stats.accept_prob(n, c, 0.08) <= 0.10


def test_quick_check_fake_pass():
    assert stats.accept_prob(20, 1, 0.10) == pytest.approx(0.3917, abs=1e-4)


def test_alpha_perfect_and_chance():
    perfect = np.array([[1, 1, 1], [0, 0, 0], [1, 1, np.nan]] * 20, dtype=float)
    assert stats.krippendorff_alpha_nominal(perfect) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 2, (3000, 3)).astype(float)
    assert abs(stats.krippendorff_alpha_nominal(noise)) < 0.05


def test_control_panel_majority_is_right():
    p = simulate.simulate_panel(np.random.default_rng(0), n_items=1000, hard_frac=0.0)
    m = simulate.majority_labels(p.labels)
    assert np.nanmean(m == p.truth) > 0.98


def test_array_and_table_agreement_match():
    p = simulate.simulate_panel(np.random.default_rng(1), n_items=300)
    arr = simulate.agreement_rates(p.labels)
    i, r = np.where(~np.isnan(p.labels))
    df = pd.DataFrame({"item": i, "rater": r, "label": p.labels[i, r]})
    tab = panel.agreement_with_panel(df).set_index("rater")["agreement"]
    assert np.allclose(arr, tab.loc[range(p.labels.shape[1])].to_numpy())


def test_gold_accuracy_screen_spares_deep_experts():
    hits = 0
    for s in range(20):
        p = simulate.simulate_panel(np.random.default_rng(s), n_items=2000)
        acc = [simulate.label_accuracy(p.labels, p.truth, np.ones(2000, bool), [j]) for j in range(10)]
        hits += p.deep[np.argsort(acc)[:2]].any()
    assert hits == 0


def test_mtbench_loads_and_counts():
    h, g = mtbench.load("human"), mtbench.load("gpt4_pair")
    assert len(h) == 3355 and len(g) == 2400
    assert set(h.vote) == {"a", "b", "tie"}
    assert int((g.raw_winner == "tie (inconsistent)").sum()) == 380


def test_report_on_example_batch():
    b = ROOT / "examples" / "batch"
    if not (b / "ratings.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "examples" / "make_example_batch.py")], check=True)
    md = build(pd.read_csv(b / "ratings.csv"), pd.read_csv(b / "gold.csv"),
               pd.read_csv(b / "canary.csv"), pd.read_csv(b / "review.csv"))
    assert "| r11 | 30 |" in md and "YES: review this rater" in md
    assert "Decision under the plan: **REJECT**" in md
    assert "Route type **B**" in md


def test_pipeline_gate_and_outputs(tmp_path):
    from rater_audit.pipeline import run
    b = ROOT / "examples" / "batch"
    if not (b / "ratings.csv").exists():
        subprocess.run([sys.executable, str(ROOT / "examples" / "make_example_batch.py")], check=True)
    res, code = run(b, ROOT / "examples" / "rater_audit.toml", tmp_path)
    assert res["gate"]["status"] == "FAIL" and code == 1
    assert (tmp_path / "results.json").exists() and (tmp_path / "qa_summary.md").exists()


def test_contract_rejects_duplicates(tmp_path):
    from rater_audit.pipeline import load_batch, ContractError
    pd.DataFrame({"item": ["a", "a", "b", "b"], "rater": ["x", "x", "x", "y"], "label": [1, 1, 0, 0]}).to_csv(tmp_path / "ratings.csv", index=False)
    pd.DataFrame({"item": ["a"], "label": [1]}).to_csv(tmp_path / "gold.csv", index=False)
    with pytest.raises(ContractError):
        load_batch(tmp_path)


def test_planning_sizes():
    from rater_audit.planning import gold_per_rater, canaries_per_rater
    assert gold_per_rater(0.05, 0.90) == 139
    c = canaries_per_rater(0.5)
    assert c["power"] >= 0.90 and c["honest_false_flag"] < 0.01


def test_dices_and_cached_model_replay():
    from rater_audit.dices import load
    from rater_audit.llm import unsafe_verdict
    m, items, raters = load()
    assert m.shape == (350, 123) and set(items["gold"].unique()) == {0, 1}
    first = items.iloc[0]
    assert unsafe_verdict("openai/gpt-4o-mini", first["context"], first["response"]) in (0, 1)
