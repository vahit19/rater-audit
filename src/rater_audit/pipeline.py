"""Per-batch pipeline: load a batch export, validate it against the data contract, analyse, write outputs.

A batch directory holds ratings.csv and gold.csv, and optionally canary.csv and review.csv (see report.py for the
columns). Settings come from a TOML file. Outputs: results.json (for dashboards and alerts), qa_summary.md (for the
client) and an exit code for automation: 0 PASS, 2 ACTION (routing or rater follow-up needed), 1 FAIL (reject).
"""
import json
import tomllib
from pathlib import Path

import pandas as pd

from .report import analyze, render, DEFAULTS

EXIT = {"PASS": 0, "FAIL": 1, "ACTION": 2}
CONTRACT = {"ratings.csv": ["item", "rater", "label"], "gold.csv": ["item", "label"],
            "canary.csv": ["item", "model_label"], "review.csv": ["item", "defect"]}


class ContractError(ValueError):
    pass


def load_config(path=None) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        for section in raw.values():
            cfg.update(section if isinstance(section, dict) else {})
    return cfg


def load_batch(batch_dir) -> dict:
    d = Path(batch_dir)
    out = {}
    for name, cols in CONTRACT.items():
        f = d / name
        if not f.exists():
            if name in ("ratings.csv", "gold.csv"):
                raise ContractError(f"missing required file {name}")
            out[name.split(".")[0]] = None
            continue
        df = pd.read_csv(f)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ContractError(f"{name}: missing columns {missing}")
        out[name.split(".")[0]] = df
    validate(out)
    return out


def validate(b: dict) -> None:
    r = b["ratings"]
    if r[["item", "rater", "label"]].isna().any().any():
        raise ContractError("ratings.csv: empty item, rater or label")
    dup = r.duplicated(["item", "rater"]).sum()
    if dup:
        raise ContractError(f"ratings.csv: {dup} duplicate (item, rater) rows")
    if (r.groupby("item").size() < 2).mean() > 0.5:
        raise ContractError("ratings.csv: most items have fewer than 2 ratings; agreement cannot be measured")
    unknown = set(b["gold"]["item"]) - set(r["item"])
    if unknown:
        raise ContractError(f"gold.csv: {len(unknown)} gold items never rated")
    if b["review"] is not None and not set(b["review"]["defect"].unique()) <= {0, 1}:
        raise ContractError("review.csv: defect must be 0 or 1")


def run(batch_dir, config=None, out_dir=None) -> tuple[dict, int]:
    b = load_batch(batch_dir)
    res = analyze(b["ratings"], b["gold"], b["canary"], b["review"], **load_config(config))
    out = Path(out_dir or Path(batch_dir) / "out")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "qa_summary.md").write_text(render(res), encoding="utf-8")
    return res, EXIT[res["gate"]["status"]]
