"""Command line.

    python -m rater_audit run BATCH_DIR [--config rater_audit.toml] [--out OUT_DIR]
        Validate and analyse one batch; write results.json and qa_summary.md; exit 0 PASS, 2 ACTION, 1 FAIL.
    python -m rater_audit plan --raters 40 --items 20000 [--half-width 0.05] [--copy-rate 0.5]
        Size the controls for a project: gold items and canaries per rater, acceptance review, overhead.
"""
import argparse
import json
import sys

from .pipeline import run, ContractError
from .planning import plan


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="rater_audit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("batch_dir")
    r.add_argument("--config")
    r.add_argument("--out")
    p = sub.add_parser("plan")
    p.add_argument("--raters", type=int, required=True)
    p.add_argument("--items", type=int, required=True)
    p.add_argument("--ratings-per-item", type=int, default=3)
    p.add_argument("--half-width", type=float, default=0.05)
    p.add_argument("--copy-rate", type=float, default=0.5)
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        print(json.dumps(plan(a.raters, a.items, a.ratings_per_item, a.half_width, copy_rate=a.copy_rate), indent=2))
        return 0
    try:
        res, code = run(a.batch_dir, a.config, a.out)
    except ContractError as e:
        print(f"data contract error: {e}", file=sys.stderr)
        return 3
    print(f"gate: {res['gate']['status']}")
    for reason in res["gate"]["reasons"]:
        print(f"  - {reason}")
    return code


if __name__ == "__main__":
    sys.exit(main())
