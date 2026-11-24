"""Command-line entry point.

    python -m parent_aid.cli --data data/records.json --out reports/rotation

Writes ``rotation.json`` (machine-readable) and ``rotation.md`` (meeting summary).
Optionally diff against a previous report with ``--previous``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import Dataset
from .report import build_report, markdown_summary, refresh_decision, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parent-aid rotation evaluation")
    parser.add_argument("--data", default="data/records.json")
    parser.add_argument("--out", default="reports/rotation")
    parser.add_argument("--bootstrap", type=int, default=60)
    parser.add_argument(
        "--capacity-vintage",
        choices=("proposal", "updated"),
        default="proposal",
        help="'updated' applies the later capacity notices (unshown_scenarios)",
    )
    parser.add_argument("--proposal", action="append", default=None)
    parser.add_argument(
        "--previous",
        default=None,
        help="previous rotation.json; prints whether the last judgement still holds",
    )
    parser.add_argument("--seed", type=int, default=20261124)
    args = parser.parse_args(argv)

    dataset = Dataset.load(args.data)
    report = build_report(
        dataset,
        args.proposal,
        bootstrap_draws=args.bootstrap,
        capacity_vintage=args.capacity_vintage,
        seed=args.seed,
    )
    out = Path(args.out)
    write_report(report, out.with_suffix(".json"))
    out.with_suffix(".md").write_text(markdown_summary(report), encoding="utf-8")
    print(f"wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")

    if args.previous:
        previous = json.loads(Path(args.previous).read_text(encoding="utf-8"))
        print(json.dumps(refresh_decision(previous, report), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
