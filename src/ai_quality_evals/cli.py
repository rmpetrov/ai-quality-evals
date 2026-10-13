"""Offline CLI. Exit 0=valid observation/pass, 1=failed gate, 2=invalid input/usage."""

import argparse
import sys
from pathlib import Path
from typing import cast

from ai_quality_evals.metrics import METRIC_NAMES
from ai_quality_evals.runner import DEFAULT_DATA, run_evaluation, write_report
from ai_quality_evals.validation import load_cases, load_corpus, load_records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "retrieval", "replay"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--kb", type=Path, default=DEFAULT_DATA / "kb")
        sub.add_argument("--cases", type=Path, default=DEFAULT_DATA / "cases.jsonl")
        if command != "retrieval":
            sub.add_argument("--traces", type=Path, required=command == "replay")
        if command != "validate":
            sub.add_argument("--top-k", type=int, default=3)
            sub.add_argument("--output", type=Path, default=Path("reports") / f"{command}.json")
            sub.add_argument(
                "--minimum",
                action="append",
                default=[],
                metavar="METRIC=VALUE",
                help="inclusive lower bound, repeatable; available: "
                + ", ".join(sorted(METRIC_NAMES)),
            )
    return parser


def _minimums(items: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or name not in METRIC_NAMES or name in result:
            raise ValueError(f"invalid or duplicate --minimum: {item}")
        result[name] = float(value)
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # argparse is the only untyped boundary; typed locals keep the execution path strict.
    command = cast(str, args.command)
    kb = cast(Path, args.kb)
    cases_path = cast(Path, args.cases)
    traces = cast(Path | None, getattr(args, "traces", None))
    try:
        if command == "validate":
            documents = load_corpus(kb)
            cases = load_cases(cases_path, documents)
            records = load_records(traces, cases, documents) if traces else ()
            print(
                f"VALID: {len(documents)} documents, {len(cases)} cases, "
                f"{len(records)} replay traces"
            )
            return 0
        report = run_evaluation(
            kb=kb,
            cases_path=cases_path,
            traces_path=traces,
            top_k=cast(int, args.top_k),
            minimums=_minimums(cast(list[str], args.minimum)),
        )
        output = cast(Path, args.output)
        write_report(report, output)
        print(
            f"{report.mode}: {report.aggregate.case_count}/{report.dataset_case_count} cases; "
            f"K={report.evaluation_k}; report={output}"
        )
        for name, value in report.aggregate.metrics.items():
            formatted = "N/A" if value is None else f"{value:.6f}"
            print(f"  {name}={formatted} (n={report.aggregate.denominators[name]})")
        for failure in report.threshold_failures:
            comparison = (
                "has no applicable observations"
                if failure.actual is None
                else f"{failure.actual} < minimum {failure.minimum}"
            )
            print(
                f"FAIL: {failure.metric}={comparison} ({failure.reason})",
                file=sys.stderr,
            )
        print(
            {
                "not_evaluated": "OBSERVATION ONLY (no thresholds configured)",
                "passed": "PASS",
                "failed": "FAIL",
            }[report.gate_status]
        )
        return 1 if report.gate_status == "failed" else 0
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
