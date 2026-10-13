"""CLI. Exit 0=observation/pass, 1=failed gate, 2=input/usage/provider failure."""

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from ai_quality_evals.capture import verify_input_hashes
from ai_quality_evals.metrics import METRIC_NAMES, aggregate, check_thresholds
from ai_quality_evals.models import CapturedRecord, GenerationSettings
from ai_quality_evals.prompt import validate_settings
from ai_quality_evals.runner import DEFAULT_DATA, run_evaluation, write_report
from ai_quality_evals.validation import load_cases, load_corpus, load_records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "retrieval", "replay", "live"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--kb", type=Path, default=DEFAULT_DATA / "kb")
        sub.add_argument("--cases", type=Path, default=DEFAULT_DATA / "cases.jsonl")
        if command in {"validate", "replay"}:
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
        if command == "live":
            sub.add_argument("--allow-live", action="store_true")
            sub.add_argument("--model", required=True)
            sub.add_argument("--max-cases", type=int, default=3)
            sub.add_argument("--case", action="append", default=[], dest="case_ids")
            sub.add_argument("--timeout", type=float, default=30.0)
            sub.add_argument("--max-output-tokens", type=int, default=800)
            sub.add_argument("--captures", type=Path)
            sub.set_defaults(output=None)
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
            for record in records:
                if isinstance(record, CapturedRecord):
                    verify_input_hashes(record, kb, cases_path)
            print(
                f"VALID: {len(documents)} documents, {len(cases)} cases, "
                f"{len(records)} replay traces"
            )
            return 0
        output = cast(Path | None, args.output)
        if command == "live":
            from ai_quality_evals.live import require_private_path, run_live_batch, select_cases
            from ai_quality_evals.provider import OpenAIProvider

            if not cast(bool, args.allow_live):
                raise ValueError("live execution requires explicit --allow-live opt-in")
            settings = GenerationSettings(
                cast(str, args.model), cast(float, args.timeout), cast(int, args.max_output_tokens)
            )
            validate_settings(settings)
            if cast(int, args.top_k) < 1:
                raise ValueError("top_k must be a positive integer")
            minimums = _minimums(cast(list[str], args.minimum))
            check_thresholds(aggregate([]), minimums)  # Validate before any paid requests.
            selected = select_cases(
                load_cases(cases_path, load_corpus(kb)),
                cast(list[str], args.case_ids),
                cast(int, args.max_cases),
            )
            run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
            traces = (
                cast(Path | None, args.captures) or Path("live-captures") / run_id / "traces.jsonl"
            )
            output = output or traces.parent / "report.json"
            require_private_path(traces)
            require_private_path(output)
            if traces.resolve() == output.resolve() or traces.exists() or output.exists():
                raise ValueError("capture/report paths must be distinct and must not already exist")
            print(
                f"Intended requests: {len(selected)} sequential; model={settings.model}; "
                f"max_output_tokens={settings.max_output_tokens}; retries=0. "
                "Request/token limits do not guarantee a strict USD spending cap.",
                flush=True,
            )
            provider = OpenAIProvider()
            try:
                records = run_live_batch(
                    provider,
                    settings=settings,
                    captures_path=traces,
                    allow_live=True,
                    kb=kb,
                    cases_path=cases_path,
                    case_ids=[case.case_id for case in selected],
                    max_cases=cast(int, args.max_cases),
                    top_k=cast(int, args.top_k),
                )
            finally:
                provider.close()
            known_usage = [
                record.generation.usage for record in records if record.generation.usage is not None
            ]
            print(
                f"Captured requests: {len(records)}; usage reported for {len(known_usage)}; "
                f"known input tokens={sum(usage.input_tokens for usage in known_usage)}; "
                f"known output tokens={sum(usage.output_tokens for usage in known_usage)}. "
                "Missing usage is unknown; USD cost is not calculated."
            )
        report = run_evaluation(
            kb=kb,
            cases_path=cases_path,
            traces_path=traces,
            top_k=cast(int, args.top_k),
            minimums=_minimums(cast(list[str], args.minimum)),
        )
        assert output is not None
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
        failures = [record for record in report.captures if record.generation.status != "completed"]
        for record in failures:
            error = record.generation.error
            print(
                f"ERROR: {record.case_id}: {record.generation.status}: "
                f"{error.message if error else 'generation failed'}",
                file=sys.stderr,
            )
        if failures:
            return 2
        return 1 if report.gate_status == "failed" else 0
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
