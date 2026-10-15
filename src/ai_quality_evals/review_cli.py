"""Offline review CLI; does not import or construct generation providers."""

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import cast

from ai_quality_evals.review import (
    compare_reviews,
    load_reviews,
    prepare_reviews,
    semantic_report,
    write_private_artifact,
)
from ai_quality_evals.runner import DEFAULT_DATA, run_evaluation
from ai_quality_evals.validation import ValidationError


def add_review_commands(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="review_command", required=True)
    for command in ("prepare", "validate", "summarize"):
        sub = commands.add_parser(command)
        sub.add_argument("--kb", type=Path, default=DEFAULT_DATA / "kb")
        sub.add_argument("--cases", type=Path, default=DEFAULT_DATA / "cases.jsonl")
        sub.add_argument("--traces", type=Path, required=True)
        if command != "prepare":
            sub.add_argument("--reviews", type=Path, required=True)
        if command != "validate":
            sub.add_argument("--output", type=Path, required=True)
        if command == "summarize":
            sub.add_argument("--top-k", type=int, default=3)
            sub.add_argument("--compare-traces", type=Path)
            sub.add_argument("--compare-reviews", type=Path)


def run_review_command(args: argparse.Namespace) -> int:
    command = cast(str, args.review_command)
    traces = cast(Path, args.traces)
    # Preparation/validation must work even for captures stored with K=1.
    top_k = cast(int, getattr(args, "top_k", 1))
    kb, cases = cast(Path, args.kb), cast(Path, args.cases)
    report = run_evaluation(kb=kb, cases_path=cases, traces_path=traces, top_k=top_k)
    if command == "prepare":
        output = cast(Path, args.output)
        write_private_artifact(output, asdict(prepare_reviews(report)), (traces,))
        print(f"PREPARED: {len(report.results)} unscored review rows; output={output}")
        return 0
    review_path = cast(Path, args.reviews)
    reviews, review_hash = load_reviews(review_path, report)
    if command == "validate":
        print(
            f"VALID REVIEWS: {len(reviews.reviews)} annotations bound to {reviews.capture_sha256}"
        )
        return 0
    runs = [semantic_report(report, reviews, review_hash)]
    comparison = None
    inputs: tuple[Path, ...] = (traces, review_path)
    compare_traces = cast(Path | None, args.compare_traces)
    compare_path = cast(Path | None, args.compare_reviews)
    if (compare_traces is None) != (compare_path is None):
        raise ValidationError("--compare-traces and --compare-reviews must be supplied together")
    if compare_traces is not None and compare_path is not None:
        other = run_evaluation(kb=kb, cases_path=cases, traces_path=compare_traces, top_k=top_k)
        other_reviews, other_hash = load_reviews(compare_path, other)
        if report.response_origin != other.response_origin:
            raise ValidationError(
                "comparison requires matching capture origins; "
                "do not mix synthetic and live results"
            )
        comparison = compare_reviews(
            reviews,
            other_reviews,
            tuple(sorted({row.case.case_id for row in report.results + other.results})),
        )
        runs.append(semantic_report(other, other_reviews, other_hash))
        inputs += (compare_traces, compare_path)
    payload = {
        "schema_version": 1,
        "report_type": "semantic_review",
        "review_gate_status": "not_evaluated",
        "runs": runs,
        "comparison": comparison,
        "limitations": [
            "Checks cover field consistency only; they cannot establish semantic truth.",
            "Semantic scores and defects are reviewer annotations, never automated judgments.",
            "Improved structural scores do not prove correctness; missing reviews do not pass.",
            "Compare paired applicable ratings; small samples and reviewer differences matter.",
        ],
    }
    output = cast(Path, args.output)
    write_private_artifact(output, payload, inputs)
    print(f"SEMANTIC REVIEW: {len(runs)} run(s); observation only; output={output}")
    return 0
