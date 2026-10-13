"""Exercise installed CLI, passing gates, expected failures, and byte reproducibility."""

import json
import subprocess
import sys
from pathlib import Path

from ai_quality_evals.runner import DEFAULT_DATA


def check(arguments: list[str], expected_exit: int, output: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "ai_quality_evals", *arguments, "--output", str(output)],
        check=False,
    )
    if completed.returncode != expected_exit:
        raise SystemExit(f"Expected exit {expected_exit}, got {completed.returncode}: {arguments}")
    payload = json.loads(output.read_text(encoding="utf-8"))
    expected_passed: bool | None
    if "--minimum" not in arguments:
        expected_status, expected_passed = "not_evaluated", None
    elif expected_exit == 0:
        expected_status, expected_passed = "passed", True
    else:
        expected_status, expected_passed = "failed", False
    if payload["gate_status"] != expected_status or payload["passed"] is not expected_passed:
        raise SystemExit(f"Unexpected report gate status: {output}")
    if expected_exit == 1 and not payload["threshold_failures"]:
        raise SystemExit(f"Expected actionable threshold diagnostics: {output}")


def main() -> None:
    retrieval = ["retrieval", "--minimum", "recall_at_k=0.95", "--minimum", "mrr=0.90"]
    check(retrieval, 0, Path("reports/retrieval.json"))
    check(retrieval, 0, Path("reports/retrieval-repeat.json"))
    if (
        Path("reports/retrieval.json").read_bytes()
        != Path("reports/retrieval-repeat.json").read_bytes()
    ):
        raise SystemExit("Repeated retrieval reports differ")
    common = [
        "--minimum",
        "recall_at_k=1",
        "--minimum",
        "mrr=1",
        "--minimum",
        "citation_validity=1",
        "--minimum",
        "exact_quote_validity=1",
        "--minimum",
        "answerability_accuracy=1",
        "--minimum",
        "citation_coverage=1",
        "--minimum",
        "evidence_coverage=1",
    ]
    check(
        ["replay", "--traces", str(DEFAULT_DATA / "replays/good.jsonl"), *common],
        0,
        Path("reports/replay-good.json"),
    )
    check(
        ["replay", "--traces", str(DEFAULT_DATA / "replays/bad.jsonl"), *common],
        1,
        Path("reports/replay-bad.json"),
    )
    check(
        ["replay", "--traces", str(DEFAULT_DATA / "replays/bad.jsonl")],
        0,
        Path("reports/replay-observation.json"),
    )
    check(
        ["retrieval", "--top-k", "1", "--minimum", "recall_at_k=0.95"],
        1,
        Path("reports/retrieval-k1.json"),
    )
    print("Offline smoke checks passed, including observation-only replay and two gate failures.")


if __name__ == "__main__":
    main()
