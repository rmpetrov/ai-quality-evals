"""Replay v2 simulated fixtures and CLI failure paths without a provider or credentials."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from ai_quality_evals.runner import report_json, run_evaluation, write_report


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/captured-format.jsonl"
    minimums = {
        "generation_success_rate": 1,
        "citation_in_context": 1,
        "context_quote_validity": 1,
        "current_citation_rate": 1,
        "effective_context_coverage": 1,
        "answerability_accuracy": 1,
    }
    first = run_evaluation(traces_path=path, minimums=minimums)
    second = run_evaluation(traces_path=path, minimums=minimums)
    if report_json(first) != report_json(second) or first.gate_status != "passed":
        raise SystemExit("Capture replay is not deterministic or its explicit gates failed")
    if first.response_origin != "synthetic_fixture" or any(
        record.execution != "simulated" for record in first.captures
    ):
        raise SystemExit("Test captures must be explicitly simulated")
    write_report(first, Path("reports/captured-format-simulated.json"))
    environment = {key: value for key, value in os.environ.items() if key != "OPENAI_API_KEY"}
    for arguments, diagnostic in [
        (["live", "--model", "simulated-model"], "--allow-live"),
        (["live", "--allow-live", "--model", "simulated-model"], "OPENAI_API_KEY"),
        (["live", "--allow-live", "--model", "simulated-model", "--max-cases", "0"], "max_cases"),
    ]:
        completed = subprocess.run(
            [sys.executable, "-m", "ai_quality_evals", *arguments],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 2 or diagnostic not in completed.stderr:
            raise SystemExit(f"Unexpected live CLI failure behavior: {completed.stderr}")
    payload = json.loads(report_json(first))
    if payload["human_review"] is not None:
        raise SystemExit("Human scores must remain unreviewed")
    print(
        "Simulated v2 capture replay and live CLI failure paths passed; provider was never invoked."
    )
    print(
        f"Optional SDK installed: {importlib.util.find_spec('openai') is not None}; "
        "offline evaluation works either way."
    )


if __name__ == "__main__":
    main()
