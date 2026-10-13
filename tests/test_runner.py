import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from ai_quality_evals.runner import report_json, run_evaluation, write_report


def test_real_retrieval_baseline_and_archived_distractors() -> None:
    report = run_evaluation(minimums={"recall_at_k": 0.95, "mrr": 0.90})
    assert report.mode == "retrieval"
    assert report.response_origin is None
    assert report.aggregate.case_count == 24
    assert report.aggregate.metrics["recall_at_k"] == 1.0
    assert report.aggregate.metrics["mrr"] == 0.95
    assert report.aggregate.denominators["recall_at_k"] == 20
    assert report.gate_status == "passed"
    assert report.passed is True
    assert report.omitted_case_ids == ()
    by_id = {row.case.case_id: row for row in report.results}
    assert by_id["case-16"].reciprocal_rank == 0.5
    assert by_id["case-17"].reciprocal_rank == 0.5
    assert report.by_category["distractor"].metrics["mrr"] == 0.8
    assert report.by_category["unanswerable"].metrics["recall_at_k"] is None


def test_correct_replays_pass_all_structural_gates(data_dir: Path) -> None:
    report = run_evaluation(
        traces_path=data_dir / "replays/good.jsonl",
        minimums={
            "recall_at_k": 1,
            "mrr": 1,
            "citation_validity": 1,
            "exact_quote_validity": 1,
            "answerability_accuracy": 1,
            "citation_coverage": 1,
            "evidence_coverage": 1,
        },
    )
    assert report.gate_status == "passed"
    assert report.passed is True
    assert report.mode == "replay"
    assert report.response_origin == "synthetic_fixture"
    assert report.retrieval_config is None
    assert report.aggregate.case_count == 6
    assert len(report.omitted_case_ids) == 18
    assert all(value == 1 for value in report.aggregate.metrics.values())
    assert report.aggregate.denominators["citation_validity"] == 6
    assert report.aggregate.denominators["exact_quote_validity"] == 7


def test_defective_replays_fail_with_traceable_evidence(data_dir: Path) -> None:
    report = run_evaluation(
        traces_path=data_dir / "replays/bad.jsonl",
        minimums={
            "recall_at_k": 0.95,
            "citation_validity": 1,
            "exact_quote_validity": 1,
            "answerability_accuracy": 1,
        },
    )
    assert report.gate_status == "failed"
    assert report.passed is False
    assert report.aggregate.metrics == {
        "recall_at_k": 0.375,
        "mrr": 0.5,
        "citation_validity": 0.8,
        "exact_quote_validity": 0.4,
        "answerability_accuracy": 2 / 3,
        "citation_coverage": 2 / 3,
        "evidence_coverage": 2 / 3,
    }
    assert {failure.metric for failure in report.threshold_failures} == {
        "recall_at_k",
        "citation_validity",
        "exact_quote_validity",
        "answerability_accuracy",
    }
    by_id = {row.case.case_id: row for row in report.results}
    assert by_id["case-01"].citations[0].document_id == "missing-policy"
    assert not by_id["case-01"].citations[0].valid
    assert by_id["case-11"].quotations[0].reason == "not_an_exact_substring"
    # An archived, exact citation passes structural checks, while current-source recall fails.
    assert by_id["case-16"].citations[0].valid
    assert by_id["case-16"].citations[0].source_status == "archived"
    assert by_id["case-16"].quotations[0].valid
    assert by_id["case-16"].recall_at_k == 0.0
    assert by_id["case-22"].answerability_correct is False


def test_degraded_real_retrieval_fails_at_one() -> None:
    report = run_evaluation(top_k=1, minimums={"recall_at_k": 0.95})
    # Ten one-source cases hit; eight two-source cases get half recall; two distractors miss.
    assert report.aggregate.metrics["recall_at_k"] == 0.7
    assert report.aggregate.metrics["mrr"] == 0.9
    assert report.gate_status == "failed"
    assert report.passed is False
    assert report.threshold_failures[0].metric == "recall_at_k"


def test_structured_report_is_deterministic_and_traceable(tmp_path: Path) -> None:
    report = run_evaluation()
    path = tmp_path / "nested/report.json"
    write_report(report, path)
    assert path.read_text(encoding="utf-8") == report_json(run_evaluation())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["gate_status"] == "not_evaluated"
    assert payload["passed"] is None
    assert len(payload["input_sha256"]["corpus"]) == 64
    assert payload["input_sha256"]["traces"] is None
    first = payload["results"][0]
    assert first["case"]["case_id"] == "case-01"
    assert first["case"]["reference_evidence"][0]["document_id"] == "enrollment-2026"
    assert first["retrieval"]["query"] == first["case"]["question"]
    assert first["response"] is None
    assert "timestamp" not in payload
    assert str(Path.cwd()) not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("minimums", "expected_status", "expected_passed"),
    [
        (None, "not_evaluated", None),
        ({}, "not_evaluated", None),
        ({"exact_quote_validity": 0.4}, "passed", True),
        ({"exact_quote_validity": 1.0}, "failed", False),
    ],
)
def test_defective_replay_gate_states_preserve_evaluation_results(
    data_dir: Path,
    minimums: dict[str, float] | None,
    expected_status: str,
    expected_passed: bool | None,
) -> None:
    report = run_evaluation(traces_path=data_dir / "replays/bad.jsonl", minimums=minimums)
    payload = json.loads(report_json(report))
    assert report.gate_status == expected_status
    assert report.passed is expected_passed
    assert payload["gate_status"] == expected_status
    assert payload["passed"] is expected_passed
    baseline = run_evaluation(traces_path=data_dir / "replays/bad.jsonl")
    assert report.aggregate == baseline.aggregate
    assert report.by_category == baseline.by_category
    assert report.results == baseline.results


def test_reports_are_byte_identical_across_hash_seeds(tmp_path: Path) -> None:
    outputs: list[bytes] = []
    for seed in ("1", "927"):
        path = tmp_path / f"{seed}.json"
        completed = subprocess.run(
            [sys.executable, "-m", "ai_quality_evals", "retrieval", "--output", str(path)],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1]


def test_retrieval_is_independent_of_gold_facts_and_intents(tmp_path: Path, data_dir: Path) -> None:
    cases = [
        cast(dict[str, object], json.loads(line))
        for line in (data_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    for case in cases:
        case["reference_facts"] = ["Reviewer annotation changed."]
        case["intent"] = "This field must never influence retrieval."
    path = tmp_path / "changed-labels.jsonl"
    path.write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")
    original = run_evaluation()
    modified = run_evaluation(cases_path=path)
    assert [row.retrieval for row in original.results] == [
        row.retrieval for row in modified.results
    ]
    assert original.input_sha256["cases"] != modified.input_sha256["cases"]


def test_replay_cutoff_cannot_exceed_stored_retrieval(data_dir: Path) -> None:
    with pytest.raises(ValueError, match="stored top_k"):
        run_evaluation(traces_path=data_dir / "replays/good.jsonl", top_k=4)


def test_atomic_report_write_preserves_previous_artifact_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "report.json"
    path.write_text("previous artifact", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr("ai_quality_evals.runner.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        write_report(run_evaluation(), path)
    assert path.read_text(encoding="utf-8") == "previous artifact"
    assert list(tmp_path.iterdir()) == [path]


def test_evaluation_does_not_open_network_connections(
    data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("Offline evaluation attempted a network connection")

    monkeypatch.setattr("socket.socket", forbidden_socket)
    assert run_evaluation().aggregate.case_count == 24
    assert run_evaluation(traces_path=data_dir / "replays/good.jsonl").aggregate.case_count == 6
