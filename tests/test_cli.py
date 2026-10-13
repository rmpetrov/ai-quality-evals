import json
from pathlib import Path

import pytest

from ai_quality_evals.cli import main


def test_validate_cli(data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", "--traces", str(data_dir / "replays/bad.jsonl")]) == 0
    assert "14 documents, 24 cases, 6 replay traces" in capsys.readouterr().out


def test_cli_threshold_failure_writes_report_and_exits_one(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "failed.json"
    assert (
        main(
            [
                "replay",
                "--traces",
                str(data_dir / "replays/bad.jsonl"),
                "--minimum",
                "exact_quote_validity=1",
                "--output",
                str(output),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "FAIL: exact_quote_validity=0.4" in captured.err
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["gate_status"] == "failed"
    assert payload["passed"] is False
    assert payload["threshold_failures"][0]["actual"] == 0.4


def test_cli_no_thresholds_is_observation_only(
    tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "observed.json"
    assert (
        main(
            [
                "replay",
                "--traces",
                str(data_dir / "replays/bad.jsonl"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert captured.out.splitlines()[-1] == "OBSERVATION ONLY (no thresholds configured)"
    assert captured.err == ""
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["gate_status"] == "not_evaluated"
    assert payload["passed"] is None
    assert payload["minimums"] == {}
    assert payload["threshold_failures"] == []
    assert payload["aggregate"]["metrics"]["exact_quote_validity"] == 0.4


def test_cli_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "passed.json"
    assert (
        main(
            [
                "retrieval",
                "--minimum",
                "recall_at_k=0.95",
                "--minimum",
                "mrr=0.90",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.splitlines()[-1] == "PASS"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["gate_status"] == "passed"
    assert payload["passed"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["--minimum", "mrr=NaN"],
        ["--minimum", "mrr=1.1"],
        ["--minimum", "mrr=not-a-number"],
        ["--minimum", "unknown=0.5"],
        ["--minimum", "mrr"],
        ["--minimum", "mrr=0.5", "--minimum", "mrr=0.7"],
        ["--top-k", "0"],
    ],
)
def test_cli_bad_configuration_is_exit_two(
    arguments: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "must-not-exist.json"
    assert main(["retrieval", "--output", str(output), *arguments]) == 2
    assert "ERROR:" in capsys.readouterr().err
    assert not output.exists()


def test_cli_missing_input_is_exit_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["validate", "--cases", str(tmp_path / "missing.jsonl")]) == 2
    assert "ERROR:" in capsys.readouterr().err


def test_unavailable_response_metric_cannot_silently_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        main(
            [
                "retrieval",
                "--minimum",
                "citation_validity=0",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 1
    )
    assert "no_applicable_observations" in capsys.readouterr().err


def test_cli_requires_replay_trace_path() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["replay"])
    assert exc.value.code == 2
