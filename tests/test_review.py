import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest

from ai_quality_evals.cli import main
from ai_quality_evals.live import run_live_batch
from ai_quality_evals.models import (
    GenerationRequest,
    GenerationSettings,
    ProviderError,
    ProviderResult,
)
from ai_quality_evals.prompt import canonical_json
from ai_quality_evals.review import (
    artifact_json,
    compare_reviews,
    load_reviews,
    parse_reviews,
    prepare_reviews,
    semantic_report,
    summarize_reviews,
    write_private_artifact,
)
from ai_quality_evals.runner import Report, report_json, run_evaluation
from ai_quality_evals.validation import ValidationError, parse_json

FIXTURE_PATH = Path(__file__).parent / "fixtures/semantic-examples.json"
EXAMPLES = {
    str(example["example_id"]): example
    for example in cast(list[dict[str, object]], json.loads(FIXTURE_PATH.read_text())["examples"])
}


class FixtureProvider:
    name = "fixture"
    is_simulated = True

    def __init__(self, responses: list[object]):
        self.responses = responses
        self.calls = 0

    def generate(self, request: GenerationRequest) -> ProviderResult:
        response = self.responses[self.calls]
        self.calls += 1
        if response is None:
            return ProviderResult(
                "error",
                None,
                None,
                None,
                None,
                None,
                0.0,
                ProviderError("connection", "Fabricated failure"),
            )
        return ProviderResult(
            "completed", canonical_json(response), None, None, "fixture-model", None, 0.0
        )

    def close(self) -> None:
        pass


def capture_examples(
    tmp_path: Path,
    names: tuple[str, ...] = ("fully-supported-answer", "correct-abstention"),
    *,
    version: str = "benefits_rag_v2",
    failed: bool = False,
) -> tuple[Path, Report]:
    path = tmp_path / "human-reviews" / "traces.jsonl"
    examples = [EXAMPLES[name] for name in names]
    responses = [example["response"] for example in examples] + ([None] if failed else [])
    ids = [str(example["case_id"]) for example in examples] + (["case-01"] if failed else [])
    run_live_batch(
        FixtureProvider(responses),
        settings=GenerationSettings("fixture-model"),
        captures_path=path,
        allow_live=True,
        case_ids=ids,
        max_cases=len(ids),
        prompt_version=version,
    )
    return path, run_evaluation(traces_path=path)


def annotations(report: Report, names: tuple[str, ...]) -> dict[str, object]:
    payload = asdict(prepare_reviews(report))
    payload["annotation_origin"] = "synthetic_fixture"
    examples = {str(EXAMPLES[name]["case_id"]): EXAMPLES[name] for name in names}
    for row in payload["reviews"]:
        if row["case_id"] in examples:
            example = examples[row["case_id"]]
            row.update(
                status="reviewed",
                reviewer="fixture-curator",
                reviewed_at="2000-01-01T00:00:00Z",
                scores=copy.deepcopy(example["scores"]),
                justifications=copy.deepcopy(example["justifications"]),
                defect_labels=copy.deepcopy(example["defect_labels"]),
            )
    return cast(dict[str, object], parse_json(artifact_json(payload)))


def first_row(payload: dict[str, object]) -> dict[str, object]:
    return cast(list[dict[str, object]], payload["reviews"])[0]


def test_prepare_never_fabricates_ratings_or_identity_and_handles_failure(tmp_path: Path) -> None:
    _, report = capture_examples(tmp_path, failed=True)
    template = prepare_reviews(report)
    assert template.capture_sha256 == report.input_sha256["traces"]
    assert template.annotation_origin == "human_review"
    assert {row.case_id: row.status for row in template.reviews} == {
        "case-01": "unreviewable",
        "case-18": "unreviewed",
        "case-22": "unreviewed",
    }
    assert all(row.reviewer is row.reviewed_at is None for row in template.reviews)
    assert all(set(row.scores.values()) == {None} for row in template.reviews)
    assert all(set(row.justifications.values()) == {None} for row in template.reviews)
    assert parse_reviews(parse_json(artifact_json(asdict(template))), report) == template
    with pytest.raises(ValidationError, match="requires"):
        prepare_reviews(run_evaluation())


@pytest.mark.parametrize("name", tuple(EXAMPLES))
def test_curated_structural_semantic_separation_and_provenance(tmp_path: Path, name: str) -> None:
    path, report = capture_examples(tmp_path, (name,))
    before = path.read_bytes()
    original_report = report_json(report)
    reviews = parse_reviews(annotations(report, (name,)), report)
    run = semantic_report(report, reviews, "0" * 64)
    human = cast(dict[str, object], run["human_semantic_review"])
    assert human["annotation_origin"] == "synthetic_fixture"
    assert human["semantic_status"] == "synthetic_annotations_recorded"
    assert human["status_counts"] == {"reviewed": 1, "unreviewed": 0, "unreviewable": 0}
    assert reviews.reviews[0].defect_labels == tuple(
        cast(list[str], EXAMPLES[name]["defect_labels"])
    )
    if name in {"source-contradiction", "unsupported-personal-claim", "insufficient-support"}:
        assert report.aggregate.metrics["citation_validity"] == 1
        assert report.aggregate.metrics["context_quote_validity"] == 1
        assert cast(dict[str, object], run["response_consistency"])["findings"] == []
        assert human["material_review_findings"]
    if name == "source-contradiction":
        assert report.aggregate.metrics["answerability_accuracy"] == 1
        assert reviews.reviews[0].scores["answer_correctness"] == 0
    if name == "unsupported-personal-claim":
        assert human["high_impact_unsupported_claim_case_ids"] == ["case-22"]
    if name == "fully-supported-answer":
        assert human["material_review_findings"] == []
        assert all(score == 2 for score in reviews.reviews[0].scores.values())
    assert cast(dict[str, object], run["automatic_evaluation"])["human_review"] is None
    assert report_json(report) == original_report
    assert path.read_bytes() == before


@pytest.mark.parametrize("rating", [-1, 3, True, 1.0, "2", [], {}])
def test_invalid_ratings_are_rejected(tmp_path: Path, rating: object) -> None:
    _, report = capture_examples(tmp_path, ("fully-supported-answer",))
    payload = annotations(report, ("fully-supported-answer",))
    cast(dict[str, object], first_row(payload)["scores"])["groundedness"] = rating
    with pytest.raises(ValidationError, match="invalid rating"):
        parse_reviews(payload, report)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reviewer", None, "require reviewer"),
        ("reviewer", " ", "nonblank"),
        ("reviewed_at", None, "require reviewer"),
        ("reviewed_at", "2000-01-01T00:00:00", "timezone"),
        ("reviewed_at", "2000-02-31T00:00:00Z", "timestamp"),
        ("reviewed_at", "2000-01-01T00:00:00+01:99", "timezone"),
        ("reviewed_at", "yesterday", "timestamp"),
        ("reviewed_at", True, "nonblank"),
        ("status", "passed", "status"),
        ("scores", {}, "missing"),
        ("justifications", {}, "missing"),
        ("defect_labels", ["unknown_defect"], "unknown defect"),
        ("defect_labels", ["missing_evidence", "missing_evidence"], "duplicate"),
    ],
)
def test_malformed_reviewer_annotations(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    _, report = capture_examples(tmp_path, ("fully-supported-answer",))
    payload = annotations(report, ("fully-supported-answer",))
    first_row(payload)[field] = value
    with pytest.raises(ValidationError, match=message):
        parse_reviews(payload, report)


@pytest.mark.parametrize("justification", [None, " "])
def test_inapplicable_ratings_still_need_written_justification(
    tmp_path: Path, justification: object
) -> None:
    _, report = capture_examples(tmp_path, ("correct-abstention",))
    payload = annotations(report, ("correct-abstention",))
    row = first_row(payload)
    assert cast(dict[str, object], row["scores"])["citation_support"] is None
    cast(dict[str, object], row["justifications"])["citation_support"] = justification
    with pytest.raises(ValidationError, match="justification|nonblank"):
        parse_reviews(payload, report)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("duplicate", "duplicate case"),
        ("unknown", "unknown reviewed case"),
        ("hash", "hash mismatch"),
        ("schema", "schema_version"),
        ("origin", "annotation_origin"),
        ("missing", "missing"),
    ],
)
def test_review_set_integrity(tmp_path: Path, change: str, message: str) -> None:
    _, report = capture_examples(tmp_path)
    payload = annotations(report, ("fully-supported-answer", "correct-abstention"))
    if change == "duplicate":
        cast(list[object], payload["reviews"]).append(copy.deepcopy(first_row(payload)))
    elif change == "unknown":
        first_row(payload)["case_id"] = "case-03"
    elif change == "hash":
        payload["capture_sha256"] = "0" * 64
    elif change == "schema":
        payload["schema_version"] = True
    elif change == "origin":
        payload["annotation_origin"] = "automatic_judge"
    else:
        first_row(payload).pop("notes")
    with pytest.raises(ValidationError, match=message):
        parse_reviews(payload, report)


@pytest.mark.parametrize("status", ["reviewed", "unreviewable"])
def test_failed_generations_cannot_be_rated(tmp_path: Path, status: str) -> None:
    _, report = capture_examples(tmp_path, failed=True)
    payload = annotations(report, ())
    row = first_row(payload)
    assert row["case_id"] == "case-01"
    row.update(status=status, reviewer="fixture-curator", reviewed_at="2000-01-01T00:00:00Z")
    cast(dict[str, object], row["scores"])["answer_correctness"] = 2
    with pytest.raises(ValidationError, match="cannot rate|cannot contain ratings"):
        parse_reviews(payload, report)


def test_manually_unreviewable_response_needs_identity_and_reason(tmp_path: Path) -> None:
    _, report = capture_examples(tmp_path, ("fully-supported-answer",))
    payload = annotations(report, ())
    row = first_row(payload)
    row.update(status="unreviewable", notes="Fixture: reviewer cannot interpret this response.")
    with pytest.raises(ValidationError, match="reviewer identity"):
        parse_reviews(payload, report)
    row.update(reviewer="fixture-curator", reviewed_at="2000-01-01T00:00:00Z")
    assert parse_reviews(payload, report).reviews[0].status == "unreviewable"
    cast(dict[str, object], row["scores"])["groundedness"] = 1
    with pytest.raises(ValidationError, match="cannot contain ratings"):
        parse_reviews(payload, report)


def test_summary_denominators_missing_and_unreviewable_cases(tmp_path: Path) -> None:
    _, report = capture_examples(tmp_path, failed=True)
    payload = annotations(report, ("correct-abstention",))
    payload["reviews"] = [
        row
        for row in cast(list[dict[str, object]], payload["reviews"])
        if row["case_id"] != "case-18"
    ]
    summary = summarize_reviews(report, parse_reviews(payload, report))
    assert summary["status_counts"] == {"reviewed": 1, "unreviewed": 1, "unreviewable": 1}
    assert summary["missing_review_case_ids"] == summary["unreviewed_case_ids"] == ["case-18"]
    assert summary["unreviewable_case_ids"] == ["case-01"]
    dimensions = cast(dict[str, dict[str, object]], summary["dimensions"])
    assert dimensions["citation_support"] == {
        "mean": None,
        "applicable_denominator": 0,
        "rating_counts": {"0": 0, "1": 0, "2": 0},
        "inapplicable_case_ids": ["case-22"],
    }
    assert dimensions["groundedness"]["mean"] == 2
    assert dimensions["groundedness"]["applicable_denominator"] == 1
    payload["reviews"] = []
    empty = summarize_reviews(report, parse_reviews(payload, report))
    assert empty["semantic_status"] == "unreviewed"
    assert empty["status_counts"] == {"reviewed": 0, "unreviewed": 2, "unreviewable": 1}
    assert all(
        value["mean"] is None
        for value in cast(dict[str, dict[str, object]], empty["dimensions"]).values()
    )


def test_paired_semantic_comparison_flags_authored_regressions(tmp_path: Path) -> None:
    _, left_report = capture_examples(tmp_path / "left", version="benefits_rag_v1")
    _, right_report = capture_examples(
        tmp_path / "right", ("source-contradiction", "unsupported-personal-claim")
    )
    left = parse_reviews(
        annotations(left_report, ("fully-supported-answer", "correct-abstention")), left_report
    )
    right = parse_reviews(
        annotations(right_report, ("source-contradiction", "unsupported-personal-claim")),
        right_report,
    )
    comparison = compare_reviews(left, right, ("case-18", "case-22", "case-02"))
    assert comparison["paired_reviewed_case_ids"] == ["case-18", "case-22"]
    assert comparison["not_paired_case_ids"] == ["case-02"]
    correctness = cast(dict[str, dict[str, object]], comparison["dimensions"])["answer_correctness"]
    assert correctness["left_mean"] == 2 and correctness["right_mean"] == 0
    assert correctness["mean_delta"] == -2 and correctness["paired_applicable_denominator"] == 2
    assert (
        cast(dict[str, dict[str, object]], comparison["dimensions"])["citation_support"][
            "paired_applicable_denominator"
        ]
        == 1
    )
    regressions = cast(list[dict[str, object]], comparison["material_regressions"])
    assert any(
        row["case_id"] == "case-22" and row["dimension"] == "groundedness" for row in regressions
    )
    assert comparison["new_defect_labels"] == [
        {"case_id": "case-18", "defect_labels": ["evidence_contradiction"]},
        {"case_id": "case-22", "defect_labels": ["unsupported_personal_claim"]},
    ]
    wrong_origin = copy.deepcopy(annotations(left_report, ("fully-supported-answer",)))
    wrong_origin["annotation_origin"] = "human_review"
    with pytest.raises(ValidationError, match="cannot compare"):
        compare_reviews(parse_reviews(wrong_origin, left_report), right)


def test_cli_prepare_validate_summarize_and_compare_is_offline(tmp_path: Path) -> None:
    left, report = capture_examples(tmp_path / "left", version="benefits_rag_v1")
    right, other = capture_examples(
        tmp_path / "right", ("source-contradiction", "unsupported-personal-claim")
    )
    destination = tmp_path / "human-reviews"
    template = destination / "template.json"
    assert main(["review", "prepare", "--traces", str(left), "--output", str(template)]) == 0
    assert json.loads(template.read_text())["reviews"][0]["reviewer"] is None
    left_reviews, right_reviews = destination / "left.json", destination / "right.json"
    left_reviews.write_text(
        artifact_json(annotations(report, ("fully-supported-answer", "correct-abstention")))
    )
    right_reviews.write_text(
        artifact_json(annotations(other, ("source-contradiction", "unsupported-personal-claim")))
    )
    assert main(["review", "validate", "--traces", str(left), "--reviews", str(left_reviews)]) == 0
    output = destination / "comparison.json"
    arguments = [
        "review",
        "summarize",
        "--traces",
        str(left),
        "--reviews",
        str(left_reviews),
        "--compare-traces",
        str(right),
        "--compare-reviews",
        str(right_reviews),
    ]
    assert main([*arguments, "--output", str(output)]) == 0
    first = output.read_bytes()
    payload = json.loads(first)
    assert payload["report_type"] == "semantic_review" and payload["schema_version"] == 1
    assert payload["review_gate_status"] == "not_evaluated"
    assert (
        payload["runs"][0]["review_sha256"] == hashlib.sha256(left_reviews.read_bytes()).hexdigest()
    )
    assert payload["runs"][0]["automatic_evaluation"] == json.loads(report_json(report))
    second = destination / "comparison-repeat.json"
    assert main([*arguments, "--output", str(second)]) == 0
    assert first == second.read_bytes()
    assert main([*arguments, "--output", str(output)]) == 2  # No historical overwrite.
    assert (
        main(
            [
                "review",
                "summarize",
                "--traces",
                str(left),
                "--reviews",
                str(left_reviews),
                "--compare-traces",
                str(right),
                "--output",
                str(destination / "invalid.json"),
            ]
        )
        == 2
    )


def test_private_artifacts_cannot_overwrite_inputs_or_escape_ignored_directories(
    tmp_path: Path,
) -> None:
    source = tmp_path / "human-reviews" / "input.json"
    source.parent.mkdir()
    source.write_text("Keep this input")
    with pytest.raises(ValidationError, match="overwrite"):
        write_private_artifact(source, {}, (source,))
    with pytest.raises(ValidationError, match="Git-ignored"):
        write_private_artifact(tmp_path / "public.json", {}, (source,))
    assert source.read_text() == "Keep this input"
    with pytest.raises(FileExistsError):
        write_private_artifact(source, {}, ())
    assert source.read_text() == "Keep this input"


def test_review_json_rejects_duplicate_keys_and_includes_path_diagnostics(tmp_path: Path) -> None:
    _, report = capture_examples(tmp_path)
    path = tmp_path / "bad.json"
    path.write_text('{"reviews": [], "reviews": []}')
    with pytest.raises(ValidationError, match="bad.json.*duplicate JSON key"):
        load_reviews(path, report)


def test_legacy_traces_and_prompt_snapshots_remain_replayable(tmp_path: Path) -> None:
    legacy = Path(__file__).parents[1] / "src/ai_quality_evals/data/replays/good.jsonl"
    report = run_evaluation(traces_path=legacy)
    original = report_json(report)
    reviews = prepare_reviews(report)
    assert parse_reviews(parse_json(artifact_json(asdict(reviews))), report) == reviews
    assert semantic_report(report, reviews, "0" * 64)["prompt_versions"] == []
    assert report_json(run_evaluation(traces_path=legacy)) == original
    for version in ("benefits_rag_v1", "benefits_rag_v2"):
        path, captured = capture_examples(tmp_path / version, version=version)
        before = path.read_bytes()
        prepare_reviews(captured)
        assert all(
            record.prompt.version == version for record in run_evaluation(traces_path=path).captures
        )
        assert path.read_bytes() == before


def test_semantic_reports_are_stable_across_hash_seeds(tmp_path: Path) -> None:
    traces, report = capture_examples(tmp_path)
    reviews = traces.parent / "reviews.json"
    reviews.write_text(
        artifact_json(annotations(report, ("fully-supported-answer", "correct-abstention")))
    )
    outputs = []
    for seed in ("1", "927"):
        output = traces.parent / f"summary-{seed}.json"
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai_quality_evals",
                "review",
                "summarize",
                "--traces",
                str(traces),
                "--reviews",
                str(reviews),
                "--output",
                str(output),
            ],
            env={
                **os.environ,
                "PYTHONHASHSEED": seed,
                "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        outputs.append(output.read_bytes())
    assert outputs[0] == outputs[1]
