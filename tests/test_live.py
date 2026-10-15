import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

import pytest

from ai_quality_evals.capture import capture_json, parse_capture
from ai_quality_evals.cli import main
from ai_quality_evals.live import require_private_path, run_live_batch, select_cases
from ai_quality_evals.models import (
    CapturedRecord,
    GenerationRequest,
    GenerationSettings,
    GenerationStatus,
    ProviderError,
    ProviderResult,
    TokenUsage,
)
from ai_quality_evals.prompt import PROMPT_INSTRUCTIONS, canonical_json, prompt_hash
from ai_quality_evals.runner import DEFAULT_DATA, report_json, run_evaluation
from ai_quality_evals.validation import ValidationError, load_cases, load_corpus, parse_json

ELIGIBILITY_QUOTE = (
    "Regular employees scheduled for at least 20 hours per week are eligible for benefits."
)


def answer(
    text: str = "Regular employees need at least 20 scheduled hours per week.",
    document_id: str = "eligibility",
    quote: str = ELIGIBILITY_QUOTE,
) -> dict[str, object]:
    return {
        "outcome": "answer",
        "text": text,
        "citations": [document_id],
        "evidence": [{"document_id": document_id, "quote": quote}],
    }


DEFAULT_USAGE = TokenUsage(123, 45, 168)


def generated(value: object, *, usage: TokenUsage | None = DEFAULT_USAGE) -> ProviderResult:
    return ProviderResult(
        "completed",
        canonical_json(value),
        "simulated-response",
        "simulated-request",
        "returned-model-snapshot",
        usage,
        12.5,
    )


class ScriptedProvider:
    name = "fake"
    is_simulated = True

    def __init__(self, results: list[ProviderResult]):
        self.results = results
        self.requests: list[GenerationRequest] = []

    def generate(self, request: GenerationRequest) -> ProviderResult:
        self.requests.append(request)
        return self.results[len(self.requests) - 1]

    def close(self) -> None:
        pass


@pytest.fixture
def captured(tmp_path: Path) -> CapturedRecord:
    return run_live_batch(
        ScriptedProvider([generated(answer())]),
        settings=GenerationSettings("requested-model"),
        captures_path=tmp_path / "capture.jsonl",
        allow_live=True,
        case_ids=["case-02"],
        top_k=1,
    )[0]


def test_real_bm25_context_and_complete_provenance(tmp_path: Path) -> None:
    provider = ScriptedProvider([generated(answer())])
    path = tmp_path / "captured.jsonl"
    record = run_live_batch(
        provider,
        settings=GenerationSettings("requested-model", 9.0, 600),
        captures_path=path,
        allow_live=True,
        case_ids=["case-02"],
        top_k=1,
    )[0]
    assert record.trace_origin == "synthetic_fixture"
    assert record.execution == "simulated"
    assert record.schema_version == 2
    assert record.retrieval.hits[0].document_id == "eligibility"
    assert record.retrieval.hits[0].score > 0
    request = provider.requests[0]
    assert request.prompt == record.prompt
    assert request.settings == record.settings
    data = json.loads(request.prompt.input_text)
    assert set(data) == {"question", "documents"}
    assert data["question"] == record.question
    assert data["documents"][0]["content"] == record.context.documents[0].content
    assert data["documents"][0]["version"] == "1.0"
    assert "reference_facts" not in request.prompt.input_text
    assert "answerable" not in request.prompt.input_text
    assert record.generation.response_id == "simulated-response"
    assert record.settings.model == "requested-model"
    assert record.generation.returned_model == "returned-model-snapshot"
    assert record.generation.usage == TokenUsage(123, 45, 168)
    assert record.generation.latency_ms == 12.5
    assert (
        len(record.corpus_sha256) == len(record.dataset_sha256) == len(record.prompt.sha256) == 64
    )
    assert parse_capture(parse_json(path.read_text())) == record  # Independently loadable.
    report = run_evaluation(traces_path=path, top_k=1)
    assert report.schema_version == 2
    assert report.response_origin == "synthetic_fixture"
    assert report.captures == (record,)
    assert all(value == 1 for value in report.aggregate.metrics.values())
    assert json.loads(report_json(report))["human_review"] is None


@pytest.mark.parametrize("prompt_version", tuple(PROMPT_INSTRUCTIONS))
def test_gold_annotations_never_change_generation_requests(
    tmp_path: Path, prompt_version: str
) -> None:
    cases = [
        cast(dict[str, object], json.loads(line))
        for line in (DEFAULT_DATA / "cases.jsonl").read_text().splitlines()
    ]
    for case in cases:
        case["case_id"] = "relabeled-" + str(case["case_id"])
        case["category"] = "unanswerable"
        case["answerable"] = False
        case["relevant_document_ids"] = []
        case["reference_evidence"] = []
        case["reference_facts"] = ["GOLD-SECRET-ANNOTATION"]
        case["intent"] = "GOLD-SECRET-INTENT"
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps(case) + "\n" for case in cases))
    first = ScriptedProvider([generated(answer())])
    second = ScriptedProvider([generated(answer())])
    for provider, cases_path, name, case_id in [
        (first, DEFAULT_DATA / "cases.jsonl", "a", "case-02"),
        (second, path, "b", "relabeled-case-02"),
    ]:
        run_live_batch(
            provider,
            settings=GenerationSettings("model"),
            captures_path=tmp_path / f"{name}.jsonl",
            cases_path=cases_path,
            case_ids=[case_id],
            allow_live=True,
            prompt_version=prompt_version,
        )
    assert first.requests == second.requests
    serialized = canonical_json(first.requests[0].prompt.input_text)
    assert "GOLD-SECRET" not in serialized
    assert "case-02" not in serialized


@pytest.mark.parametrize("prompt_version", tuple(PROMPT_INSTRUCTIONS))
def test_selected_prompt_is_captured_and_replayed_without_outcome_repair(
    tmp_path: Path, prompt_version: str
) -> None:
    inconsistent = answer()
    inconsistent["outcome"] = "abstain"
    raw = generated(inconsistent)
    provider = ScriptedProvider([raw])
    path = tmp_path / "selected.jsonl"
    record = run_live_batch(
        provider,
        settings=GenerationSettings("model"),
        captures_path=path,
        allow_live=True,
        case_ids=["case-02"],
        prompt_version=prompt_version,
    )[0]
    assert record.prompt == provider.requests[0].prompt
    assert record.prompt.version == prompt_version
    assert record.prompt.instructions == PROMPT_INSTRUCTIONS[prompt_version]
    assert record.generation == raw
    assert record.response is not None
    assert record.response.outcome == "abstain"
    assert record.response.citations == ("eligibility",)
    assert record.response.text == inconsistent["text"]
    assert parse_capture(parse_json(path.read_text())) == record
    report = run_evaluation(traces_path=path)
    assert report.captures == (record,)
    assert report.aggregate.metrics["answerability_accuracy"] == 0
    assert report_json(report) == report_json(run_evaluation(traces_path=path))


def test_unknown_prompt_version_sends_no_requests_and_creates_no_capture(tmp_path: Path) -> None:
    provider = ScriptedProvider([])
    path = tmp_path / "unknown.jsonl"
    with pytest.raises(ValueError, match="unsupported prompt version"):
        run_live_batch(
            provider,
            settings=GenerationSettings("model"),
            captures_path=path,
            allow_live=True,
            prompt_version="benefits_rag_v999",
        )
    assert provider.requests == []
    assert not path.exists()


@pytest.mark.parametrize("prompt_version", tuple(PROMPT_INSTRUCTIONS))
@pytest.mark.parametrize(
    "tampering", ["version", "mislabeled_version", "instructions", "response_schema"]
)
def test_rehashed_unrecognized_prompt_snapshot_is_rejected(
    tmp_path: Path, prompt_version: str, tampering: str
) -> None:
    record = run_live_batch(
        ScriptedProvider([generated(answer())]),
        settings=GenerationSettings("model"),
        captures_path=tmp_path / "prompt.jsonl",
        allow_live=True,
        max_cases=1,
        prompt_version=prompt_version,
    )[0]
    payload = cast(dict[str, object], parse_json(capture_json(record)))
    prompt = cast(dict[str, object], payload["prompt"])
    if tampering == "version":
        prompt["version"] = "benefits_rag_v999"
    elif tampering == "mislabeled_version":
        prompt["version"] = (
            "benefits_rag_v2" if prompt_version == "benefits_rag_v1" else "benefits_rag_v1"
        )
    elif tampering == "instructions":
        prompt["instructions"] = str(prompt["instructions"]) + "\nModified instructions."
    else:
        cast(dict[str, object], prompt["response_schema"])["description"] = "Modified schema."
    prompt["sha256"] = prompt_hash(
        cast(str, prompt["version"]),
        cast(str, prompt["instructions"]),
        cast(dict[str, object], prompt["response_schema"]),
    )
    with pytest.raises(ValidationError, match="unsupported prompt"):
        parse_capture(payload)


@pytest.mark.parametrize("case_id", ["case-16", "case-17"])
def test_archived_first_hit_is_excluded_without_refill(tmp_path: Path, case_id: str) -> None:
    provider = ScriptedProvider(
        [
            generated(
                {
                    "outcome": "abstain",
                    "text": "Insufficient current context.",
                    "citations": [],
                    "evidence": [],
                }
            )
        ]
    )
    record = run_live_batch(
        provider,
        settings=GenerationSettings("model"),
        captures_path=tmp_path / "distractor.jsonl",
        case_ids=[case_id],
        top_k=1,
        allow_live=True,
    )[0]
    assert len(record.retrieval.hits) == 1
    assert record.context.documents == ()
    assert len(record.context.excluded) == 1
    assert record.context.excluded[0].document.status == "archived"
    assert record.context.excluded[0].reason == "archived_policy"
    assert json.loads(provider.requests[0].prompt.input_text)["documents"] == []
    report = run_evaluation(traces_path=tmp_path / "distractor.jsonl", top_k=1)
    assert report.aggregate.metrics["recall_at_k"] == 0
    assert report.aggregate.metrics["effective_context_coverage"] == 0


def test_outdated_source_misuse_is_a_quality_defect(tmp_path: Path) -> None:
    doc = load_corpus(DEFAULT_DATA / "kb")["enrollment-2025"]
    provider = ScriptedProvider(
        [generated(answer("Use the old deadline.", doc.document_id, doc.content.splitlines()[-1]))]
    )
    path = tmp_path / "old.jsonl"
    record = run_live_batch(
        provider,
        settings=GenerationSettings("model"),
        captures_path=path,
        case_ids=["case-17"],
        allow_live=True,
    )[0]
    assert record.context.excluded[0].document.document_id == "enrollment-2025"
    report = run_evaluation(
        traces_path=path, minimums={"citation_in_context": 1, "current_citation_rate": 1}
    )
    assert report.aggregate.metrics["citation_validity"] == 1
    assert report.aggregate.metrics["exact_quote_validity"] == 1
    assert report.aggregate.metrics["citation_in_context"] == 0
    assert report.aggregate.metrics["context_quote_validity"] == 0
    assert report.aggregate.metrics["current_citation_rate"] == 0
    assert report.results[0].context_evaluation is not None
    assert report.results[0].context_evaluation.quotations[0].reason == "not_in_generation_context"
    assert report.gate_status == "failed"


@pytest.mark.parametrize(
    ("response", "metric", "expected"),
    [
        (
            {
                "outcome": "answer",
                "text": "Unsupported invented benefit.",
                "citations": [],
                "evidence": [],
            },
            "citation_coverage",
            0,
        ),
        (answer(document_id="missing-document"), "citation_validity", 0),
        (answer(quote="Invented quotation."), "context_quote_validity", 0),
        (answer(document_id="dental-limits", quote="Unrelated quote."), "citation_in_context", 0),
        (
            {"outcome": "abstain", "text": "I cannot tell.", "citations": [], "evidence": []},
            "answerability_accuracy",
            0,
        ),
    ],
)
def test_prediction_defects_reach_independent_evaluators(
    tmp_path: Path,
    response: object,
    metric: str,
    expected: float,
) -> None:
    path = tmp_path / "defect.jsonl"
    run_live_batch(
        ScriptedProvider([generated(response)]),
        settings=GenerationSettings("model"),
        captures_path=path,
        allow_live=True,
        case_ids=["case-02"],
        top_k=1,
    )
    report = run_evaluation(traces_path=path, top_k=1, minimums={metric: 1})
    assert report.aggregate.metrics[metric] == expected
    assert report.gate_status == "failed"
    assert (
        main(
            [
                "replay",
                "--traces",
                str(path),
                "--top-k",
                "1",
                "--minimum",
                f"{metric}=1",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 1
    )


def test_substantively_wrong_answer_can_pass_every_structural_check(tmp_path: Path) -> None:
    path = tmp_path / "wrong.jsonl"
    run_live_batch(
        ScriptedProvider([generated(answer("Regular employees need 200 hours per week."))]),
        settings=GenerationSettings("model"),
        captures_path=path,
        allow_live=True,
        case_ids=["case-02"],
        top_k=1,
    )
    report = run_evaluation(traces_path=path, top_k=1)
    assert all(value == 1 for value in report.aggregate.metrics.values())
    payload = json.loads(report_json(report))
    assert payload["human_review"] is None
    assert "do not establish semantic faithfulness" in payload["measurement_limits"]
    assert report.results[0].response is not None
    assert "200 hours" in report.results[0].response.text


def test_correct_abstention_has_null_citation_denominators(tmp_path: Path) -> None:
    path = tmp_path / "abstain.jsonl"
    run_live_batch(
        ScriptedProvider(
            [
                generated(
                    {
                        "outcome": "abstain",
                        "text": "Personal records absent.",
                        "citations": [],
                        "evidence": [],
                    },
                    usage=None,
                )
            ]
        ),
        settings=GenerationSettings("model"),
        captures_path=path,
        allow_live=True,
        case_ids=["case-21"],
    )
    report = run_evaluation(traces_path=path, minimums={"citation_in_context": 0})
    assert report.aggregate.metrics["answerability_accuracy"] == 1
    assert report.aggregate.metrics["effective_context_coverage"] is None
    assert report.aggregate.metrics["citation_in_context"] is None
    assert report.aggregate.denominators["citation_in_context"] == 0
    assert report.threshold_failures[0].reason == "no_applicable_observations"
    assert report.captures[0].generation.usage is None


@pytest.mark.parametrize(
    "kind", ["authentication", "rate_limit", "timeout", "incomplete", "refusal"]
)
def test_provider_failures_are_preserved_not_abstentions_and_stop_batch(
    tmp_path: Path,
    kind: str,
) -> None:
    status = "incomplete" if kind == "incomplete" else "refusal" if kind == "refusal" else "error"
    failure = replace(
        generated(answer()),
        status=cast("GenerationStatus", status),
        error=ProviderError(
            kind, "Simulated failure.", "test-code", 429 if kind == "rate_limit" else None
        ),
        usage=None,
    )
    provider = ScriptedProvider([failure, generated(answer())])
    path = tmp_path / "failure.jsonl"
    records = run_live_batch(
        provider,
        settings=GenerationSettings("model"),
        captures_path=path,
        allow_live=True,
        max_cases=2,
    )
    assert len(records) == len(provider.requests) == 1
    assert records[0].response is None
    assert records[0].generation == failure
    report = run_evaluation(traces_path=path, minimums={"generation_success_rate": 1})
    assert report.aggregate.metrics["generation_success_rate"] == 0
    assert report.aggregate.metrics["answerability_accuracy"] is None
    assert len(report.omitted_case_ids) == 23
    assert main(["replay", "--traces", str(path), "--output", str(tmp_path / "report.json")]) == 2


@pytest.mark.parametrize("value", [{"text": "missing fields"}, {"outcome": "guess"}, []])
def test_malformed_model_output_is_captured_and_replayable(tmp_path: Path, value: object) -> None:
    path = tmp_path / "malformed.jsonl"
    record = run_live_batch(
        ScriptedProvider([generated(value)]),
        settings=GenerationSettings("model"),
        captures_path=path,
        allow_live=True,
        max_cases=1,
    )[0]
    assert record.generation.status == "malformed"
    assert record.response is None
    assert run_evaluation(traces_path=path).aggregate.metrics["generation_success_rate"] == 0


class BatchOptions(TypedDict, total=False):
    allow_live: bool
    max_cases: int
    case_ids: list[str]


def test_live_opt_in_and_count_limits_precede_provider_requests(tmp_path: Path) -> None:
    provider = ScriptedProvider([generated(answer())])
    options: list[BatchOptions] = [
        {},
        {"allow_live": True, "max_cases": 0},
        {"allow_live": True, "max_cases": 25},
        {"allow_live": True, "max_cases": 1, "case_ids": ["case-01", "case-02"]},
        {"allow_live": True, "case_ids": ["unknown"]},
    ]
    for kwargs in options:
        with pytest.raises(ValueError):
            run_live_batch(
                provider,
                settings=GenerationSettings("model"),
                captures_path=tmp_path / "never.jsonl",
                **kwargs,
            )
    assert provider.requests == []
    assert not (tmp_path / "never.jsonl").exists()
    cases = load_cases(DEFAULT_DATA / "cases.jsonl", load_corpus(DEFAULT_DATA / "kb"))
    assert len(select_cases(cases, [], 3)) == 3
    assert [case.case_id for case in select_cases(cases, ["case-02", "case-01"], 3)] == [
        "case-02",
        "case-01",
    ]
    with pytest.raises(ValueError, match="duplicate"):
        select_cases(cases, ["case-01", "case-01"], 3)


def test_no_overwrite_and_actual_artifacts_require_private_directory(tmp_path: Path) -> None:
    path = tmp_path / "existing.jsonl"
    path.write_text("keep")
    provider = ScriptedProvider([generated(answer())])
    with pytest.raises(FileExistsError):
        run_live_batch(
            provider,
            settings=GenerationSettings("model"),
            captures_path=path,
            allow_live=True,
            max_cases=1,
        )
    assert provider.requests == []
    assert path.read_text() == "keep"
    with pytest.raises(ValueError, match="live-captures"):
        require_private_path(tmp_path / "reports.json")


def test_cli_requires_opt_in_and_key(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["live", "--model", "model"]) == 2
    assert "--allow-live" in capsys.readouterr().err
    assert main(["live", "--allow-live", "--model", "model", "--max-cases", "1"]) == 2
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY is required" in captured.err
    assert "Intended requests: 1 sequential" in captured.out
    assert "strict USD spending cap" in captured.out


@pytest.mark.parametrize(
    "arguments",
    [
        ["--max-cases", "0"],
        ["--max-output-tokens", "4097"],
        ["--timeout", "nan"],
        ["--minimum", "mrr=nan"],
        ["--case", "unknown"],
        ["--top-k", "0"],
    ],
)
def test_invalid_cli_settings_never_start_a_request(arguments: list[str]) -> None:
    assert main(["live", "--allow-live", "--model", "model", *arguments]) == 2


def test_cli_model_must_be_explicit() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["live", "--allow-live"])
    assert exc.value.code == 2


@pytest.mark.parametrize("prompt_version", [None, "benefits_rag_v1", "benefits_rag_v2"])
def test_live_cli_uses_fake_provider_and_reports_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prompt_version: str | None,
) -> None:
    provider = ScriptedProvider([generated(answer(), usage=None)])
    monkeypatch.setattr("ai_quality_evals.provider.OpenAIProvider", lambda: provider)
    path = tmp_path / "live-captures" / "cli.jsonl"
    assert (
        main(
            [
                "live",
                "--allow-live",
                "--model",
                "model",
                "--case",
                "case-02",
                "--top-k",
                "1",
                "--captures",
                str(path),
                "--minimum",
                "generation_success_rate=1",
                *(["--prompt-version", prompt_version] if prompt_version is not None else []),
            ]
        )
        == 0
    )
    assert len(provider.requests) == 1
    selected_version = prompt_version or "benefits_rag_v1"
    assert provider.requests[0].prompt.version == selected_version
    output = capsys.readouterr().out
    assert f"prompt_version={selected_version}" in output
    assert "usage reported for 0" in output
    assert "Missing usage is unknown" in output
    assert (
        json.loads((path.parent / "report.json").read_text())["response_origin"]
        == "synthetic_fixture"
    )


def test_cli_rejects_unknown_prompt_before_constructing_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_provider() -> None:
        raise AssertionError("Unknown prompt must fail before provider construction")

    monkeypatch.setattr("ai_quality_evals.provider.OpenAIProvider", forbidden_provider)
    path = tmp_path / "live-captures" / "unknown.jsonl"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "live",
                "--allow-live",
                "--model",
                "model",
                "--prompt-version",
                "benefits_rag_v999",
                "--captures",
                str(path),
            ]
        )
    assert exc.value.code == 2
    assert not path.exists()


def test_offline_replay_preserves_captured_facts_across_runs_and_hash_seeds(
    tmp_path: Path,
    captured: CapturedRecord,
) -> None:
    path = tmp_path / "replay.jsonl"
    path.write_text(capture_json(captured))
    first = run_evaluation(traces_path=path, top_k=1)
    assert report_json(first) == report_json(run_evaluation(traces_path=path, top_k=1))
    assert first.captures[0].generation.latency_ms == 12.5
    outputs: list[bytes] = []
    for seed in ["1", "927"]:
        destination = tmp_path / f"report-{seed}.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ai_quality_evals",
                "replay",
                "--traces",
                str(path),
                "--top-k",
                "1",
                "--output",
                str(destination),
            ],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        outputs.append(destination.read_bytes())
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("execution", "live", "origin and execution"),
        ("prompt", {}, "missing"),
        ("corpus_sha256", "abc", "SHA-256"),
        ("response", None, "completed capture"),
        ("schema_version", True, "schema_version"),
    ],
)
def test_malformed_capture_is_rejected(
    captured: CapturedRecord,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = cast(dict[str, object], parse_json(capture_json(captured)))
    payload[field] = value
    with pytest.raises(ValidationError, match=message):
        parse_capture(payload)


def test_capture_rejects_tampered_context_prompt_output_and_input_hash(
    tmp_path: Path,
    captured: CapturedRecord,
) -> None:
    payload = cast(dict[str, object], parse_json(capture_json(captured)))
    cast(dict[str, object], payload["prompt"])["input_text"] = "gold leakage"
    with pytest.raises(ValidationError, match="prompt input"):
        parse_capture(payload)
    payload = cast(dict[str, object], parse_json(capture_json(captured)))
    cast(dict[str, object], payload["prompt"])["instructions"] = "changed"
    with pytest.raises(ValidationError, match="prompt hash"):
        parse_capture(payload)
    payload = cast(dict[str, object], parse_json(capture_json(captured)))
    cast(dict[str, object], payload["context"])["text"] = "wrong"
    with pytest.raises(ValidationError, match="context differs"):
        parse_capture(payload)
    payload = cast(dict[str, object], parse_json(capture_json(captured)))
    cast(dict[str, object], payload["response"])["text"] = "changed"
    with pytest.raises(ValidationError, match="structured response differs"):
        parse_capture(payload)
    path = tmp_path / "changed-hash.jsonl"
    path.write_text(capture_json(replace(captured, dataset_sha256="0" * 64)))
    with pytest.raises(ValidationError, match="hash mismatch"):
        run_evaluation(traces_path=path, top_k=1)
    assert main(["validate", "--traces", str(path)]) == 2


def test_recorded_format_fixture_is_explicitly_simulated() -> None:
    report = run_evaluation(traces_path=Path(__file__).parent / "fixtures/captured-format.jsonl")
    assert report.response_origin == "synthetic_fixture"
    assert all(record.execution == "simulated" for record in report.captures)
    assert report.aggregate.case_count == 2


def test_default_batch_sends_exactly_three_requests(tmp_path: Path) -> None:
    abstain = {
        "outcome": "abstain",
        "text": "Simulated count test.",
        "citations": [],
        "evidence": [],
    }
    provider = ScriptedProvider([generated(abstain)] * 3)
    records = run_live_batch(
        provider,
        settings=GenerationSettings("model"),
        captures_path=tmp_path / "three.jsonl",
        allow_live=True,
    )
    assert len(provider.requests) == len(records) == 3
    assert [record.case_id for record in records] == ["case-01", "case-02", "case-03"]


def test_whitespace_malformed_output_is_retained(tmp_path: Path) -> None:
    raw = replace(generated(answer()), output_text="   ")
    path = tmp_path / "whitespace.jsonl"
    record = run_live_batch(
        ScriptedProvider([raw]),
        settings=GenerationSettings("model"),
        captures_path=path,
        allow_live=True,
        max_cases=1,
    )[0]
    assert record.generation.output_text == "   "
    assert record.generation.status == "malformed"
    assert run_evaluation(traces_path=path).captures == (record,)


def test_missing_evidence_requires_a_coverage_gate(tmp_path: Path) -> None:
    value = answer()
    value["evidence"] = []
    path = tmp_path / "missing-evidence.jsonl"
    run_live_batch(
        ScriptedProvider([generated(value)]),
        settings=GenerationSettings("model"),
        captures_path=path,
        allow_live=True,
        max_cases=1,
        case_ids=["case-02"],
    )
    report = run_evaluation(traces_path=path, minimums={"evidence_coverage": 1})
    assert report.aggregate.metrics["evidence_coverage"] == 0
    assert report.aggregate.metrics["context_quote_validity"] is None
    assert report.gate_status == "failed"


def test_raw_retrieval_is_distinct_from_effective_context_at_three(tmp_path: Path) -> None:
    path = tmp_path / "context.jsonl"
    run_live_batch(
        ScriptedProvider(
            [
                generated(
                    {
                        "outcome": "abstain",
                        "text": "Simulated context audit.",
                        "citations": [],
                        "evidence": [],
                    }
                )
            ]
        ),
        settings=GenerationSettings("model"),
        captures_path=path,
        allow_live=True,
        case_ids=["case-16"],
    )
    report = run_evaluation(traces_path=path)
    record = report.captures[0]
    assert record.retrieval.hits[0].document_id == "wellness-2025"
    assert record.context.documents[0].document_id == "wellness-2026"
    assert report.aggregate.metrics["recall_at_k"] == 1
    assert report.aggregate.metrics["mrr"] == 0.5
    assert report.aggregate.metrics["effective_context_coverage"] == 1
    # Lower replay K changes retrieval scoring only, preserving historical supplied context.
    lower = run_evaluation(traces_path=path, top_k=1)
    assert lower.aggregate.metrics["recall_at_k"] == 0
    assert lower.aggregate.metrics["effective_context_coverage"] == 1


def test_offline_import_and_replay_never_import_openai() -> None:
    script = """
import importlib.abc
import sys
class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openai" or fullname.startswith("openai."):
            raise AssertionError("Offline import attempted to load OpenAI SDK")
sys.meta_path.insert(0, BlockSDK())
from pathlib import Path
import ai_quality_evals
import ai_quality_evals.provider
from ai_quality_evals.runner import run_evaluation
assert run_evaluation().aggregate.case_count == 24
assert run_evaluation(traces_path=Path(sys.argv[1])).aggregate.case_count == 2
assert "openai" not in sys.modules
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(Path(__file__).parent / "fixtures/captured-format.jsonl"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
