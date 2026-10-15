"""All Ollama responses are fabricated; the suite's socket guard forbids inference."""

import json
from pathlib import Path
from typing import cast

import pytest

from ai_quality_evals.capture import capture_json, parse_capture
from ai_quality_evals.cli import main
from ai_quality_evals.live import run_live_batch
from ai_quality_evals.models import GenerationRequest, GenerationSettings, TokenUsage
from ai_quality_evals.ollama import OllamaProvider, _HTTPTransport
from ai_quality_evals.prompt import build_context, build_prompt, canonical_json
from ai_quality_evals.retrieval import BM25Retriever
from ai_quality_evals.runner import DEFAULT_DATA, report_json, run_evaluation
from ai_quality_evals.validation import ValidationError, load_corpus, parse_json

MODEL = "qwen2.5:3b"
QUOTE = "Regular employees scheduled for at least 20 hours per week are eligible for benefits."
ANSWER = canonical_json(
    {
        "outcome": "answer",
        "text": "Regular employees need at least 20 scheduled hours per week.",
        "citations": ["eligibility"],
        "evidence": [{"document_id": "eligibility", "quote": QUOTE}],
    }
)
LOCAL_INFO: dict[str, object] = {"model_info": {"general.architecture": "qwen2"}}


def chat_payload(text: str = ANSWER) -> dict[str, object]:
    return {
        "model": MODEL,
        "message": {"role": "assistant", "content": text},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 123,
        "eval_count": 45,
    }


class FakeTransport:
    def __init__(self, chat: object, info: object = LOCAL_INFO):
        self.chat = chat
        self.info = info
        self.calls: list[tuple[str, dict[str, object], float]] = []

    def __call__(self, path: str, parameters: dict[str, object], timeout_seconds: float) -> object:
        self.calls.append((path, parameters, timeout_seconds))
        return self.info if path == "/api/show" else self.chat


@pytest.fixture
def generation_request() -> GenerationRequest:
    documents = load_corpus(DEFAULT_DATA / "kb")
    question = "How many hours are required?"
    context = build_context(BM25Retriever(documents.values()).retrieve(question), documents)
    return GenerationRequest(GenerationSettings(MODEL, 7.5, 300), build_prompt(question, context))


def test_native_schema_prompt_usage_and_simulated_provenance(
    generation_request: GenerationRequest,
) -> None:
    transport = FakeTransport(chat_payload())
    provider = OllamaProvider(transport=transport)
    result = provider.generate(generation_request)
    assert provider.name == "ollama" and provider.is_simulated
    assert transport.calls == [
        ("/api/show", {"model": MODEL}, 7.5),
        (
            "/api/chat",
            {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": generation_request.prompt.instructions},
                    {"role": "user", "content": generation_request.prompt.input_text},
                ],
                "format": generation_request.prompt.response_schema,
                "stream": False,
                "options": {"num_predict": 300},
            },
            7.5,
        ),
    ]
    assert result.status == "completed" and result.output_text == ANSWER
    assert result.returned_model == MODEL and result.usage == TokenUsage(123, 45, 168)
    assert result.response_id is None and result.request_id is None
    assert result.latency_ms >= 0
    provider.close()


@pytest.mark.parametrize(
    "text",
    [
        "bad-json",
        "   ",
        "{}",
        '{"outcome":"answer","outcome":"abstain"}',
        '{"text":NaN}',
        '{"outcome":"answer","text":"","citations":[],"evidence":[]}',
        '{"outcome":"abstain","text":"Missing","citations":[],"evidence":[],"extra":true}',
    ],
)
def test_malformed_json_is_preserved_without_repair_or_retry(
    generation_request: GenerationRequest,
    text: str,
) -> None:
    transport = FakeTransport(chat_payload(text))
    result = OllamaProvider(transport=transport).generate(generation_request)
    assert result.status == "malformed" and result.output_text == text
    assert result.returned_model == MODEL and result.usage == TokenUsage(123, 45, 168)
    assert result.error is not None and result.error.kind == "malformed_output"
    assert len(transport.calls) == 2


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ("prompt_eval_count", TokenUsage(None, 45, None)),
        ("eval_count", TokenUsage(123, None, None)),
        ("both", None),
    ],
)
def test_missing_usage_is_unknown_and_available_counts_survive(
    generation_request: GenerationRequest,
    missing: str,
    expected: TokenUsage | None,
) -> None:
    payload = chat_payload()
    for key in ("prompt_eval_count", "eval_count") if missing == "both" else (missing,):
        del payload[key]
    result = OllamaProvider(transport=FakeTransport(payload)).generate(generation_request)
    assert result.status == "completed" and result.usage == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_eval_count", True),
        ("eval_count", -1),
        ("eval_count", 1.5),
        ("message", []),
        ("message", {"role": "user", "content": ANSWER}),
        ("message", {"role": "assistant", "content": 123}),
    ],
)
def test_malformed_envelopes_are_failures(
    generation_request: GenerationRequest,
    field: str,
    value: object,
) -> None:
    payload = chat_payload()
    payload[field] = value
    result = OllamaProvider(transport=FakeTransport(payload)).generate(generation_request)
    assert result.status == "malformed" and result.error is not None


@pytest.mark.parametrize(("field", "value"), [("done", False), ("done_reason", "length")])
def test_incomplete_generation_is_not_a_completed_answer(
    generation_request: GenerationRequest,
    field: str,
    value: object,
) -> None:
    payload = chat_payload()
    payload[field] = value
    result = OllamaProvider(transport=FakeTransport(payload)).generate(generation_request)
    assert result.status == "incomplete" and result.output_text == ANSWER
    assert result.usage == TokenUsage(123, 45, 168)


@pytest.mark.parametrize(
    "model",
    [
        "qwen:cloud",
        "gpt-oss:20b-cloud",
        "https://ollama.com/qwen",
        "host:11434/qwen",
        "registry.ollama.ai/library/qwen",
        "ollama.com/qwen",
        " qwen2.5:3b",
        "",
    ],
)
def test_remote_model_names_are_rejected_before_transport(
    model: str, generation_request: GenerationRequest
) -> None:
    transport = FakeTransport(chat_payload())
    with pytest.raises(ValueError):
        OllamaProvider(transport=transport).generate(
            GenerationRequest(GenerationSettings(model), generation_request.prompt)
        )
    assert transport.calls == []


@pytest.mark.parametrize(
    "info",
    [
        {**LOCAL_INFO, "remote_host": "https://ollama.com"},
        {**LOCAL_INFO, "remote_model": "cloud-alias"},
        {"model_info": {}},
        {"model_info": {"general.architecture": ""}},
    ],
)
def test_cloud_aliases_or_unverified_weights_never_reach_chat(
    generation_request: GenerationRequest,
    info: object,
) -> None:
    transport = FakeTransport(chat_payload(), info)
    result = OllamaProvider(transport=transport).generate(generation_request)
    assert result.status == "error" and result.error is not None
    assert result.error.kind == "local_only"
    assert [path for path, _, _ in transport.calls] == ["/api/show"]


def test_unexpected_remote_chat_is_a_failure(generation_request: GenerationRequest) -> None:
    payload = chat_payload()
    payload["remote_host"] = "https://ollama.com"
    result = OllamaProvider(transport=FakeTransport(payload)).generate(generation_request)
    assert result.status == "error" and result.error is not None
    assert result.error.kind == "local_only"
    assert result.output_text == ANSWER and result.usage == TokenUsage(123, 45, 168)


def test_missing_returned_model_is_not_invented(generation_request: GenerationRequest) -> None:
    payload = chat_payload()
    del payload["model"]
    result = OllamaProvider(transport=FakeTransport(payload)).generate(generation_request)
    assert result.status == "completed" and result.returned_model is None


def test_remote_returned_model_is_rejected(generation_request: GenerationRequest) -> None:
    payload = chat_payload()
    payload["model"] = "other:cloud"
    result = OllamaProvider(transport=FakeTransport(payload)).generate(generation_request)
    assert result.status == "error" and result.error is not None
    assert result.error.kind == "local_only" and result.returned_model == "other:cloud"


class FakeResponse:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self.body = body

    def read(self) -> bytes:
        return self.body


def http_response(value: object) -> FakeResponse:
    return FakeResponse(200, canonical_json(value).encode("utf-8"))


class FakeConnection:
    def __init__(self, reply: FakeResponse | Exception):
        self.reply = reply
        self.closed = False
        self.sent: tuple[str, str, bytes, dict[str, str]] | None = None

    def request(self, method: str, path: str, body: bytes, headers: dict[str, str]) -> None:
        self.sent = (method, path, body, headers)

    def getresponse(self) -> FakeResponse:
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    def close(self) -> None:
        self.closed = True


class FakeHTTP:
    def __init__(self, replies: list[FakeResponse | Exception]):
        self.replies = replies
        self.connections: list[FakeConnection] = []
        self.targets: list[tuple[str, int, float]] = []

    def connect(self, host: str, port: int, *, timeout: float) -> FakeConnection:
        self.targets.append((host, port, timeout))
        connection = FakeConnection(self.replies[len(self.connections)])
        self.connections.append(connection)
        return connection


@pytest.mark.parametrize(
    ("reply", "kind"),
    [
        (ConnectionRefusedError("fabricated unavailable server"), "connection"),
        (TimeoutError("fabricated timeout"), "timeout"),
        (FakeResponse(404, b'{"error":"model not found; pull it manually"}'), "provider_error"),
        (FakeResponse(302, b"redirect to remote"), "provider_error"),
        (FakeResponse(500, b"not JSON"), "provider_error"),
    ],
)
@pytest.mark.parametrize("phase", ["show", "chat"])
def test_native_failures_are_typed_with_no_retry_or_fallback(
    generation_request: GenerationRequest,
    monkeypatch: pytest.MonkeyPatch,
    reply: FakeResponse | Exception,
    kind: str,
    phase: str,
) -> None:
    replies = [http_response(LOCAL_INFO), reply] if phase == "chat" else [reply]
    http = FakeHTTP(replies)
    monkeypatch.setattr("ai_quality_evals.ollama.http.client.HTTPConnection", http.connect)
    provider = OllamaProvider()
    assert not provider.is_simulated
    result = provider.generate(generation_request)
    assert result.status == "error" and result.error is not None
    assert result.error.kind == kind and result.usage is None and result.output_text is None
    assert len(http.connections) == len(replies)
    assert all(connection.closed for connection in http.connections)
    assert http.targets == [("127.0.0.1", 11434, 7.5)] * len(replies)
    if isinstance(reply, FakeResponse):
        assert result.error.status_code == reply.status


@pytest.mark.parametrize("body", [b"bad-json", b"\xff", b'{"done":true,"done":false}'])
def test_invalid_http_json_is_a_diagnostic(
    generation_request: GenerationRequest,
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
) -> None:
    http = FakeHTTP([http_response(LOCAL_INFO), FakeResponse(200, body)])
    monkeypatch.setattr("ai_quality_evals.ollama.http.client.HTTPConnection", http.connect)
    result = OllamaProvider().generate(generation_request)
    assert result.status == "malformed"
    assert all(connection.closed for connection in http.connections)


def test_loopback_ignores_endpoint_and_proxy_environment_and_has_no_download_path(
    generation_request: GenerationRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("OLLAMA_HOST", "OPENAI_BASE_URL", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://remote.invalid:8888")
    http = FakeHTTP([http_response(LOCAL_INFO), http_response(chat_payload())])
    monkeypatch.setattr("ai_quality_evals.ollama.http.client.HTTPConnection", http.connect)
    assert OllamaProvider().generate(generation_request).status == "completed"
    assert http.targets == [("127.0.0.1", 11434, 7.5)] * 2
    sent = http.connections[1].sent
    assert sent is not None
    assert sent[:2] == ("POST", "/api/chat")
    assert json.loads(sent[2])["format"] == generation_request.prompt.response_schema
    assert sent[3] == {"Content-Type": "application/json"}
    with pytest.raises(ValueError, match="endpoint"):
        _HTTPTransport()("/api/pull", {"model": MODEL}, 1)
    assert len(http.connections) == 2


@pytest.mark.parametrize("simulated", [True, False])
@pytest.mark.parametrize("partial", [True, False])
def test_v2_provenance_usage_and_deterministic_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simulated: bool,
    partial: bool,
) -> None:
    payload = chat_payload()
    if partial:
        del payload["eval_count"]
    transport = FakeTransport(payload)
    http = FakeHTTP([http_response(LOCAL_INFO), http_response(payload)])
    monkeypatch.setattr("ai_quality_evals.ollama.http.client.HTTPConnection", http.connect)
    provider = OllamaProvider(transport=transport) if simulated else OllamaProvider()
    path = tmp_path / "live-captures" / "traces.jsonl"
    record = run_live_batch(
        provider,
        settings=GenerationSettings(MODEL),
        captures_path=path,
        allow_live=True,
        case_ids=["case-02"],
        top_k=1,
    )[0]
    assert record.provider == "ollama" and record.settings.model == MODEL
    assert record.trace_origin == ("synthetic_fixture" if simulated else "live_capture")
    assert record.execution == ("simulated" if simulated else "live")
    assert record.generation.returned_model == MODEL
    assert record.generation.output_text == ANSWER
    assert record.generation.usage == (
        TokenUsage(123, None, None) if partial else TokenUsage(123, 45, 168)
    )
    assert parse_capture(parse_json(capture_json(record))) == record
    first = run_evaluation(traces_path=path, top_k=1)
    assert first.captures == (record,) and first.response_origin == record.trace_origin
    assert all(value == 1 for value in first.aggregate.metrics.values())
    assert report_json(first) == report_json(run_evaluation(traces_path=path, top_k=1))
    # Replay cannot relax OpenAI's existing usage contract or admit unknown real providers.
    raw = cast(dict[str, object], parse_json(capture_json(record)))
    raw["trace_origin"], raw["execution"], raw["provider"] = "live_capture", "live", "unknown"
    with pytest.raises(ValidationError, match="provider"):
        parse_capture(raw)
    raw["provider"] = "openai"
    if partial:
        with pytest.raises(ValidationError, match="nonnegative integer"):
            parse_capture(raw)
    else:
        assert parse_capture(raw).provider == "openai"


@pytest.mark.parametrize("prompt_version", ["benefits_rag_v1", "benefits_rag_v2"])
def test_local_cli_without_key_captures_and_reports_native_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    prompt_version: str,
) -> None:
    # Exercise production provenance through a fake HTTP connection, never a real socket.
    http = FakeHTTP([http_response(LOCAL_INFO), http_response(chat_payload())])
    monkeypatch.setattr("ai_quality_evals.ollama.http.client.HTTPConnection", http.connect)
    path = tmp_path / "live-captures" / "local" / "traces.jsonl"
    assert (
        main(
            [
                "live",
                "--provider",
                "ollama",
                "--allow-live",
                "--model",
                MODEL,
                "--prompt-version",
                prompt_version,
                "--case",
                "case-02",
                "--max-cases",
                "1",
                "--top-k",
                "1",
                "--timeout",
                "12.5",
                "--captures",
                str(path),
                "--minimum",
                "generation_success_rate=1",
            ]
        )
        == 0
    )
    assert http.targets == [("127.0.0.1", 11434, 12.5)] * 2
    report = json.loads((path.parent / "report.json").read_text())
    assert report["response_origin"] == "live_capture"
    assert report["captures"][0]["provider"] == "ollama"
    assert report["captures"][0]["prompt"]["version"] == prompt_version
    sent = http.connections[-1].sent
    assert sent is not None
    assert (
        json.loads(sent[2])["messages"][0]["content"]
        == (report["captures"][0]["prompt"]["instructions"])
    )
    assert report["captures"][0]["settings"]["max_retries"] == 0
    output = capsys.readouterr()
    assert "no cloud fallback" in output.out and output.err == ""
    assert "known input tokens=123" in output.out
    assert "known output tokens=45" in output.out


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["--allow-live", "--max-cases", "0"],
        ["--allow-live", "--max-cases", "25"],
        ["--allow-live", "--max-cases", "1", "--case", "case-01", "--case", "case-02"],
        ["--allow-live", "--timeout", "nan"],
        ["--allow-live", "--timeout", "0"],
        ["--allow-live", "--max-output-tokens", "4097"],
        ["--allow-live", "--model", "qwen:cloud"],
        ["--allow-live", "--minimum", "mrr=nan"],
        ["--allow-live", "--captures", "reports/local.jsonl"],
    ],
)
def test_local_cli_safety_checks_precede_http(
    arguments: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Invalid configuration attempted HTTP")

    monkeypatch.setattr("ai_quality_evals.ollama.http.client.HTTPConnection", forbidden)
    assert main(["live", "--provider", "ollama", "--model", MODEL, *arguments]) == 2


@pytest.mark.parametrize(
    "failure",
    [
        http_response(chat_payload("malformed")),
        ConnectionRefusedError("fake"),
        TimeoutError("fake"),
    ],
)
def test_local_failure_is_saved_reported_replayable_and_stops_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: FakeResponse | Exception,
) -> None:
    http = FakeHTTP([http_response(LOCAL_INFO), failure])
    monkeypatch.setattr("ai_quality_evals.ollama.http.client.HTTPConnection", http.connect)
    path = tmp_path / "live-captures" / "failed" / "traces.jsonl"
    assert (
        main(
            [
                "live",
                "--provider",
                "ollama",
                "--allow-live",
                "--model",
                MODEL,
                "--max-cases",
                "3",
                "--captures",
                str(path),
            ]
        )
        == 2
    )
    assert len(http.connections) == 2  # One preflight, one chat attempt; no second case or retry.
    report = run_evaluation(traces_path=path)
    assert report.aggregate.case_count == 1 and len(report.omitted_case_ids) == 23
    assert report.aggregate.metrics["generation_success_rate"] == 0
    assert report.captures[0].response is None and report.captures[0].provider == "ollama"
    assert (path.parent / "report.json").exists()
