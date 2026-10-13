import json
import sys
from types import ModuleType

import pytest

from ai_quality_evals.models import GenerationRequest, GenerationSettings, ProviderError, TokenUsage
from ai_quality_evals.prompt import build_context, build_prompt
from ai_quality_evals.provider import OpenAIProvider, TransportError
from ai_quality_evals.retrieval import BM25Retriever
from ai_quality_evals.runner import DEFAULT_DATA
from ai_quality_evals.validation import load_corpus


@pytest.fixture
def generation_request() -> GenerationRequest:
    documents = load_corpus(DEFAULT_DATA / "kb")
    question = "How many hours are required?"
    context = build_context(BM25Retriever(documents.values()).retrieve(question), documents)
    return GenerationRequest(
        GenerationSettings("requested-model", 5.0, 300), build_prompt(question, context)
    )


def payload(
    *, text: str = '{"outcome":"abstain","text":"Missing records.","citations":[],"evidence":[]}'
) -> dict[str, object]:
    return {
        "id": "resp-test",
        "request_id": "req-test",
        "model": "model-snapshot",
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
        "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    }


def test_adapter_sends_official_responses_parameters_and_parses_actual_usage(
    generation_request: GenerationRequest,
) -> None:
    sent: list[dict[str, object]] = []

    def transport(parameters: dict[str, object]) -> object:
        sent.append(parameters)
        return payload()

    provider = OpenAIProvider(transport=transport)
    result = provider.generate(generation_request)
    assert provider.is_simulated
    assert sent[0] == {
        "model": "requested-model",
        "instructions": generation_request.prompt.instructions,
        "input": generation_request.prompt.input_text,
        "max_output_tokens": 300,
        "timeout": 5.0,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "benefits_response",
                "strict": True,
                "schema": generation_request.prompt.response_schema,
            }
        },
    }
    assert result.status == "completed"
    assert result.response_id == "resp-test"
    assert result.request_id == "req-test"
    assert result.returned_model == "model-snapshot"
    assert result.usage == TokenUsage(100, 20, 120)
    assert result.latency_ms >= 0
    provider.close()


@pytest.mark.parametrize(
    "kind", ["authentication", "rate_limit", "timeout", "connection", "provider_error"]
)
def test_transport_error_preserves_provider_diagnostics_without_retry(
    generation_request: GenerationRequest,
    kind: str,
) -> None:
    count = 0

    def transport(parameters: dict[str, object]) -> object:
        nonlocal count
        count += 1
        raise TransportError(
            ProviderError(kind, "Simulated provider error", "error-code", 429), "req-fail"
        )

    result = OpenAIProvider(transport=transport).generate(generation_request)
    assert count == 1
    assert result.status == "error"
    assert result.error == ProviderError(kind, "Simulated provider error", "error-code", 429)
    assert result.request_id == "req-fail"
    assert result.response_id is None
    assert result.usage is None


@pytest.mark.parametrize(
    ("status", "details", "expected"),
    [
        ("incomplete", {"reason": "max_output_tokens"}, "incomplete"),
        ("failed", {"code": "server_error", "message": "Upstream failed"}, "error"),
        ("queued", None, "error"),
    ],
)
def test_incomplete_and_other_statuses_preserve_response_facts(
    generation_request: GenerationRequest,
    status: str,
    details: object,
    expected: str,
) -> None:
    value = payload()
    value["status"] = status
    value["incomplete_details" if status == "incomplete" else "error"] = details
    result = OpenAIProvider(transport=lambda parameters: value).generate(generation_request)
    assert result.status == expected
    assert result.response_id == "resp-test"
    assert result.usage == TokenUsage(100, 20, 120)
    assert result.error is not None
    if status == "incomplete":
        assert result.error.code == "max_output_tokens"


def test_refusal_is_not_a_model_abstention(generation_request: GenerationRequest) -> None:
    value = payload()
    value["output"] = [
        {"type": "message", "content": [{"type": "refusal", "refusal": "Cannot comply."}]}
    ]
    result = OpenAIProvider(transport=lambda parameters: value).generate(generation_request)
    assert result.status == "refusal"
    assert result.output_text is None
    assert result.error is not None and result.error.kind == "refusal"


@pytest.mark.parametrize(
    "text",
    [
        "bad-json",
        "{}",
        '{"outcome":"answer","outcome":"abstain"}',
        '{"text":NaN}',
        '{"outcome":"answer","text":"","citations":[],"evidence":[]}',
    ],
)
def test_invalid_json_and_response_schema_are_diagnostics(
    generation_request: GenerationRequest, text: str
) -> None:
    result = OpenAIProvider(transport=lambda parameters: payload(text=text)).generate(
        generation_request
    )
    assert result.status == "malformed"
    assert result.output_text == text
    assert result.response_id == "resp-test"
    assert result.usage is not None


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"status": "completed", "output": []},
        {"status": "completed", "output": "bad"},
        {"status": "completed", "usage": {"input_tokens": True}},
    ],
)
def test_malformed_provider_envelopes_are_handled(
    generation_request: GenerationRequest, value: object
) -> None:
    result = OpenAIProvider(transport=lambda parameters: value).generate(generation_request)
    assert result.status == "malformed"
    assert result.error is not None


def test_missing_usage_and_model_metadata_are_not_invented(
    generation_request: GenerationRequest,
) -> None:
    value = payload()
    del value["usage"]
    del value["model"]
    result = OpenAIProvider(transport=lambda parameters: value).generate(generation_request)
    assert result.status == "completed"
    assert result.usage is None
    assert result.returned_model is None


class SDKError(Exception):
    status_code: int | None = 400
    code = "safe-code"
    request_id = "req-sdk"


class AuthenticationError(SDKError):
    status_code = 401


class RateLimitError(SDKError):
    status_code = 429


class APITimeoutError(SDKError):
    status_code = None


class APIConnectionError(SDKError):
    status_code = None


@pytest.mark.parametrize(
    ("error_type", "kind"),
    [
        (AuthenticationError, "authentication"),
        (RateLimitError, "rate_limit"),
        (APITimeoutError, "timeout"),
        (APIConnectionError, "connection"),
        (SDKError, "provider_error"),
    ],
)
def test_sdk_exception_mapping_retry_settings_and_secret_redaction(
    monkeypatch: pytest.MonkeyPatch,
    generation_request: GenerationRequest,
    error_type: type[SDKError],
    kind: str,
) -> None:
    key = "sk-simulated-secret-for-unit-test"
    monkeypatch.setenv("OPENAI_API_KEY", key)
    sdk = ModuleType("openai")
    for name, value in {
        "APIError": SDKError,
        "AuthenticationError": AuthenticationError,
        "RateLimitError": RateLimitError,
        "APITimeoutError": APITimeoutError,
        "APIConnectionError": APIConnectionError,
    }.items():
        setattr(sdk, name, value)
    initialized: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []
    closed: list[bool] = []

    class Responses:
        def create(self, **parameters: object) -> object:
            calls.append(parameters)
            raise error_type(f"Provider rejected credential {key}; Bearer secret-token")

    class Client:
        def __init__(self, **parameters: object):
            initialized.append(parameters)
            self.responses = Responses()

        def close(self) -> None:
            closed.append(True)

    sdk.__dict__["OpenAI"] = Client
    monkeypatch.setitem(sys.modules, "openai", sdk)
    provider = OpenAIProvider()
    assert provider.is_simulated is False
    result = provider.generate(generation_request)
    provider.close()
    assert initialized[0]["max_retries"] == 0
    assert len(calls) == 1
    assert closed == [True]
    assert result.error is not None and result.error.kind == kind
    assert result.error.status_code == error_type.status_code
    assert result.error.code == "safe-code"
    assert result.request_id == "req-sdk"
    assert key not in json.dumps(result.error.message)
    assert "secret-token" not in result.error.message
    assert "[REDACTED]" in result.error.message


def test_missing_key_and_missing_optional_sdk_are_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIProvider()
    monkeypatch.setenv("OPENAI_API_KEY", "simulated-key")

    def unavailable(name: str) -> ModuleType:
        raise ImportError("not installed")

    monkeypatch.setattr("ai_quality_evals.provider.importlib.import_module", unavailable)
    with pytest.raises(ValueError, match=r"install ai-quality-evals\[live\]"):
        OpenAIProvider()


def test_schema_validation_is_required_even_with_structured_outputs(
    generation_request: GenerationRequest,
) -> None:
    value = payload(
        text=json.dumps(
            {
                "outcome": "answer",
                "text": "Unknown claim",
                "citations": ["INVALID ID"],
                "evidence": [],
            }
        )
    )
    result = OpenAIProvider(transport=lambda parameters: value).generate(generation_request)
    assert result.status == "malformed"
    assert result.error is not None and "stable ID" in result.error.message
