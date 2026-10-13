"""Minimal provider contract and lazy official OpenAI Responses API adapter.

No SDK import, client creation, retries, or network activity occurs at module import.
"""

import importlib
import os
import re
from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Protocol, cast

from ai_quality_evals.models import (
    GenerationRequest,
    ProviderError,
    ProviderResult,
    TokenUsage,
)
from ai_quality_evals.prompt import validate_settings
from ai_quality_evals.validation import ValidationError, parse_json, parse_response


class Provider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def is_simulated(self) -> bool: ...

    def generate(self, request: GenerationRequest) -> ProviderResult: ...


type Transport = Callable[[dict[str, object]], object]


class TransportError(Exception):
    def __init__(self, error: ProviderError, request_id: str | None = None):
        super().__init__(error.message)
        self.error = error
        self.request_id = request_id


def redact(value: str) -> str:
    secret = os.environ.get("OPENAI_API_KEY")
    if secret:
        value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", value)
    return re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", value)


def _optional_text(value: object) -> str | None:
    return redact(value) if isinstance(value, str) and value.strip() else None


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValidationError("provider response must be a JSON object")
    return cast(dict[str, object], value)


def _usage(value: object) -> TokenUsage | None:
    if value is None:
        return None
    usage = _mapping(value)
    counts = [usage.get(name) for name in ("input_tokens", "output_tokens", "total_tokens")]
    if any(type(count) is not int or count < 0 for count in counts):
        raise ValidationError("provider usage must contain nonnegative integer token counts")
    return TokenUsage(*(cast(int, count) for count in counts))


class _SDKTransport:
    def __init__(self, *, http_client: object | None = None) -> None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key or not key.strip():
            raise ValueError("OPENAI_API_KEY is required for live execution")
        try:
            self._sdk = importlib.import_module("openai")
        except ImportError as exc:
            raise ValueError("live dependency missing; install ai-quality-evals[live]") from exc
        # The SDK is the only untyped external boundary. Parameters and parsed output are typed.
        self._client = self._sdk.OpenAI(api_key=key, max_retries=0, http_client=http_client)

    def __call__(self, parameters: dict[str, object]) -> object:
        try:
            response = self._client.responses.create(**parameters)
            payload = cast(dict[str, object], response.model_dump(mode="json"))
            payload["request_id"] = response._request_id
            return payload
        except self._sdk.APIError as exc:
            if isinstance(exc, self._sdk.AuthenticationError):
                kind = "authentication"
            elif isinstance(exc, self._sdk.RateLimitError):
                kind = "rate_limit"
            elif isinstance(exc, self._sdk.APITimeoutError):
                kind = "timeout"
            elif isinstance(exc, self._sdk.APIConnectionError):
                kind = "connection"
            else:
                kind = "provider_error"
            status = getattr(exc, "status_code", None)
            # Keep the provider's error code, safe message, HTTP status and request ID.
            # Never serialize exception repr, request headers, or the entire HTTP body.
            raise TransportError(
                ProviderError(
                    kind,
                    redact(str(exc)),
                    _optional_text(getattr(exc, "code", None)),
                    status if type(status) is int else None,
                ),
                _optional_text(getattr(exc, "request_id", None)),
            ) from None

    def close(self) -> None:
        self._client.close()


class OpenAIProvider:
    name = "openai"

    def __init__(self, *, transport: Transport | None = None):
        self.is_simulated = transport is not None
        self._sdk_transport = _SDKTransport() if transport is None else None
        self._transport: Transport = transport or cast(_SDKTransport, self._sdk_transport)

    def close(self) -> None:
        if self._sdk_transport is not None:
            self._sdk_transport.close()

    def generate(self, request: GenerationRequest) -> ProviderResult:
        validate_settings(request.settings)
        parameters: dict[str, object] = {
            "model": request.settings.model,
            "instructions": request.prompt.instructions,
            "input": request.prompt.input_text,
            "max_output_tokens": request.settings.max_output_tokens,
            "timeout": request.settings.timeout_seconds,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "benefits_response",
                    "strict": True,
                    "schema": request.prompt.response_schema,
                }
            },
        }
        start = perf_counter()
        try:
            payload = self._transport(parameters)
        except TransportError as exc:
            return ProviderResult(
                "error",
                None,
                None,
                _optional_text(exc.request_id),
                None,
                None,
                (perf_counter() - start) * 1000,
                ProviderError(
                    exc.error.kind,
                    redact(exc.error.message),
                    _optional_text(exc.error.code),
                    exc.error.status_code,
                ),
            )
        latency = (perf_counter() - start) * 1000
        response_id: str | None = None
        request_id: str | None = None
        model: str | None = None
        usage: TokenUsage | None = None
        output_text: str | None = None
        try:
            obj = _mapping(payload)
            response_id = _optional_text(obj.get("id"))
            request_id = _optional_text(obj.get("request_id"))
            model = _optional_text(obj.get("model"))
            usage = _usage(obj.get("usage"))
            # output_text is an SDK convenience property; model_dump contains output items.
            parts: list[str] = []
            refusal: str | None = None
            output = obj.get("output", [])
            if not isinstance(output, list):
                raise ValidationError("provider output must be an array")
            for item in output:
                message = _mapping(item)
                if message.get("type") != "message":
                    continue
                content = message.get("content")
                if not isinstance(content, list):
                    raise ValidationError("provider message content must be an array")
                for part in content:
                    entry = _mapping(part)
                    if entry.get("type") == "refusal":
                        refusal = _optional_text(entry.get("refusal")) or "Provider refused"
                    elif entry.get("type") == "output_text":
                        part_text = entry.get("text")
                        if not isinstance(part_text, str):
                            raise ValidationError("provider output text must be a string")
                        parts.append(redact(part_text))
            output_text = "".join(parts) or None
            status = obj.get("status")
            if status == "incomplete":
                details = _mapping(obj.get("incomplete_details") or {})
                return ProviderResult(
                    "incomplete",
                    output_text,
                    response_id,
                    request_id,
                    model,
                    usage,
                    latency,
                    ProviderError(
                        "incomplete",
                        "Provider response incomplete",
                        _optional_text(details.get("reason")),
                    ),
                )
            if status != "completed":
                error = _mapping(obj.get("error") or {})
                return ProviderResult(
                    "error",
                    output_text,
                    response_id,
                    request_id,
                    model,
                    usage,
                    latency,
                    ProviderError(
                        "provider_status",
                        _optional_text(error.get("message"))
                        or redact(f"Unexpected provider status: {status!s}"),
                        _optional_text(error.get("code")),
                    ),
                )
            if refusal is not None:
                return ProviderResult(
                    "refusal",
                    output_text,
                    response_id,
                    request_id,
                    model,
                    usage,
                    latency,
                    ProviderError("refusal", refusal),
                )
            if output_text is None:
                raise ValidationError("completed response has no output text")
            parse_response(parse_json(output_text))
            return ProviderResult(
                "completed",
                output_text,
                response_id,
                request_id,
                model,
                usage,
                latency,
            )
        except ValidationError as exc:
            return ProviderResult(
                "malformed",
                output_text,
                response_id,
                request_id,
                model,
                usage,
                latency,
                ProviderError("malformed_output", redact(str(exc))),
            )
