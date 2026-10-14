"""Optional native Ollama adapter: fixed loopback HTTP, no SDK, downloads or retries."""

import http.client
import re
from time import perf_counter
from typing import Protocol, cast

from ai_quality_evals.models import GenerationRequest, ProviderError, ProviderResult, TokenUsage
from ai_quality_evals.prompt import canonical_json, validate_settings
from ai_quality_evals.provider import TransportError, redact
from ai_quality_evals.validation import ValidationError, parse_json, parse_response


class OllamaTransport(Protocol):
    def __call__(
        self, path: str, parameters: dict[str, object], timeout_seconds: float
    ) -> object: ...


def validate_local_model(model: str) -> None:
    # Deliberately accept only unqualified local model names, optionally namespace/tag.
    # Remote registries, URLs and explicit cloud model names are outside this adapter.
    if (
        not re.fullmatch(
            r"(?:[A-Za-z0-9_-]+/)?[A-Za-z0-9][A-Za-z0-9_.-]*"
            r"(?::[A-Za-z0-9][A-Za-z0-9_.-]*)?",
            model,
        )
        or "cloud" in model.lower()
    ):
        raise ValueError("Ollama requires a local model ID, for example qwen2.5:3b")


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValidationError("Ollama response must be a JSON object")
    return cast(dict[str, object], value)


def _require_local(value: dict[str, object]) -> None:
    if any(value.get(name) not in (None, "") for name in ("remote_host", "remote_model")):
        raise TransportError(
            ProviderError("local_only", "Cloud-backed Ollama models are forbidden")
        )


def _usage(value: dict[str, object]) -> TokenUsage | None:
    counts: list[int | None] = []
    for name in ("prompt_eval_count", "eval_count"):
        count = value.get(name)
        if count is not None and (type(count) is not int or count < 0):
            raise ValidationError("Ollama usage must be nonnegative integer counts or null")
        counts.append(count)
    input_tokens, output_tokens = counts
    if input_tokens is None and output_tokens is None:
        return None
    total = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    return TokenUsage(input_tokens, output_tokens, total)


class _HTTPTransport:
    def __call__(self, path: str, parameters: dict[str, object], timeout_seconds: float) -> object:
        if path not in {"/api/show", "/api/chat"}:
            raise ValueError("unsupported local Ollama endpoint")
        # HTTPConnection uses neither proxy environment variables nor redirect handlers.
        # A numeric, fixed host avoids DNS and ignores OLLAMA_HOST/OPENAI_BASE_URL.
        connection = http.client.HTTPConnection("127.0.0.1", 11434, timeout=timeout_seconds)
        try:
            connection.request(
                "POST",
                path,
                canonical_json(parameters).encode("utf-8"),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                message = f"Local Ollama HTTP error {response.status}; no retry or fallback"
                if 400 <= response.status < 600:
                    try:
                        detail = _object(parse_json(response.read().decode("utf-8"))).get("error")
                        if isinstance(detail, str) and detail.strip():
                            message += ": " + redact(detail)
                    except (ValidationError, UnicodeDecodeError):
                        pass
                raise TransportError(
                    ProviderError("provider_error", message, status_code=response.status)
                )
            try:
                return parse_json(response.read().decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValidationError("Ollama response must be UTF-8 JSON") from exc
        except TimeoutError:
            raise TransportError(
                ProviderError("timeout", "Local Ollama request timed out")
            ) from None
        except (OSError, http.client.HTTPException):
            raise TransportError(
                ProviderError("connection", "Local Ollama server unavailable at 127.0.0.1:11434")
            ) from None
        finally:
            connection.close()


class OllamaProvider:
    name = "ollama"

    def __init__(self, *, transport: OllamaTransport | None = None):
        self.is_simulated = transport is not None
        self._transport: OllamaTransport = _HTTPTransport() if transport is None else transport

    def close(self) -> None:
        # Each native HTTP connection is closed in the transport's finally block.
        pass

    def generate(self, request: GenerationRequest) -> ProviderResult:
        validate_settings(request.settings)
        validate_local_model(request.settings.model)
        parameters: dict[str, object] = {
            "model": request.settings.model,
            "messages": [
                {"role": "system", "content": request.prompt.instructions},
                {"role": "user", "content": request.prompt.input_text},
            ],
            "format": request.prompt.response_schema,
            "stream": False,
            "options": {"num_predict": request.settings.max_output_tokens},
        }
        start = perf_counter()
        output_text: str | None = None
        model: str | None = None
        usage: TokenUsage | None = None
        try:
            # Local metadata only: verify weights exist and reject remote aliases before chat.
            # No /api/pull and no inference request on a failed preflight.
            info = _object(
                self._transport(
                    "/api/show", {"model": request.settings.model}, request.settings.timeout_seconds
                )
            )
            _require_local(info)
            architecture = _object(info.get("model_info")).get("general.architecture")
            if not isinstance(architecture, str) or not architecture.strip():
                raise TransportError(
                    ProviderError(
                        "local_only", "Cannot verify installed local Ollama model weights"
                    )
                )
            obj = _object(
                self._transport("/api/chat", parameters, request.settings.timeout_seconds)
            )
            returned_model = obj.get("model")
            if isinstance(returned_model, str) and returned_model.strip():
                model = redact(returned_model)
            message = _object(obj.get("message"))
            content = message.get("content")
            if not isinstance(content, str) or message.get("role") != "assistant":
                raise ValidationError("Ollama response requires assistant message content")
            output_text = redact(content)
            usage = _usage(obj)
            _require_local(obj)
            if model is not None:
                try:
                    validate_local_model(model)
                except ValueError:
                    raise TransportError(
                        ProviderError("local_only", "Unexpected remote Ollama model metadata")
                    ) from None
            if obj.get("done") is not True or obj.get("done_reason") == "length":
                return ProviderResult(
                    "incomplete",
                    output_text,
                    None,
                    None,
                    model,
                    usage,
                    (perf_counter() - start) * 1000,
                    ProviderError("incomplete", "Local Ollama response incomplete"),
                )
            parse_response(parse_json(output_text))
            return ProviderResult(
                "completed",
                output_text,
                None,
                None,
                model,
                usage,
                (perf_counter() - start) * 1000,
            )
        except TransportError as exc:
            return ProviderResult(
                "error",
                output_text,
                None,
                None,
                model,
                usage,
                (perf_counter() - start) * 1000,
                ProviderError(
                    exc.error.kind, redact(exc.error.message), exc.error.code, exc.error.status_code
                ),
            )
        except ValidationError as exc:
            return ProviderResult(
                "malformed",
                output_text,
                None,
                None,
                model,
                usage,
                (perf_counter() - start) * 1000,
                ProviderError("malformed_output", redact(str(exc))),
            )
