"""Optional SDK contract check using HTTPX2 MockTransport, sockets disabled, no paid calls.

Run in an environment with the live SDK installed, including the separate sdk-contract CI job.
Every response below is simulated, never a live capture or a model benchmark.
"""

import importlib
import json
import os
from typing import Protocol, cast
from unittest.mock import patch

from ai_quality_evals.models import GenerationRequest, GenerationSettings
from ai_quality_evals.prompt import build_context, build_prompt
from ai_quality_evals.provider import OpenAIProvider, _SDKTransport
from ai_quality_evals.retrieval import BM25Retriever
from ai_quality_evals.runner import DEFAULT_DATA
from ai_quality_evals.validation import load_corpus


class RequestContent(Protocol):
    content: bytes


def main() -> None:
    http = importlib.import_module("httpx2")
    sdk = importlib.import_module("openai")
    documents = load_corpus(DEFAULT_DATA / "kb")
    question = "How many hours are required?"
    context = build_context(BM25Retriever(documents.values()).retrieve(question), documents)
    generation_request = GenerationRequest(
        GenerationSettings("simulated-model", 5, 300), build_prompt(question, context)
    )
    cases = [
        (200, "completed"),
        (401, "error"),
        (429, "error"),
        (500, "error"),
        (0, "error"),
        (200, "incomplete"),
        (200, "refusal"),
        (200, "malformed"),
    ]
    for code, expected in cases:
        calls: list[object] = []

        def handler(
            request: object, scenario: tuple[int, str, list[object]] = (code, expected, calls)
        ) -> object:
            code, expected, calls = scenario
            calls.append(request)
            sent = cast(dict[str, object], json.loads(cast(RequestContent, request).content))
            assert sent["model"] == "simulated-model"
            assert sent["store"] is False
            assert sent["max_output_tokens"] == 300
            assert sent["text"] == {
                "format": {
                    "type": "json_schema",
                    "name": "benefits_response",
                    "strict": True,
                    "schema": generation_request.prompt.response_schema,
                }
            }
            if code == 0:
                raise http.ReadTimeout("Simulated timeout", request=request)
            if code != 200:
                return http.Response(
                    code,
                    request=request,
                    headers={"x-request-id": "req-simulated"},
                    json={
                        "error": {
                            "message": "Simulated API failure",
                            "type": "test_error",
                            "code": "test_code",
                        }
                    },
                )
            body: dict[str, object] = {
                "id": "resp-simulated",
                "object": "response",
                "created_at": 1,
                "model": "simulated-model-snapshot",
                "status": "completed",
                "output": [
                    {
                        "id": "msg-simulated",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "annotations": [],
                                "text": (
                                    '{"outcome":"abstain","text":"Missing records",'
                                    '"citations":[],"evidence":[]}'
                                ),
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "total_tokens": 120,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
            if expected == "incomplete":
                body["status"] = "incomplete"
                body["incomplete_details"] = {"reason": "max_output_tokens"}
            if expected == "refusal":
                body["output"] = [
                    {
                        "id": "msg-simulated",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "refusal", "refusal": "Simulated refusal"}],
                    }
                ]
            if expected == "malformed":
                body["output"] = []
            return http.Response(
                200, request=request, headers={"x-request-id": "req-simulated"}, json=body
            )

        # Replace any configured key for this process, never reading or sending it.
        with patch.dict(os.environ, {"OPENAI_API_KEY": "simulated-contract-key"}):
            client = http.Client(transport=http.MockTransport(handler), trust_env=False)
            transport = _SDKTransport(http_client=client)
            provider = OpenAIProvider(transport=transport)
            try:
                with patch(
                    "socket.socket",
                    side_effect=AssertionError("SDK contract check opened a socket"),
                ):
                    result = provider.generate(generation_request)
            finally:
                transport.close()
        assert provider.is_simulated
        assert result.status == expected, result
        assert len(calls) == 1, "SDK retried despite max_retries=0"
        if code == 0:
            assert result.error is not None and result.error.kind == "timeout"
        if code == 401:
            assert result.error is not None and result.error.kind == "authentication"
        if code == 429:
            assert result.error is not None and result.error.kind == "rate_limit"
        if code == 200:
            assert result.response_id == "resp-simulated"
            assert result.returned_model == "simulated-model-snapshot"
            assert result.usage is not None and result.usage.input_tokens == 100
        if code != 0:
            assert result.request_id == "req-simulated"
    print(
        f"OpenAI SDK {sdk.__version__}: {len(cases)} offline transport scenarios passed; "
        "zero network/paid calls."
    )


if __name__ == "__main__":
    main()
