"""Strict v2 capture loading, provenance verification and incremental durable writes."""

import hashlib
import math
import os
import re
from collections.abc import Mapping
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Literal, cast

from ai_quality_evals.models import (
    CapturedRecord,
    Document,
    EvaluationCase,
    GenerationSettings,
    GenerationStatus,
    PromptSnapshot,
    ProviderError,
    ProviderResult,
    TokenUsage,
)
from ai_quality_evals.prompt import (
    build_context,
    canonical_json,
    instructions_for,
    prompt_hash,
    response_schema,
    validate_settings,
)
from ai_quality_evals.validation import (
    ValidationError,
    array,
    object_fields,
    parse_json,
    parse_response,
    parse_retrieval,
    positive_int,
    text,
)


def corpus_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.md")):
        digest.update(path.name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def dataset_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: object) -> str:
    result = text(value)
    if not re.fullmatch(r"[0-9a-f]{64}", result):
        raise ValidationError("expected a lowercase SHA-256 hash")
    return result


def _nullable_text(value: object) -> str | None:
    return None if value is None else text(value)


def _raw_output(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValidationError("output_text must be a string or null")
    return value


def _nonnegative_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValidationError("expected a nonnegative integer")
    return value


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("expected a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValidationError("number exceeds finite float range") from exc
    if not math.isfinite(result) or result < 0:
        raise ValidationError("expected a finite nonnegative number")
    return result


def _document(value: object) -> Document:
    obj = object_fields(
        value, {"document_id", "title", "version", "effective_date", "status", "content"}
    )
    status = text(obj["status"])
    if status not in {"current", "archived"}:
        raise ValidationError("document status must be current or archived")
    effective = text(obj["effective_date"])
    if date.fromisoformat(effective).isoformat() != effective:
        raise ValidationError("effective_date must use YYYY-MM-DD")
    return Document(
        text(obj["document_id"], identifier=True),
        text(obj["title"]),
        text(obj["version"]),
        effective,
        cast("LiteralStatus", status),
        text(obj["content"]),
    )


type LiteralStatus = Literal["current", "archived"]


def _generation(value: object, *, allow_partial_usage: bool = False) -> ProviderResult:
    obj = object_fields(
        value,
        {
            "status",
            "output_text",
            "response_id",
            "request_id",
            "returned_model",
            "usage",
            "latency_ms",
            "error",
        },
    )
    status = text(obj["status"])
    if status not in {"completed", "incomplete", "refusal", "error", "malformed"}:
        raise ValidationError("unknown generation status")
    usage = None
    if obj["usage"] is not None:
        counts = object_fields(obj["usage"], {"input_tokens", "output_tokens", "total_tokens"})
        usage = TokenUsage(
            *(
                None
                if allow_partial_usage and counts[name] is None
                else _nonnegative_int(counts[name])
                for name in ("input_tokens", "output_tokens", "total_tokens")
            )
        )
    error = None
    if obj["error"] is not None:
        info = object_fields(obj["error"], {"kind", "message", "code", "status_code"})
        error = ProviderError(
            text(info["kind"]),
            text(info["message"]),
            _nullable_text(info["code"]),
            positive_int(info["status_code"]) if info["status_code"] is not None else None,
        )
    if (status == "completed") != (error is None):
        raise ValidationError("generation status and error disagree")
    return ProviderResult(
        cast(GenerationStatus, status),
        _raw_output(obj["output_text"]),
        _nullable_text(obj["response_id"]),
        _nullable_text(obj["request_id"]),
        _nullable_text(obj["returned_model"]),
        usage,
        _finite_number(obj["latency_ms"]),
        error,
    )


def parse_capture(
    value: object,
    cases: Mapping[str, EvaluationCase] | None = None,
    documents: Mapping[str, Document] | None = None,
) -> CapturedRecord:
    """Independently load snapshots; optional originals verify source/question integrity."""
    obj = object_fields(
        value,
        {
            "schema_version",
            "case_id",
            "trace_origin",
            "execution",
            "provider",
            "question",
            "retrieval",
            "context",
            "prompt",
            "settings",
            "generation",
            "response",
            "corpus_sha256",
            "dataset_sha256",
        },
    )
    if type(obj["schema_version"]) is not int or obj["schema_version"] != 2:
        raise ValidationError("unsupported capture schema_version; expected integer 2")
    origin, execution = obj["trace_origin"], obj["execution"]
    if (origin, execution) not in {("synthetic_fixture", "simulated"), ("live_capture", "live")}:
        raise ValidationError(
            "trace origin and execution disagree; simulations are synthetic_fixture"
        )
    provider = text(obj["provider"])
    if origin == "live_capture" and provider not in {"openai", "ollama"}:
        raise ValidationError("live captures require provider openai or ollama")
    case_id = text(obj["case_id"], identifier=True)
    question = text(obj["question"])
    if cases is not None:
        if case_id not in cases:
            raise ValidationError(f"unknown trace case ID: {case_id}")
        if question != cases[case_id].question:
            raise ValidationError("capture question differs from case question")
    ctx = object_fields(obj["context"], {"policy", "documents", "excluded", "text"})
    snapshots = [_document(item) for item in array(ctx["documents"])]
    for item in array(ctx["excluded"]):
        entry = object_fields(item, {"document", "reason"})
        if entry["reason"] != "archived_policy":
            raise ValidationError("unsupported source exclusion reason")
        snapshots.append(_document(entry["document"]))
    snapshot_map = {doc.document_id: doc for doc in snapshots}
    if len(snapshot_map) != len(snapshots):
        raise ValidationError("duplicate context/excluded document ID")
    if documents is not None:
        for doc in snapshots:
            if documents.get(doc.document_id) != doc:
                raise ValidationError(f"context snapshot differs from corpus: {doc.document_id}")
    retrieval = parse_retrieval(obj["retrieval"], question, snapshot_map)
    if {hit.document_id for hit in retrieval.hits} != set(snapshot_map):
        raise ValidationError("context snapshots must cover exactly the raw retrieval hits")
    context = build_context(retrieval, snapshot_map)
    # asdict contains tuples, whereas JSON contains lists: compare canonical serialization.
    if canonical_json(asdict(context)) != canonical_json(ctx):
        raise ValidationError("context differs from exclude_archived_v1 policy or exact text")
    prm = object_fields(
        obj["prompt"], {"version", "sha256", "instructions", "input_text", "response_schema"}
    )
    schema_value = prm["response_schema"]
    if not isinstance(schema_value, dict) or any(not isinstance(key, str) for key in schema_value):
        raise ValidationError("response_schema must be an object")
    schema = cast(dict[str, object], schema_value)
    version, instructions = text(prm["version"]), text(prm["instructions"])
    prompt = PromptSnapshot(
        version, _hash(prm["sha256"]), instructions, text(prm["input_text"]), schema
    )
    if prompt.sha256 != prompt_hash(version, instructions, schema):
        raise ValidationError("prompt hash mismatch")
    try:
        expected_instructions = instructions_for(version)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if instructions != expected_instructions or schema != response_schema():
        raise ValidationError("unsupported prompt version, instructions or response schema")
    expected_input = canonical_json(
        {"question": question, "documents": [asdict(doc) for doc in context.documents]}
    )
    if prompt.input_text != expected_input:
        raise ValidationError("prompt input differs from question and actual generation context")
    cfg = object_fields(
        obj["settings"], {"model", "timeout_seconds", "max_output_tokens", "max_retries"}
    )
    if type(cfg["max_retries"]) is not int or cfg["max_retries"] != 0:
        raise ValidationError("max_retries must be 0")
    settings = GenerationSettings(
        text(cfg["model"]),
        _finite_number(cfg["timeout_seconds"]),
        positive_int(cfg["max_output_tokens"]),
    )
    validate_settings(settings)
    generation = _generation(obj["generation"], allow_partial_usage=provider == "ollama")
    response = None if obj["response"] is None else parse_response(obj["response"])
    if generation.status == "completed":
        if generation.output_text is None or response is None:
            raise ValidationError("completed capture requires raw and structured response")
        if parse_response(parse_json(generation.output_text)) != response:
            raise ValidationError("structured response differs from captured model output")
    elif response is not None:
        raise ValidationError("failed generation must have a null structured response")
    return CapturedRecord(
        2,
        case_id,
        cast("LiteralOrigin", origin),
        cast("LiteralExecution", execution),
        provider,
        question,
        retrieval,
        context,
        prompt,
        settings,
        generation,
        response,
        _hash(obj["corpus_sha256"]),
        _hash(obj["dataset_sha256"]),
    )


type LiteralOrigin = Literal["synthetic_fixture", "live_capture"]
type LiteralExecution = Literal["simulated", "live"]


def verify_input_hashes(record: CapturedRecord, kb: Path, cases_path: Path) -> None:
    if record.corpus_sha256 != corpus_hash(kb) or record.dataset_sha256 != dataset_hash(cases_path):
        raise ValidationError(
            "captured corpus/dataset hash mismatch; replay requires original inputs"
        )


def capture_json(record: CapturedRecord) -> str:
    return canonical_json(asdict(record)) + "\n"


def append_capture(record: CapturedRecord, path: Path) -> None:
    """One complete record per line, flushed after every request; caller creates with mode x."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(capture_json(record))
        stream.flush()
        os.fsync(stream.fileno())
