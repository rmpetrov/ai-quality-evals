"""Explicit sequential RAG generation, captured separately from evaluation."""

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from ai_quality_evals.capture import (
    append_capture,
    capture_json,
    corpus_hash,
    dataset_hash,
    parse_capture,
)
from ai_quality_evals.models import (
    CapturedRecord,
    EvaluationCase,
    GenerationRequest,
    GenerationSettings,
    ProviderError,
)
from ai_quality_evals.prompt import (
    PROMPT_VERSION,
    build_context,
    build_prompt,
    instructions_for,
    validate_settings,
)
from ai_quality_evals.provider import Provider
from ai_quality_evals.retrieval import BM25Retriever
from ai_quality_evals.runner import DEFAULT_DATA
from ai_quality_evals.validation import (
    ValidationError,
    load_cases,
    load_corpus,
    parse_json,
    parse_response,
)


def select_cases(
    cases: Sequence[EvaluationCase], case_ids: Sequence[str], max_cases: int
) -> tuple[EvaluationCase, ...]:
    if type(max_cases) is not int or not 1 <= max_cases <= 24:
        raise ValueError("max_cases must be an integer within [1, 24]")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate --case selections")
    by_id = {case.case_id: case for case in cases}
    unknown = set(case_ids) - set(by_id)
    if unknown:
        raise ValueError(f"unknown case selections: {sorted(unknown)}")
    if len(case_ids) > max_cases:
        raise ValueError("selected cases exceed max_cases; no requests sent")
    return tuple(by_id[case_id] for case_id in case_ids) if case_ids else tuple(cases[:max_cases])


def require_private_path(path: Path) -> None:
    if "live-captures" not in path.resolve().parts:
        raise ValueError("live artifacts must be inside a live-captures directory (Git ignored)")


def run_live_batch(
    provider: Provider,
    *,
    settings: GenerationSettings,
    captures_path: Path,
    allow_live: bool = False,
    kb: Path = DEFAULT_DATA / "kb",
    cases_path: Path = DEFAULT_DATA / "cases.jsonl",
    case_ids: Sequence[str] = (),
    max_cases: int = 3,
    top_k: int = 3,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[CapturedRecord, ...]:
    if not allow_live:
        raise ValueError("live execution requires explicit --allow-live opt-in")
    validate_settings(settings)
    instructions_for(prompt_version)  # Reject unknown versions before creating files or requests.
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not provider.is_simulated:
        require_private_path(captures_path)
    documents = load_corpus(kb)
    cases = load_cases(cases_path, documents)
    selected = select_cases(cases, case_ids, max_cases)
    corpus_sha256, dataset_sha256 = corpus_hash(kb), dataset_hash(cases_path)
    retriever = BM25Retriever(documents.values())
    captures_path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwrite before inference. Each result is persisted immediately, even on failure.
    with captures_path.open("x", encoding="utf-8"):
        pass
    records: list[CapturedRecord] = []
    for case in selected:
        retrieval = retriever.retrieve(case.question, top_k=top_k)
        context = build_context(retrieval, documents)
        prompt = build_prompt(case.question, context, prompt_version=prompt_version)
        generation = provider.generate(GenerationRequest(settings, prompt))
        response = None
        if generation.status == "completed":
            try:
                if generation.output_text is None:
                    raise ValidationError("completed response has no output text")
                response = parse_response(parse_json(generation.output_text))
            except ValidationError as exc:
                generation = replace(
                    generation,
                    status="malformed",
                    error=ProviderError("malformed_output", str(exc)),
                )
        record = CapturedRecord(
            2,
            case.case_id,
            "synthetic_fixture" if provider.is_simulated else "live_capture",
            "simulated" if provider.is_simulated else "live",
            provider.name,
            case.question,
            retrieval,
            context,
            prompt,
            settings,
            generation,
            response,
            corpus_sha256,
            dataset_sha256,
        )
        # Exercise the same strict boundary as offline replay before writing the trace.
        record = parse_capture(parse_json(capture_json(record)), {case.case_id: case}, documents)
        append_capture(record, captures_path)
        records.append(record)
        if generation.status != "completed":
            break
    return tuple(records)
