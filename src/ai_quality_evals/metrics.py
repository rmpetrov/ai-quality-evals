"""Deterministic structural metrics, with explicit denominators and null applicability.

Citation existence and verbatim evidence validation do not measure semantic faithfulness.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean

from ai_quality_evals.models import (
    CapturedRecord,
    Document,
    EvaluationCase,
    ResponseTrace,
    RetrievalResult,
)

CONTEXT_METRIC_NAMES = frozenset(
    {
        "citation_in_context",
        "context_quote_validity",
        "current_citation_rate",
        "effective_context_coverage",
        "generation_success_rate",
    }
)

METRIC_NAMES = (
    frozenset(
        {
            "recall_at_k",
            "mrr",
            "citation_validity",
            "exact_quote_validity",
            "answerability_accuracy",
            "citation_coverage",
            "evidence_coverage",
        }
    )
    | CONTEXT_METRIC_NAMES
)


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float | None:
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    gold = set(relevant_ids)
    return len(set(ranked_ids[:k]) & gold) / len(gold) if gold else None


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float | None:
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    gold = set(relevant_ids)
    if not gold:
        return None
    return next((1 / rank for rank, doc_id in enumerate(ranked_ids[:k], 1) if doc_id in gold), 0.0)


@dataclass(frozen=True, slots=True)
class CitationCheck:
    document_id: str
    valid: bool
    in_retrieved_context: bool
    source_status: str | None


@dataclass(frozen=True, slots=True)
class QuoteCheck:
    document_id: str
    quote: str
    valid: bool
    reason: str


def validate_citations(
    citations: Sequence[str], documents: Mapping[str, Document], context_ids: Sequence[str] = ()
) -> tuple[CitationCheck, ...]:
    return tuple(
        CitationCheck(
            doc_id,
            doc_id in documents,
            doc_id in context_ids,
            documents[doc_id].status if doc_id in documents else None,
        )
        for doc_id in citations
    )


def validate_quotes(
    response: ResponseTrace, documents: Mapping[str, Document]
) -> tuple[QuoteCheck, ...]:
    checks: list[QuoteCheck] = []
    for item in response.evidence:
        if item.document_id not in documents:
            reason = "unknown_document"
        elif item.document_id not in response.citations:
            reason = "source_not_cited"
        elif not item.quote.strip() or item.quote not in documents[item.document_id].content:
            reason = "not_an_exact_substring"
        else:
            reason = "exact_match"
        checks.append(QuoteCheck(item.document_id, item.quote, reason == "exact_match", reason))
    return tuple(checks)


@dataclass(frozen=True, slots=True)
class ContextEvaluation:
    effective_context_coverage: float | None
    citations: tuple[CitationCheck, ...]
    quotations: tuple[QuoteCheck, ...]
    generation_success: bool


def evaluate_context(case: EvaluationCase, record: CapturedRecord) -> ContextEvaluation:
    supplied = {doc.document_id: doc for doc in record.context.documents}
    response = record.response
    quotations = validate_quotes(response, supplied) if response else ()
    return ContextEvaluation(
        recall_at_k(tuple(supplied), case.relevant_document_ids, record.retrieval.top_k),
        validate_citations(response.citations, supplied, tuple(supplied)) if response else (),
        tuple(
            QuoteCheck(
                check.document_id,
                check.quote,
                check.valid,
                "not_in_generation_context" if check.reason == "unknown_document" else check.reason,
            )
            for check in quotations
        ),
        record.generation.status == "completed",
    )


@dataclass(frozen=True, slots=True)
class CaseResult:
    case: EvaluationCase
    retrieval: RetrievalResult
    response: ResponseTrace | None
    recall_at_k: float | None
    reciprocal_rank: float | None
    citations: tuple[CitationCheck, ...]
    quotations: tuple[QuoteCheck, ...]
    answerability_correct: bool | None
    context_evaluation: ContextEvaluation | None = None


def evaluate_case(
    case: EvaluationCase,
    retrieval: RetrievalResult,
    documents: Mapping[str, Document],
    k: int,
    response: ResponseTrace | None = None,
    capture: CapturedRecord | None = None,
) -> CaseResult:
    if retrieval.query != case.question or retrieval.top_k < k:
        raise ValueError(
            "retrieval query must match the case and stored top_k must cover evaluation k"
        )
    ranked_ids = tuple(hit.document_id for hit in retrieval.hits)
    return CaseResult(
        case,
        retrieval,
        response,
        recall_at_k(ranked_ids, case.relevant_document_ids, k),
        reciprocal_rank(ranked_ids, case.relevant_document_ids, k),
        validate_citations(response.citations, documents, ranked_ids) if response else (),
        validate_quotes(response, documents) if response else (),
        (response.outcome == "answer") == case.answerable if response else None,
        evaluate_context(case, capture) if capture else None,
    )


@dataclass(frozen=True, slots=True)
class Aggregate:
    metrics: dict[str, float | None]
    denominators: dict[str, int]
    case_count: int


def aggregate(results: Sequence[CaseResult]) -> Aggregate:
    recall = [result.recall_at_k for result in results if result.recall_at_k is not None]
    ranks = [result.reciprocal_rank for result in results if result.reciprocal_rank is not None]
    citations = [check.valid for result in results for check in result.citations]
    quotations = [check.valid for result in results for check in result.quotations]
    outcomes = [
        result.answerability_correct
        for result in results
        if result.answerability_correct is not None
    ]
    answered = [
        result.response
        for result in results
        if result.response is not None and result.response.outcome == "answer"
    ]
    samples: dict[str, Sequence[float]] = {
        "recall_at_k": recall,
        "mrr": ranks,
        "citation_validity": citations,
        "exact_quote_validity": quotations,
        "answerability_accuracy": outcomes,
        "citation_coverage": [bool(response.citations) for response in answered],
        "evidence_coverage": [bool(response.evidence) for response in answered],
    }
    contexts = [
        result.context_evaluation for result in results if result.context_evaluation is not None
    ]
    if contexts:
        samples.update(
            {
                "citation_in_context": [check.valid for ctx in contexts for check in ctx.citations],
                "context_quote_validity": [
                    check.valid for ctx in contexts for check in ctx.quotations
                ],
                "current_citation_rate": [
                    check.valid and check.source_status == "current"
                    for result in results
                    if result.context_evaluation is not None
                    for check in result.citations
                ],
                "effective_context_coverage": [
                    ctx.effective_context_coverage
                    for ctx in contexts
                    if ctx.effective_context_coverage is not None
                ],
                "generation_success_rate": [ctx.generation_success for ctx in contexts],
            }
        )
    return Aggregate(
        {name: fmean(values) if values else None for name, values in samples.items()},
        {name: len(values) for name, values in samples.items()},
        len(results),
    )


@dataclass(frozen=True, slots=True)
class ThresholdFailure:
    metric: str
    minimum: float
    actual: float | None
    reason: str


def check_thresholds(
    values: Aggregate, minimums: Mapping[str, float]
) -> tuple[ThresholdFailure, ...]:
    failures: list[ThresholdFailure] = []
    for name, minimum in sorted(minimums.items()):
        if name not in METRIC_NAMES:
            raise ValueError(f"unknown threshold metric: {name}")
        if isinstance(minimum, bool) or not math.isfinite(minimum) or not 0 <= minimum <= 1:
            raise ValueError(f"minimum for {name} must be finite and within [0, 1]")
        actual = values.metrics.get(name)
        if actual is None or actual < minimum:
            failures.append(
                ThresholdFailure(
                    name,
                    minimum,
                    actual,
                    "no_applicable_observations" if actual is None else "below_minimum",
                )
            )
    return tuple(failures)
