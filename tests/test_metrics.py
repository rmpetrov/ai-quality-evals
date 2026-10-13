from dataclasses import replace

import pytest

from ai_quality_evals.metrics import (
    aggregate,
    check_thresholds,
    evaluate_case,
    recall_at_k,
    reciprocal_rank,
    validate_citations,
    validate_quotes,
)
from ai_quality_evals.models import Document, Evidence, ResponseTrace, RetrievalHit, RetrievalResult
from ai_quality_evals.validation import parse_case


@pytest.mark.parametrize(
    ("ranked", "gold", "k", "recall", "rr"),
    [
        (["a", "b", "c"], ["b", "c"], 2, 0.5, 0.5),
        (["x", "y", "a"], ["a"], 3, 1.0, 1 / 3),
        (["x", "y", "a"], ["a"], 2, 0.0, 0.0),
        (["a", "a"], ["a", "b"], 2, 0.5, 1.0),
        (["a"], ["a", "a"], 3, 1.0, 1.0),
        ([], ["a"], 3, 0.0, 0.0),
        (["a"], [], 3, None, None),
    ],
)
def test_retrieval_metrics_have_independent_expected_values(
    ranked: list[str], gold: list[str], k: int, recall: float | None, rr: float | None
) -> None:
    assert recall_at_k(ranked, gold, k) == recall
    assert reciprocal_rank(ranked, gold, k) == rr


@pytest.mark.parametrize("k", [0, -1, True])
def test_metric_k_validation(k: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        recall_at_k([], [], k)
    with pytest.raises(ValueError, match="positive integer"):
        reciprocal_rank([], [], k)


def test_citation_existence_is_distinct_from_context_and_currency(
    documents: dict[str, Document],
) -> None:
    checks = validate_citations(["policy-a", "missing", "policy-old"], documents, ["policy-a"])
    assert [check.valid for check in checks] == [True, False, True]
    assert [check.in_retrieved_context for check in checks] == [True, False, False]
    assert [check.source_status for check in checks] == ["current", None, "archived"]
    assert validate_citations([], documents) == ()


@pytest.mark.parametrize(
    ("doc_id", "quote", "citations", "valid", "reason"),
    [
        ("policy-a", "Enrollment closes on November 20.", ("policy-a",), True, "exact_match"),
        ("policy-a", "closes on November 20", ("policy-a",), True, "exact_match"),
        (
            "policy-a",
            "enrollment closes on November 20.",
            ("policy-a",),
            False,
            "not_an_exact_substring",
        ),
        (
            "policy-a",
            "Upload a receipt within 60 days.",
            ("policy-a",),
            False,
            "not_an_exact_substring",
        ),
        ("policy-a", "", ("policy-a",), False, "not_an_exact_substring"),
        ("missing", "Any quote.", ("missing",), False, "unknown_document"),
        ("policy-a", "Enrollment closes on November 20.", (), False, "source_not_cited"),
        ("policy-old", "Enrollment closes on October 15.", ("policy-old",), True, "exact_match"),
    ],
)
def test_exact_evidence_quotation_validation(
    doc_id: str,
    quote: str,
    citations: tuple[str, ...],
    valid: bool,
    reason: str,
    documents: dict[str, Document],
) -> None:
    response = ResponseTrace("answer", "An answer.", citations, (Evidence(doc_id, quote),))
    check = validate_quotes(response, documents)[0]
    assert check.valid is valid
    assert check.reason == reason


def test_structural_checks_do_not_claim_semantic_faithfulness(
    documents: dict[str, Document],
) -> None:
    response = ResponseTrace(
        "answer",
        "Enrollment never closes and everyone receives a million dollars.",
        ("policy-a",),
        (Evidence("policy-a", "Enrollment closes on November 20."),),
    )
    assert validate_citations(response.citations, documents)[0].valid
    assert validate_quotes(response, documents)[0].valid


def test_aggregate_macro_retrieval_and_micro_structural_denominators(
    case_payload: dict[str, object], documents: dict[str, Document]
) -> None:
    case = parse_case(case_payload, documents)
    ranking = RetrievalResult(
        case.question, 3, (RetrievalHit("policy-b", 2), RetrievalHit("policy-a", 1))
    )
    good_response = ResponseTrace(
        "answer",
        "November 20.",
        ("policy-a", "policy-b"),
        (Evidence("policy-a", "Enrollment closes on November 20."),),
    )
    bad_response = ResponseTrace(
        "abstain", "Cannot answer.", ("missing",), (Evidence("missing", "False."),)
    )
    results = [
        evaluate_case(case, ranking, documents, 3, good_response),
        evaluate_case(
            replace(case, case_id="case-other"),
            RetrievalResult(case.question, 3, ()),
            documents,
            3,
            bad_response,
        ),
    ]
    summary = aggregate(results)
    assert summary.case_count == 2
    assert summary.metrics == {
        "recall_at_k": 0.5,
        "mrr": 0.25,
        "citation_validity": 2 / 3,
        "exact_quote_validity": 0.5,
        "answerability_accuracy": 0.5,
        "citation_coverage": 1.0,
        "evidence_coverage": 1.0,
    }
    assert summary.denominators == {
        "recall_at_k": 2,
        "mrr": 2,
        "citation_validity": 3,
        "exact_quote_validity": 2,
        "answerability_accuracy": 2,
        "citation_coverage": 1,
        "evidence_coverage": 1,
    }
    assert check_thresholds(summary, {"recall_at_k": 0.5}) == ()
    failure = check_thresholds(summary, {"recall_at_k": 0.6})[0]
    assert failure.actual == 0.5
    assert failure.minimum == 0.6


def test_empty_denominators_are_null_and_cannot_pass_a_requested_gate() -> None:
    summary = aggregate([])
    assert all(value is None for value in summary.metrics.values())
    assert check_thresholds(summary, {}) == ()
    failure = check_thresholds(summary, {"citation_validity": 0.0})[0]
    assert failure.actual is None
    assert failure.reason == "no_applicable_observations"


@pytest.mark.parametrize(
    "minimums",
    [
        {"unknown": 0.5},
        {"mrr": -0.1},
        {"mrr": 1.1},
        {"mrr": float("nan")},
        {"mrr": float("inf")},
        {"mrr": True},
    ],
)
def test_bad_thresholds(minimums: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        check_thresholds(aggregate([]), minimums)


def test_retrieval_mode_does_not_fabricate_response_metrics(
    case_payload: dict[str, object], documents: dict[str, Document]
) -> None:
    case = parse_case(case_payload, documents)
    result = evaluate_case(case, RetrievalResult(case.question, 3, ()), documents, 3)
    assert result.recall_at_k == 0.0
    assert result.response is None
    assert result.answerability_correct is None
    assert aggregate([result]).metrics["citation_validity"] is None


def test_replay_must_cover_requested_k(
    case_payload: dict[str, object], documents: dict[str, Document]
) -> None:
    case = parse_case(case_payload, documents)
    with pytest.raises(ValueError, match="stored top_k"):
        evaluate_case(case, RetrievalResult(case.question, 1, ()), documents, 3)
