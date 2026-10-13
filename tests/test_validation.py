import json
from collections import Counter
from pathlib import Path
from typing import cast

import pytest

from ai_quality_evals.models import Document
from ai_quality_evals.validation import (
    ValidationError,
    load_cases,
    load_corpus,
    load_records,
    parse_case,
    parse_record,
    read_jsonl,
)


def test_authored_dataset_integrity(data_dir: Path) -> None:
    corpus = load_corpus(data_dir / "kb")
    cases = load_cases(data_dir / "cases.jsonl", corpus)
    assert len(corpus) == 14
    assert Counter(doc.status for doc in corpus.values()) == {"current": 12, "archived": 2}
    assert len(cases) == 24
    assert Counter(case.category for case in cases) == {
        "straightforward": 9,
        "multi_document": 6,
        "distractor": 5,
        "unanswerable": 4,
    }
    assert sum(case.answerable for case in cases) == 20
    for name in ("good", "bad"):
        assert len(load_records(data_dir / "replays" / f"{name}.jsonl", cases, corpus)) == 6


@pytest.mark.parametrize("field", ["case_id", "question", "intent", "reference_evidence"])
def test_missing_case_fields(
    field: str, case_payload: dict[str, object], documents: dict[str, Document]
) -> None:
    del case_payload[field]
    with pytest.raises(ValidationError, match="missing"):
        parse_case(case_payload, documents)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("case_id", "", "nonblank"),
        ("case_id", "Case X", "stable ID"),
        ("question", "  ", "nonblank"),
        ("schema_version", True, "schema_version"),
        ("schema_version", 2, "schema_version"),
        ("answerable", 1, "boolean"),
        ("category", "unknown", "category"),
        ("category", "unanswerable", "disagree"),
        ("category", "multi_document", "at least two"),
        ("reference_facts", [], "reference_facts"),
        ("relevant_document_ids", "policy-a", "array"),
        ("relevant_document_ids", ["policy-a", "policy-a"], "duplicate"),
        ("reference_evidence", [], "evidence for every source"),
        ("reference_evidence", [{"document_id": "policy-a", "quote": "invented"}], "quote absent"),
    ],
)
def test_bad_case_values(
    field: str,
    value: object,
    message: str,
    case_payload: dict[str, object],
    documents: dict[str, Document],
) -> None:
    case_payload[field] = value
    with pytest.raises(ValidationError, match=message):
        parse_case(case_payload, documents)


@pytest.mark.parametrize(
    ("doc_id", "quote", "message"),
    [
        ("missing", "Any sentence.", "unknown relevant"),
        ("policy-old", "Enrollment closes on October 15.", "gold source is archived"),
    ],
)
def test_unknown_and_archived_gold_sources(
    doc_id: str,
    quote: str,
    message: str,
    case_payload: dict[str, object],
    documents: dict[str, Document],
) -> None:
    case_payload["relevant_document_ids"] = [doc_id]
    case_payload["reference_evidence"] = [{"document_id": doc_id, "quote": quote}]
    with pytest.raises(ValidationError, match=message):
        parse_case(case_payload, documents)


def test_unanswerable_cannot_supply_gold_sources(
    case_payload: dict[str, object], documents: dict[str, Document]
) -> None:
    case_payload.update(category="unanswerable", answerable=False)
    with pytest.raises(ValidationError, match="empty relevant"):
        parse_case(case_payload, documents)
    case_payload.update(relevant_document_ids=[], reference_evidence=[])
    assert not parse_case(case_payload, documents).answerable


def test_unknown_fields_are_rejected(
    case_payload: dict[str, object], documents: dict[str, Document]
) -> None:
    case_payload["answer_hint"] = "gold leakage"
    with pytest.raises(ValidationError, match="unexpected"):
        parse_case(case_payload, documents)


def test_duplicate_case_ids_have_line_diagnostic(
    tmp_path: Path, case_payload: dict[str, object], documents: dict[str, Document]
) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text((json.dumps(case_payload) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValidationError, match=r"cases.jsonl:2: duplicate case ID"):
        load_cases(path, documents)


@pytest.mark.parametrize("raw", ["", "\n", "[]\n\n", '{"a":1,"a":2}\n', '{"a":NaN}\n', "{"])
def test_invalid_jsonl(tmp_path: Path, raw: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValidationError):
        read_jsonl(path)


def _markdown(
    document_id: str = "policy-a", effective_date: str = "2026-01-01", status: str = "current"
) -> str:
    return (
        f"---\nid: {document_id}\ntitle: Test policy\nversion: 1.0\n"
        f"effective_date: {effective_date}\nstatus: {status}\n---\n# Test policy\n\nPolicy text.\n"
    )


def test_duplicate_document_ids(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text(_markdown(), encoding="utf-8")
    (tmp_path / "b.md").write_text(_markdown(), encoding="utf-8")
    with pytest.raises(ValidationError, match="duplicate document ID"):
        load_corpus(tmp_path)


@pytest.mark.parametrize(
    "raw",
    [
        "# No metadata\nText",
        _markdown().replace("id: policy-a\n", ""),
        _markdown().replace("version: 1.0", "id: other"),
        _markdown(effective_date="2026-02-30"),
        _markdown(effective_date="20260101"),
        _markdown(status="obsolete"),
        _markdown().replace("# Test policy", "# Wrong title"),
    ],
)
def test_invalid_document_metadata(tmp_path: Path, raw: str) -> None:
    (tmp_path / "bad.md").write_text(raw, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_corpus(tmp_path)


def test_empty_corpus(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="no Markdown"):
        load_corpus(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("case_id", "unknown", "unknown trace"),
        ("schema_version", True, "schema_version"),
        ("trace_origin", "live_model", "only synthetic_fixture"),
    ],
)
def test_bad_trace_identity(
    field: str,
    value: object,
    message: str,
    trace_payload: dict[str, object],
    case_payload: dict[str, object],
    documents: dict[str, Document],
) -> None:
    trace_payload[field] = value
    case = parse_case(case_payload, documents)
    with pytest.raises(ValidationError, match=message):
        parse_record(trace_payload, {case.case_id: case}, documents)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("query", "a different query", "differs"),
        ("top_k", 0, "positive integer"),
        ("top_k", True, "positive integer"),
        ("hits", [{"document_id": "ghost", "score": 1.0}], "unknown retrieved"),
        ("hits", [{"document_id": "policy-a", "score": -1}], "finite and nonnegative"),
        ("hits", [{"document_id": "policy-a", "score": True}], "finite and nonnegative"),
        ("hits", [{"document_id": "policy-a", "score": float("inf")}], "finite and nonnegative"),
        ("hits", [{"document_id": "policy-a", "score": 10**400}], "finite float range"),
        ("hits", [{"document_id": "policy-a", "score": 1}] * 2, "duplicate IDs"),
        (
            "hits",
            [{"document_id": "policy-a", "score": 1}, {"document_id": "policy-b", "score": 2}],
            "rank order",
        ),
    ],
)
def test_bad_retrieval_structure(
    field: str,
    value: object,
    message: str,
    trace_payload: dict[str, object],
    case_payload: dict[str, object],
    documents: dict[str, Document],
) -> None:
    cast(dict[str, object], trace_payload["retrieval"])[field] = value
    case = parse_case(case_payload, documents)
    with pytest.raises(ValidationError, match=message):
        parse_record(trace_payload, {case.case_id: case}, documents)


def test_hits_cannot_exceed_declared_k(
    trace_payload: dict[str, object],
    case_payload: dict[str, object],
    documents: dict[str, Document],
) -> None:
    retrieval = cast(dict[str, object], trace_payload["retrieval"])
    retrieval.update(
        top_k=1,
        hits=[{"document_id": "policy-a", "score": 2}, {"document_id": "policy-b", "score": 1}],
    )
    case = parse_case(case_payload, documents)
    with pytest.raises(ValidationError, match="exceed top_k"):
        parse_record(trace_payload, {case.case_id: case}, documents)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("outcome", "guess", "outcome"),
        ("text", "", "nonblank"),
        ("citations", ["policy-a", "policy-a"], "duplicate"),
        ("evidence", [{"document_id": "policy-a", "quote": ""}], "nonblank"),
    ],
)
def test_bad_response_structure(
    field: str,
    value: object,
    message: str,
    trace_payload: dict[str, object],
    case_payload: dict[str, object],
    documents: dict[str, Document],
) -> None:
    cast(dict[str, object], trace_payload["response"])[field] = value
    case = parse_case(case_payload, documents)
    with pytest.raises(ValidationError, match=message):
        parse_record(trace_payload, {case.case_id: case}, documents)


def test_quality_errors_are_not_swallowed_as_schema_errors(
    trace_payload: dict[str, object],
    case_payload: dict[str, object],
    documents: dict[str, Document],
) -> None:
    response = cast(dict[str, object], trace_payload["response"])
    response.update(
        citations=["missing-document"],
        evidence=[{"document_id": "missing-document", "quote": "A false quotation."}],
    )
    case = parse_case(case_payload, documents)
    record = parse_record(trace_payload, {case.case_id: case}, documents)
    assert record.response.citations == ("missing-document",)


def test_duplicate_trace_ids(
    tmp_path: Path,
    trace_payload: dict[str, object],
    case_payload: dict[str, object],
    documents: dict[str, Document],
) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text((json.dumps(trace_payload) + "\n") * 2, encoding="utf-8")
    case = parse_case(case_payload, documents)
    with pytest.raises(ValidationError, match=r"traces.jsonl:2: duplicate trace"):
        load_records(path, (case,), documents)
