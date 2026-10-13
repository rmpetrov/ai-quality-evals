"""Strict JSON boundary validation with actionable file/line diagnostics.

Gold labels are checked here but are never passed into BM25Retriever.
Invalid citation IDs and false quotations are valid prediction *structure*:
they must reach the quality evaluators instead of disappearing as parse errors.
"""

import json
import math
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Literal, cast

from ai_quality_evals.models import (
    CapturedRecord,
    Category,
    Document,
    EvaluationCase,
    EvaluationRecord,
    Evidence,
    Outcome,
    ResponseTrace,
    RetrievalHit,
    RetrievalResult,
)

ID_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
type LiteralStatus = Literal["current", "archived"]


class ValidationError(ValueError):
    """An input violates the versioned evaluation contract."""


def object_fields(value: object, fields: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValidationError("expected a JSON object")
    obj = cast(dict[str, object], value)
    if set(obj) != fields:
        raise ValidationError(
            f"fields differ: missing={sorted(fields - set(obj))}, "
            f"unexpected={sorted(set(obj) - fields)}"
        )
    return obj


def text(value: object, *, identifier: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("expected a nonblank string")
    if identifier and not ID_PATTERN.fullmatch(value):
        raise ValidationError(f"invalid stable ID: {value!r}")
    return value


def array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValidationError("expected a JSON array")
    return cast(list[object], value)


def strings(value: object, *, identifiers: bool = False) -> tuple[str, ...]:
    result = tuple(text(item, identifier=identifiers) for item in array(value))
    if len(result) != len(set(result)):
        raise ValidationError("duplicate list entries")
    return result


def positive_int(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValidationError("expected a positive integer")
    return value


def evidence(value: object) -> tuple[Evidence, ...]:
    result: list[Evidence] = []
    for item in array(value):
        obj = object_fields(item, {"document_id", "quote"})
        result.append(Evidence(text(obj["document_id"], identifier=True), text(obj["quote"])))
    if len(result) != len(set(result)):
        raise ValidationError("duplicate evidence entries")
    return tuple(result)


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValidationError(f"nonfinite JSON constant: {value}")


def parse_json(raw: str) -> object:
    """Reject duplicate keys and nonfinite numbers in provider and recorded JSON alike."""
    try:
        return json.loads(raw, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def read_jsonl(path: Path) -> list[object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValidationError(f"{path}: empty JSONL file")
    result: list[object] = []
    for number, line in enumerate(lines, 1):
        try:
            result.append(parse_json(line))
        except (ValueError, ValidationError) as exc:
            raise ValidationError(f"{path}:{number}: {exc}") from exc
    return result


def load_corpus(directory: Path) -> dict[str, Document]:
    documents: dict[str, Document] = {}
    for path in sorted(directory.glob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8")
            if not raw.startswith("---\n") or "\n---\n" not in raw[4:]:
                raise ValidationError("expected --- delimited metadata header")
            header, body = raw[4:].split("\n---\n", 1)
            metadata: dict[str, object] = {}
            for line in header.splitlines():
                key, separator, value = line.partition(":")
                if not separator or key in metadata:
                    raise ValidationError("invalid or duplicate metadata field")
                metadata[key] = value.strip()
            obj = object_fields(metadata, {"id", "title", "version", "effective_date", "status"})
            document_id = text(obj["id"], identifier=True)
            title = text(obj["title"])
            effective_date = text(obj["effective_date"])
            if date.fromisoformat(effective_date).isoformat() != effective_date:
                raise ValidationError("effective_date must use YYYY-MM-DD")
            status = text(obj["status"])
            if status not in {"current", "archived"}:
                raise ValidationError("status must be current or archived")
            content = text(body).strip()
            if not content.startswith(f"# {title}\n"):
                raise ValidationError("body must start with the metadata title as an H1")
            if document_id in documents:
                raise ValidationError(f"duplicate document ID: {document_id}")
            documents[document_id] = Document(
                document_id,
                title,
                text(obj["version"]),
                effective_date,
                cast("LiteralStatus", status),
                content,
            )
        except ValueError as exc:
            raise ValidationError(f"{path}: {exc}") from exc
    if not documents:
        raise ValidationError(f"{directory}: no Markdown documents")
    return documents


def parse_case(value: object, documents: Mapping[str, Document]) -> EvaluationCase:
    obj = object_fields(
        value,
        {
            "schema_version",
            "case_id",
            "question",
            "category",
            "answerable",
            "relevant_document_ids",
            "reference_facts",
            "reference_evidence",
            "intent",
        },
    )
    if type(obj["schema_version"]) is not int or obj["schema_version"] != 1:
        raise ValidationError("unsupported case schema_version; expected integer 1")
    answerable = obj["answerable"]
    if type(answerable) is not bool:
        raise ValidationError("answerable must be a boolean")
    category = text(obj["category"])
    if category not in {"straightforward", "multi_document", "unanswerable", "distractor"}:
        raise ValidationError(f"unknown category: {category}")
    relevant = strings(obj["relevant_document_ids"], identifiers=True)
    facts = strings(obj["reference_facts"])
    gold_evidence = evidence(obj["reference_evidence"])
    if (category == "unanswerable") != (not answerable):
        raise ValidationError("category and answerability disagree")
    if not facts:
        raise ValidationError("reference_facts must explain the expected result")
    if answerable:
        if not relevant or {item.document_id for item in gold_evidence} != set(relevant):
            raise ValidationError(
                "answerable cases need relevant IDs and evidence for every source"
            )
    elif relevant or gold_evidence:
        raise ValidationError("unanswerable cases must have empty relevant IDs and evidence")
    if category == "multi_document" and len(relevant) < 2:
        raise ValidationError("multi_document cases need at least two sources")
    for document_id in relevant:
        if document_id not in documents:
            raise ValidationError(f"unknown relevant document: {document_id}")
        if documents[document_id].status != "current":
            raise ValidationError(f"gold source is archived: {document_id}")
    for item in gold_evidence:
        if item.quote not in documents[item.document_id].content:
            raise ValidationError(f"gold quote absent from source: {item.document_id}")
    return EvaluationCase(
        text(obj["case_id"], identifier=True),
        text(obj["question"]),
        cast(Category, category),
        answerable,
        relevant,
        facts,
        gold_evidence,
        text(obj["intent"]),
    )


def load_cases(path: Path, documents: Mapping[str, Document]) -> tuple[EvaluationCase, ...]:
    cases: dict[str, EvaluationCase] = {}
    for number, value in enumerate(read_jsonl(path), 1):
        try:
            case = parse_case(value, documents)
            if case.case_id in cases:
                raise ValidationError(f"duplicate case ID: {case.case_id}")
            cases[case.case_id] = case
        except ValidationError as exc:
            raise ValidationError(f"{path}:{number}: {exc}") from exc
    return tuple(cases.values())


def parse_record(
    value: object, cases: Mapping[str, EvaluationCase], documents: Mapping[str, Document]
) -> EvaluationRecord | CapturedRecord:
    if isinstance(value, dict) and value.get("schema_version") == 2:
        from ai_quality_evals.capture import parse_capture

        return parse_capture(value, cases, documents)
    obj = object_fields(
        value, {"schema_version", "case_id", "trace_origin", "retrieval", "response"}
    )
    if type(obj["schema_version"]) is not int or obj["schema_version"] != 1:
        raise ValidationError("unsupported trace schema_version; expected integer 1")
    if obj["trace_origin"] != "synthetic_fixture":
        raise ValidationError("Phase 1 supports only synthetic_fixture replay origin")
    case_id = text(obj["case_id"], identifier=True)
    if case_id not in cases:
        raise ValidationError(f"unknown trace case ID: {case_id}")
    retrieval = parse_retrieval(obj["retrieval"], cases[case_id].question, documents)
    response = parse_response(obj["response"])
    return EvaluationRecord(
        1,
        case_id,
        "synthetic_fixture",
        retrieval,
        response,
    )


def parse_retrieval(
    value: object, question: str, documents: Mapping[str, Document]
) -> RetrievalResult:
    retrieval = object_fields(value, {"query", "top_k", "hits"})
    query = text(retrieval["query"])
    if query != question:
        raise ValidationError("trace query differs from case question")
    top_k = positive_int(retrieval["top_k"])
    hits: list[RetrievalHit] = []
    for item in array(retrieval["hits"]):
        hit = object_fields(item, {"document_id", "score"})
        document_id = text(hit["document_id"], identifier=True)
        if document_id not in documents:
            raise ValidationError(f"unknown retrieved document: {document_id}")
        raw_score = hit["score"]
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValidationError("retrieval score must be finite and nonnegative")
        try:
            score = float(raw_score)
        except OverflowError as exc:
            raise ValidationError("retrieval score exceeds finite float range") from exc
        if not math.isfinite(score) or score < 0:
            raise ValidationError("retrieval score must be finite and nonnegative")
        hits.append(RetrievalHit(document_id, score))
    if len(hits) > top_k or len({item.document_id for item in hits}) != len(hits):
        raise ValidationError("retrieval hits exceed top_k or contain duplicate IDs")
    if any(left.score < right.score for left, right in zip(hits, hits[1:], strict=False)):
        raise ValidationError("retrieval scores must be in nonincreasing rank order")
    return RetrievalResult(query, top_k, tuple(hits))


def parse_response(value: object) -> ResponseTrace:
    response = object_fields(value, {"outcome", "text", "citations", "evidence"})
    outcome = text(response["outcome"])
    if outcome not in {"answer", "abstain"}:
        raise ValidationError("outcome must be answer or abstain")
    return ResponseTrace(
        cast(Outcome, outcome),
        text(response["text"]),
        strings(response["citations"], identifiers=True),
        evidence(response["evidence"]),
    )


def load_records(
    path: Path, cases: tuple[EvaluationCase, ...], documents: Mapping[str, Document]
) -> tuple[EvaluationRecord | CapturedRecord, ...]:
    case_map = {case.case_id: case for case in cases}
    records: dict[str, EvaluationRecord | CapturedRecord] = {}
    for number, value in enumerate(read_jsonl(path), 1):
        try:
            record = parse_record(value, case_map, documents)
            if record.case_id in records:
                raise ValidationError(f"duplicate trace case ID: {record.case_id}")
            records[record.case_id] = record
        except ValueError as exc:
            raise ValidationError(f"{path}:{number}: {exc}") from exc
    # Replays intentionally cover subsets; report evaluated and omitted IDs explicitly.
    return tuple(records.values())
