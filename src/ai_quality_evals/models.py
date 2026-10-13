"""Immutable records shared by the loader, retriever, and evaluators."""

from dataclasses import dataclass
from typing import Literal

type Category = Literal["straightforward", "multi_document", "unanswerable", "distractor"]
type Outcome = Literal["answer", "abstain"]


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    title: str
    version: str
    effective_date: str
    status: Literal["current", "archived"]
    content: str


@dataclass(frozen=True, slots=True)
class Evidence:
    document_id: str
    quote: str


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    case_id: str
    question: str
    category: Category
    answerable: bool
    relevant_document_ids: tuple[str, ...]
    reference_facts: tuple[str, ...]
    reference_evidence: tuple[Evidence, ...]
    intent: str


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    document_id: str
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    top_k: int
    hits: tuple[RetrievalHit, ...]


@dataclass(frozen=True, slots=True)
class ResponseTrace:
    outcome: Outcome
    text: str
    citations: tuple[str, ...]
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    schema_version: int
    case_id: str
    trace_origin: Literal["synthetic_fixture"]
    retrieval: RetrievalResult
    response: ResponseTrace
