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


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    model: str
    timeout_seconds: float = 30.0
    max_output_tokens: int = 800
    max_retries: Literal[0] = 0


@dataclass(frozen=True, slots=True)
class ExcludedSource:
    document: Document
    reason: Literal["archived_policy"] = "archived_policy"


@dataclass(frozen=True, slots=True)
class GenerationContext:
    policy: Literal["exclude_archived_v1"]
    documents: tuple[Document, ...]
    excluded: tuple[ExcludedSource, ...]
    text: str


@dataclass(frozen=True, slots=True)
class PromptSnapshot:
    version: str
    sha256: str
    instructions: str
    input_text: str
    response_schema: dict[str, object]


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    settings: GenerationSettings
    prompt: PromptSnapshot


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Reported counts; Ollama may omit individual counts (validated per provider)."""

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


type GenerationStatus = Literal["completed", "incomplete", "refusal", "error", "malformed"]


@dataclass(frozen=True, slots=True)
class ProviderError:
    kind: str
    message: str
    code: str | None = None
    status_code: int | None = None


@dataclass(frozen=True, slots=True)
class ProviderResult:
    status: GenerationStatus
    output_text: str | None
    response_id: str | None
    request_id: str | None
    returned_model: str | None
    usage: TokenUsage | None
    latency_ms: float
    error: ProviderError | None = None


@dataclass(frozen=True, slots=True)
class CapturedRecord:
    schema_version: Literal[2]
    case_id: str
    trace_origin: Literal["synthetic_fixture", "live_capture"]
    execution: Literal["simulated", "live"]
    provider: str
    question: str
    retrieval: RetrievalResult
    context: GenerationContext
    prompt: PromptSnapshot
    settings: GenerationSettings
    generation: ProviderResult
    response: ResponseTrace | None
    corpus_sha256: str
    dataset_sha256: str
