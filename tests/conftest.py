from pathlib import Path

import pytest

from ai_quality_evals.models import Document
from ai_quality_evals.runner import DEFAULT_DATA


@pytest.fixture
def data_dir() -> Path:
    return DEFAULT_DATA


@pytest.fixture
def documents() -> dict[str, Document]:
    return {
        "policy-a": Document(
            "policy-a",
            "Enrollment",
            "1.0",
            "2026-01-01",
            "current",
            "# Enrollment\n\nEnrollment closes on November 20. Upload a receipt within 90 days.",
        ),
        "policy-b": Document(
            "policy-b",
            "Dependents",
            "1.0",
            "2026-01-01",
            "current",
            "# Dependents\n\nUpload a birth certificate to add a child.",
        ),
        "policy-old": Document(
            "policy-old",
            "Old enrollment",
            "1.0",
            "2025-01-01",
            "archived",
            "# Old enrollment\n\nEnrollment closes on October 15.",
        ),
    }


@pytest.fixture
def case_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": "case-test",
        "question": "When does enrollment close?",
        "category": "straightforward",
        "answerable": True,
        "relevant_document_ids": ["policy-a"],
        "reference_facts": ["Enrollment closes November 20."],
        "reference_evidence": [
            {"document_id": "policy-a", "quote": "Enrollment closes on November 20."}
        ],
        "intent": "Use the current enrollment closing date.",
    }


@pytest.fixture
def trace_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "case_id": "case-test",
        "trace_origin": "synthetic_fixture",
        "retrieval": {
            "query": "When does enrollment close?",
            "top_k": 3,
            "hits": [{"document_id": "policy-a", "score": 2.0}],
        },
        "response": {
            "outcome": "answer",
            "text": "Enrollment closes November 20.",
            "citations": ["policy-a"],
            "evidence": [{"document_id": "policy-a", "quote": "Enrollment closes on November 20."}],
        },
    }
