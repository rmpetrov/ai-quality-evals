import re
from pathlib import Path

import pytest

from ai_quality_evals.capture import parse_capture
from ai_quality_evals.models import GenerationContext
from ai_quality_evals.prompt import (
    PROMPT_INSTRUCTIONS,
    PROMPT_VERSION,
    build_prompt,
    context_text,
    prompt_hash,
    response_schema,
)
from ai_quality_evals.runner import DEFAULT_DATA
from ai_quality_evals.validation import load_cases, load_corpus, parse_json


def test_v1_default_and_explicit_selection_match_original_snapshots() -> None:
    fixture = Path(__file__).parent / "fixtures/captured-format.jsonl"
    for line in fixture.read_text().splitlines():
        record = parse_capture(parse_json(line))
        assert record.prompt.version == PROMPT_VERSION == "benefits_rag_v1"
        assert record.prompt.sha256 == (
            "008e7c5d94a0f33d64e85f78964b68a9660ff11bfc53b3569f9295ce9ac36a58"
        )
        assert build_prompt(record.question, record.context) == record.prompt
        assert (
            build_prompt(record.question, record.context, prompt_version="benefits_rag_v1")
            == record.prompt
        )


def test_v2_changes_only_template_with_independent_schema_snapshots() -> None:
    context = GenerationContext("exclude_archived_v1", (), (), context_text(()))
    v1 = build_prompt("What information is available?", context)
    v2 = build_prompt("What information is available?", context, prompt_version="benefits_rag_v2")
    assert v2.version == "benefits_rag_v2"
    assert v2.instructions != v1.instructions
    assert v2.sha256 != v1.sha256
    assert v2.sha256 == "42b707506ede48a26939e828384fd8dd68b6655de35101847b35df6b81a7196f"
    assert v2.sha256 == prompt_hash(v2.version, v2.instructions, v2.response_schema)
    assert v2.input_text == v1.input_text
    assert v2.response_schema == v1.response_schema == response_schema()
    v2.response_schema.clear()
    assert v1.response_schema == response_schema()
    assert build_prompt("Question", context, prompt_version="benefits_rag_v2").response_schema == (
        response_schema()
    )


def test_unknown_prompt_version_is_rejected() -> None:
    context = GenerationContext("exclude_archived_v1", (), (), context_text(()))
    with pytest.raises(ValueError, match="unsupported prompt version"):
        build_prompt("Question", context, prompt_version="benefits_rag_v999")


@pytest.mark.parametrize("prompt_version", tuple(PROMPT_INSTRUCTIONS))
def test_templates_do_not_embed_gold_labels_or_dataset_solutions(prompt_version: str) -> None:
    documents = load_corpus(DEFAULT_DATA / "kb")
    cases = load_cases(DEFAULT_DATA / "cases.jsonl", documents)
    instructions = PROMPT_INSTRUCTIONS[prompt_version]
    assert not re.search(r"case-\d+", instructions)
    for field in ("reference_facts", "reference_evidence", "relevant_document_ids", "answerable"):
        assert field not in instructions
    for case in cases:
        assert case.case_id not in instructions
        for fact in case.reference_facts:
            assert fact not in instructions
        for evidence in case.reference_evidence:
            assert evidence.quote not in instructions
