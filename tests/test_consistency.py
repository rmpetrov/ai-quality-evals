import json
from pathlib import Path
from typing import cast

import pytest

from ai_quality_evals.consistency import check_consistency
from ai_quality_evals.models import Evidence, ResponseTrace
from ai_quality_evals.validation import parse_response


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            ResponseTrace("abstain", "Missing records.", ("policy-a",), ()),
            ["abstain_has_citations"],
        ),
        (
            ResponseTrace("abstain", "Missing records.", (), (Evidence("policy-a", "Quote"),)),
            ["abstain_has_evidence", "evidence_source_not_cited"],
        ),
        (
            ResponseTrace("answer", "A fact.", (), (Evidence("policy-a", "Quote"),)),
            ["answer_missing_citations", "evidence_source_not_cited"],
        ),
        (ResponseTrace("answer", "A fact.", ("policy-a",), ()), ["answer_missing_evidence"]),
        (
            ResponseTrace("answer", "A fact.", ("policy-b",), (Evidence("policy-a", "Quote"),)),
            ["evidence_source_not_cited"],
        ),
        (ResponseTrace("abstain", "...?! 123", (), ()), ["abstain_explanation_no_letters"]),
        (ResponseTrace("abstain", "个人记录不可用。", (), ()), []),
        (ResponseTrace("abstain", "Personal records are unavailable.", (), ()), []),
        (
            ResponseTrace(
                "answer", "A supported fact.", ("policy-a",), (Evidence("policy-a", "Quote"),)
            ),
            [],
        ),
    ],
)
def test_each_rule_and_clean_responses(response: ResponseTrace, expected: list[str]) -> None:
    findings = check_consistency("case-test", response)
    assert [finding.rule_id for finding in findings] == expected
    assert all(finding.case_id == "case-test" for finding in findings)
    assert all(finding.details == {} or finding.details for finding in findings)


def test_findings_include_objective_details_without_mutating_response() -> None:
    response = ResponseTrace("abstain", "...", ("policy-a",), (Evidence("policy-b", "Quote"),))
    before = repr(response)
    findings = check_consistency("case-test", response)
    assert findings[0].details == {"citations": ("policy-a",)}
    assert findings[1].details == {"evidence_count": 1}
    assert findings[2].details == {"text": "..."}
    assert findings[3].details == {"evidence_index": 0, "document_id": "policy-b"}
    assert repr(response) == before


def test_curated_semantic_defects_remain_external_annotations() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures/semantic-examples.json").read_text())
    assert fixture["fixture_origin"] == fixture["annotation_origin"] == "synthetic_fixture"
    for example in fixture["examples"]:
        response = parse_response(example["response"])
        assert [finding.rule_id for finding in check_consistency(example["case_id"], response)] == (
            example["expected_rules"]
        )
    by_id = {example["example_id"]: example for example in fixture["examples"]}
    # Independently curated labels differ despite identical empty automated finding lists.
    assert by_id["source-contradiction"]["defect_labels"] == ["evidence_contradiction"]
    assert by_id["unsupported-personal-claim"]["defect_labels"] == ["unsupported_personal_claim"]
    assert by_id["fully-supported-answer"]["defect_labels"] == []
    assert by_id["vague-abstention"]["defect_labels"] == ["inadequate_abstention_explanation"]
    assert cast(dict[str, object], by_id["insufficient-support"]["scores"])["citation_support"] == 0
