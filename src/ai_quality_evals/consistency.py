"""Offline field consistency checks; no semantic inference or response repair."""

from dataclasses import dataclass

from ai_quality_evals.models import ResponseTrace

RULES_VERSION = "response_consistency_v1"
RULES = {
    "abstain_has_citations": "Abstentions must have empty citations.",
    "abstain_has_evidence": "Abstentions must have empty evidence.",
    "answer_missing_citations": "Answers must include citations.",
    "answer_missing_evidence": "Answers must include evidence.",
    "evidence_source_not_cited": "Each evidence document must be listed in citations.",
    "abstain_explanation_no_letters": (
        "Abstention text with no Unicode alphabetic characters cannot explain missing information."
    ),
}


@dataclass(frozen=True, slots=True)
class ConsistencyFinding:
    case_id: str
    rule_id: str
    details: dict[str, object]


def check_consistency(case_id: str, response: ResponseTrace) -> tuple[ConsistencyFinding, ...]:
    """Check only observable structure; empty findings do not establish correctness."""
    findings: list[ConsistencyFinding] = []
    if response.outcome == "abstain":
        if response.citations:
            findings.append(
                ConsistencyFinding(
                    case_id, "abstain_has_citations", {"citations": response.citations}
                )
            )
        if response.evidence:
            findings.append(
                ConsistencyFinding(
                    case_id, "abstain_has_evidence", {"evidence_count": len(response.evidence)}
                )
            )
        # Deliberately conservative: no word lists, sentiment, or claim interpretation.
        if not any(character.isalpha() for character in response.text):
            findings.append(
                ConsistencyFinding(
                    case_id, "abstain_explanation_no_letters", {"text": response.text}
                )
            )
    else:
        if not response.citations:
            findings.append(ConsistencyFinding(case_id, "answer_missing_citations", {}))
        if not response.evidence:
            findings.append(ConsistencyFinding(case_id, "answer_missing_evidence", {}))
    for index, evidence in enumerate(response.evidence):
        if evidence.document_id not in response.citations:
            findings.append(
                ConsistencyFinding(
                    case_id,
                    "evidence_source_not_cited",
                    {"evidence_index": index, "document_id": evidence.document_id},
                )
            )
    return tuple(findings)
