# Human review rubric

All generated reports begin with `human_review: null`. This means **unreviewed**, not zero,
pass or perfect. Deterministic metrics never create, infer or replace a reviewer score.
Human ratings are not included in automated gates in Phase 2.

Review each parsed response against the question, the exact supplied current context,
the independently authored reference facts and evidence, and the full corpus when diagnosing
retrieval/context omissions. Inspect raw ranking and archived exclusions first. Mark provider
failures and absent responses unreviewable; do not score them as model abstentions.

For every applicable dimension, supply an integer 0, 1 or 2 and a short justification:

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Answer correctness | Materially wrong or misleading | Partially correct; important qualification missing | Correct for the question and available records |
| Groundedness / unsupported claims | Material claims lack supplied support | Minor unsupported claim or overstatement | All material claims supported by supplied current context |
| Necessary fact coverage | Necessary reference facts omitted | Some necessary facts covered | Necessary facts and qualifications covered |
| Appropriate abstention | Answers without needed support, or unjustifiably refuses | Abstention boundary or explanation partly suitable | Answers when supported; otherwise abstains and identifies missing information |
| Citation support | Sources/quotes do not support material claims | Partial or unclear support | Sources and exact quotes support the material claims they accompany |

Use `null` with an explanation for inapplicable dimensions. For appropriate abstention,
consider both gold answerability and actual context sufficiency: an answerable gold case
can legitimately require abstention if retrieval omitted necessary sources. The automated
answerability metric cannot express that distinction. An abstention can earn 2 for this
rubric while failing gold answerability accuracy.

Record ratings separately from immutable raw captures. A suggested reviewer-owned record:

```json
{
  "capture_sha256": "SHA-256 of the exact reviewed capture file",
  "case_id": "case-02",
  "reviewer": "Reviewer identifier",
  "reviewed_at": "Reviewer-supplied date/time",
  "status": "reviewed",
  "scores": {
    "answer_correctness": null,
    "groundedness": null,
    "necessary_fact_coverage": null,
    "appropriate_abstention": null,
    "citation_support": null
  },
  "notes": "Populate ratings and supporting rationale after human inspection."
}
```

This is a template, not a completed review or an auto-populated score record. A future
review-ingestion feature could validate reviewer data; Phase 2 deliberately does not add
one. Never present a fixture result or an unreviewed capture as an independently reviewed
model benchmark. The “200 hours” counterexample in `test_live.py` shows why passing
structural checks alone is insufficient.
