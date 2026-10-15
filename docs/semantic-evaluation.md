# Offline semantic evaluation

The saved local A/B runs declared answerability accuracy of 3/8 for prompt v1 and 6/8
for v2. These scores measure outcome classification, not factual correctness. Inspection
also found an answer equating dental and medical limits while quoting a source that
separates them, and an assertion of personal claim payment supported only by a general
definition of "Paid". A valid source ID and exact quote can accompany an unsupported or
contradictory conclusion. No unreviewed output has passed semantic evaluation.

Phase 3 adds an offline consistency layer and reviewer-owned annotations. It preserves
both prompt templates/hashes, captures, provider implementations, retrieval, original
metric definitions and replay report formats. It makes no model requests and adds no
semantic quality gates or LLM judge.

## What is automated

The separate `response_consistency_v1` rules report case IDs and objective details:

| Rule ID | Condition |
| --- | --- |
| `abstain_has_citations` | Abstention has one or more citations |
| `abstain_has_evidence` | Abstention has one or more evidence entries |
| `answer_missing_citations` | Answer has no citations |
| `answer_missing_evidence` | Answer has no evidence |
| `evidence_source_not_cited` | An evidence document ID is absent from citations |
| `abstain_explanation_no_letters` | Abstention text has no Unicode alphabetic characters |

The final rule only rejects nonlinguistic text such as `...?!` or numbers alone. It does
not assess meaningful prose: "I don't know" can be inadequate without triggering it.
There are no keyword-based semantic detectors. A correct factual answer labeled
`abstain` with empty citations/evidence also requires human judgment. Failed generations
are uncheckable, not clean answers. Findings never rewrite model output. An empty finding
list is not evidence of factual correctness, grounding or a useful abstention explanation.

## Prepare and complete annotations

Run from the repository root with the existing Python 3.13 `.venv`. `PYTHONPATH=src`
uses this checkout. The four existing A/B captures are under:

```text
live-captures/prompt-ab-v1-01/traces.jsonl
live-captures/prompt-ab-v2-01/traces.jsonl
live-captures/prompt-ab-v1-02/traces.jsonl
live-captures/prompt-ab-v2-02/traces.jsonl
```

Create separate, unscored templates bound to each exact file's SHA-256:

```bash
for run in prompt-ab-v1-01 prompt-ab-v2-01 prompt-ab-v1-02 prompt-ab-v2-02; do
  PYTHONPATH=src .venv/bin/python -m ai_quality_evals review prepare \
    --traces "live-captures/$run/traces.jsonl" \
    --output "human-reviews/$run.json"
done
```

Preparation never supplies a reviewer, timestamp, rating or semantic defect. Parsed
responses start `unreviewed`; provider failures start `unreviewable` with a machine
explanation and null annotations. Output paths must be in Git-ignored `human-reviews/`
or `live-captures/`; commands refuse to overwrite existing artifacts or inputs. If the
templates already exist, edit those files rather than recreating them.

Open the JSON annotations in your editor, leaving all capture files unchanged. Use the
[five-dimension rubric](human-review.md) to inspect each question, exact supplied context,
model response and reference facts. For every reviewed case:

1. Change `status` to `reviewed`.
2. Enter your actual identifier in `reviewer` and the actual review time in `reviewed_at`
   using `YYYY-MM-DDTHH:MM:SSZ` or an explicit timezone offset.
3. Set each applicable `scores` entry to integer 0, 1 or 2. Use null for an inapplicable
   dimension and explain why. Supply a nonblank `justifications` entry for every dimension.
4. Optionally add `notes` and defect labels from `evidence_contradiction`,
   `unsupported_personal_claim`, `unsupported_claim`, `incorrect_abstention`,
   `missing_evidence`, or `inadequate_abstention_explanation`.

Do not insert a sample reviewer identity or timestamp as a real review. Keep unfinished
cases `unreviewed` with null scores/justifications/identity and empty labels. To mark a
parsed response `unreviewable`, provide your identity, review time and nonblank `notes`
explaining why; all scores/justifications stay null and labels empty. Failed generations
cannot be rated. Missing case rows are allowed but explicitly remain unreviewed or
unreviewable; malformed or partly populated reviewed rows are rejected.

The review set declares `annotation_origin: human_review`. Curated tests instead use
`synthetic_fixture`, including clearly simulated reviewer identifiers and times. Capture
origin is reported separately: human annotations on a synthetic response do not turn it
into a real model result. The software validates the data contract; it cannot authenticate
reviewer identity or prove who authored an annotation.

## Validate, summarize and compare offline

Validation checks exact hash binding, known and unique case IDs, states, rating types,
timestamps, justifications and labels. A valid annotation file can still contain no
completed reviews; validation success is not a semantic pass.

```bash
for run in prompt-ab-v1-01 prompt-ab-v2-01 prompt-ab-v1-02 prompt-ab-v2-02; do
  PYTHONPATH=src .venv/bin/python -m ai_quality_evals review validate \
    --traces "live-captures/$run/traces.jsonl" \
    --reviews "human-reviews/$run.json"
done

# One-run report, also usable while every case is still unreviewed
PYTHONPATH=src .venv/bin/python -m ai_quality_evals review summarize \
  --traces live-captures/prompt-ab-v2-01/traces.jsonl \
  --reviews human-reviews/prompt-ab-v2-01.json --top-k 3 \
  --output human-reviews/v2-01-summary.json

# Two separate paired comparisons: v1 on the left, v2 on the right
for pair in 01 02; do
  PYTHONPATH=src .venv/bin/python -m ai_quality_evals review summarize \
    --traces "live-captures/prompt-ab-v1-$pair/traces.jsonl" \
    --reviews "human-reviews/prompt-ab-v1-$pair.json" --top-k 3 \
    --compare-traces "live-captures/prompt-ab-v2-$pair/traces.jsonl" \
    --compare-reviews "human-reviews/prompt-ab-v2-$pair.json" \
    --output "human-reviews/comparison-$pair.json"
done
```

No Ollama server, SDK credentials or network is needed. Exit 0 means successful review
processing; exit 2 means invalid input, annotations or output paths. No human scores are
added to automatic quality gates. Choose a fresh summary filename after editing annotations.

Version-1 `semantic_review` reports keep three separate sections for each run:

- `automatic_evaluation`: the unchanged replay report and structural metrics, denominators
  and provenance. Its original `human_review` field stays null.
- `response_consistency`: rule definitions, checked/uncheckable IDs and explicit findings.
- `human_semantic_review`: authored ratings and justifications, defect counts, applicable
  denominators, inapplicable dimensions and missing/unreviewed/unreviewable cases.

Zero ratings and explicit defect labels are highlighted as reviewer findings. Personal
claim defects are highlighted only when a reviewer supplies `unsupported_personal_claim`.
Without reviews, semantic means remain null. Comparisons use only matching reviewed cases
with applicable ratings on both sides, show right-minus-left deltas and denominators,
list newly annotated defects and highlight score decreases to zero as material regressions.
Different reviewers and cases that cannot be paired remain visible. Synthetic and live
capture origins, or synthetic and human annotation origins, cannot be mixed in comparisons.

Judge evidence contradiction by reading the claim and the cited source together. Judge
personal claims by whether actual individual records support them; a status definition
does not establish an individual's status. These decisions are manual annotations, not
text-pattern detection. Review evidence completeness as well as quote validity.

Four cases repeated twice are a targeted regression sample, not a general model-quality
estimate. Reviewer subjectivity, inconsistent applicability decisions, model variation
and missing reviews limit comparisons. Confirm matching case/context/settings/model
provenance before drawing an A/B conclusion. Keep raw files and annotated reviews private;
do not publish local personal responses or reviewer data. Test fixtures are authored
synthetic examples with expected annotations, not independently reviewed real results.
