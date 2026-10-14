# Live RAG and captured replay contract

Phase 2 extends the existing standard-library evaluator with four small modules:
`prompt.py`, `provider.py`, `capture.py`, and `live.py`. Phase 1 corpus, case labels,
BM25 ranking, deterministic metric definitions and offline commands remain intact.
The optional native `ollama.py` adapter reuses these contracts; see the
[local setup and execution guide](local-ollama.md). The OpenAI path remains the default.

## Retrieval, generation and evaluation

Retrieval finds lexical matches using the existing BM25 index over all 14 document bodies.
Generation builds context from those actual hits, constructs one prompt and calls a model.
Evaluation compares captured predictions with independently authored gold labels. It does
not send the labels to the model or invoke another model to score predictions.

Only question text, current retrieved document bodies/metadata, general task instructions
and the response schema reach the provider. The provider interface accepts a
`GenerationRequest`; it cannot accept an `EvaluationCase`. Tests change reference facts
and intent annotations and assert that the complete provider request stays identical.

## Prompt and archived context policy

`benefits_rag_v1` is defined in `prompt.py`. Its SHA-256 covers the version, exact general
instructions and strict JSON schema using canonical JSON. Each trace also stores the exact
serialized user input, so question/context can be verified independently of that template
hash. There are no prompt variants or reference-answer injections.

The prompt requires current-document-only guidance, abstention for insufficient context,
source IDs and exact supporting quotes, and no unsupported personal eligibility decisions.
It identifies both documents and the question as untrusted data, never instructions.
This instruction reduces an attack surface but does not prove prompt-injection immunity.

`exclude_archived_v1` is a transparent post-retrieval filter. The raw top-K rankings and
scores remain untouched. All archived hits are omitted from generation context and stored
with full text/metadata and reason `archived_policy`. Current hits retain relative rank
order. No extra retrieval or refill occurs. Gold sources and dates are not used to select
replacement documents; the corpus's explicit `status` metadata drives the decision.

At K=1 the archived distractors in cases 16 and 17 produce an empty context, even though a
current relevant source exists at rank 2. At K=3 those sources are included. This preserves
retrieval defects and makes effective context coverage separately observable. The policy
supports current guidance only; historical-policy questions would need a later explicit
context policy. “Current” means corpus metadata, not a clock-based policy interpretation.

## Minimal response schema

```json
{
  "outcome": "answer",
  "text": "Regular employees need at least 20 scheduled hours per week.",
  "citations": ["eligibility"],
  "evidence": [{
    "document_id": "eligibility",
    "quote": "Regular employees scheduled for at least 20 hours per week are eligible for benefits."
  }]
}
```

All four fields are required; extra fields are forbidden. Outcome is `answer` or `abstain`;
text must be nonblank. Citations are unique stable document IDs; evidence contains unique
`document_id`/nonblank `quote` pairs. The API schema enforces the basic object structure;
the offline parser additionally enforces IDs, nonblank text, duplicate keys and uniqueness.

Missing citations, nonexistent IDs and fabricated quotes are prediction-quality defects,
not invalid input structure, and reach deterministic evaluators. Abstention is a model
outcome; provider refusal or transport failure has no parsed response and is a separate
operational failure. The prompt requests empty citations/evidence on abstention; the
schema does not enforce semantic relationships between fields.

## Official SDK adapter

The `live` extra pins `openai==3.29.0`. The SDK is lazily imported only when a real
`OpenAIProvider` is explicitly constructed. Ordinary installation/import/evaluation,
pytest, Ruff, mypy and the offline `quality` CI job do not need it. A separate `sdk-contract`
CI job installs the pinned SDK dependencies and local package, then runs
`scripts/check_sdk_offline.py` with local MockTransport responses and sockets blocked during
requests. It uses a dummy credential and needs no API key secret or provider access.

The adapter uses `client.responses.create`, `instructions`, `input`, a strict
`text.format` JSON schema, explicit `model`, `max_output_tokens` and per-request `timeout`.
It sets `store=False`, supplies no tools or conversation history, and sets client
`max_retries=0`. Models without these capabilities fail visibly rather than falling back
to another model, JSON mode or synthetic responses. Provider availability is account-specific.

The implementation was checked against official [Python SDK documentation](https://developers.openai.com/api/reference/python),
[Responses create reference](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
and [Structured Outputs guidance](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)
on October 13, 2026. The SDK documents default retries and configurable timeouts; this
adapter disables retries. A scalar SDK timeout controls transport operations and is not
a guarantee of cancellation or a strict overall spending deadline.

Authentication, rate limiting, connection failure and timeout exceptions map to typed
provider error information. Other API errors retain a safe message, error code, HTTP status
and request ID when available. Provider messages redact the configured key, key-shaped
strings and bearer tokens; raw exceptions, request headers and complete HTTP error bodies
are never saved. SDK responses are inspected for incomplete status/details, refusal blocks,
missing/invalid structured text and failure status. There is no automatic retry.

## V2 capture provenance

Each JSONL line contains:

| Field | Captured fact |
| --- | --- |
| `schema_version` | Integer 2; Phase 1 records remain integer 1 |
| `case_id`, `question` | Selected dataset case and exact user question |
| `trace_origin`, `execution`, `provider` | `live_capture`/`live` with provider `openai` or `ollama`, or explicitly `synthetic_fixture`/`simulated` for fakes |
| `retrieval` | Raw query, requested top-K and original ranked IDs/scores |
| `context` | Policy version, exact supplied document text/metadata, excluded archived snapshots/reasons, canonical context text |
| `prompt` | Version, hash, exact instructions, schema and serialized model input |
| `settings` | Requested model, timeout, max output tokens and retry count 0 |
| `generation` | Status, raw output text, response/request IDs, returned model, actual reported token counts, request latency in milliseconds, error details |
| `response` | Independently parsed response; null on generation failure |
| `corpus_sha256`, `dataset_sha256` | Original source bytes hashed with Phase 1 conventions |

Usage is null when unavailable, never inferred from text or from configured limits. No
pricing is captured or calculated. Returned model metadata may differ from the requested
alias and is preserved. A response ID and HTTP request ID are different fields.
Ollama preserves partially reported input/output token counts with null for missing counts
and total; OpenAI retains the existing all-integer usage contract. Native Ollama response
and request IDs remain null. Its latency includes the local model metadata check and chat.

The raw model text is preserved except credential redaction. Failures without model text
have null output. Partial output from incomplete responses is retained for diagnostics but
not treated as an answer. Batch generation saves each result and stops on the first failure;
omitted cases remain explicit in the report. A process killed before a result is received
cannot capture that result; truncated or otherwise malformed JSONL is rejected on replay.

Actual captures are private artifacts under `live-captures/`, never bundled into the wheel
or uploaded by CI. Test fixtures are under `tests/fixtures/`, expressly marked as simulated.
An injected OpenAI or Ollama transport is also marked simulated. No synthetic response is
reported as a real-provider capture, and no fixture score is a live model benchmark.

## Validation and deterministic replay

`parse_capture` can load a record independently from its self-contained document/prompt
snapshots. Full gold-based evaluation uses the original corpus and dataset (bundled by
default), and checks their hashes, the case/question, and all retrieved document snapshots.
A capture does not embed gold answers in the provider input or substitute predictions for
references. Keep the original inputs if evaluating captures after the repository changes.

Validation checks the trace version and origin/execution pairing, document snapshot types,
exact retrieved/context/excluded membership and order, exclusion reason, serialized context,
known prompt version/instructions/schema and hash, exact question/context model input,
settings, generation status/error consistency, actual token field types, finite latency,
and agreement between raw JSON output and parsed response. Invalid input produces an
explicit error with the JSONL filename/line when loaded through the CLI.

Replay runs neither BM25 nor provider inference. Captured timing, errors, model and usage
remain historical facts. Reports generate no new execution timestamps or latency. Stable
sorting and serialization yield identical report bytes across repeated runs and hash seeds.
The report includes per-case diagnostics, category aggregates, omitted IDs and provenance.
Human review is `null`; no reviewer ratings are inferred from metrics.

## Additional structural metrics

The seven Phase 1 metrics keep their original definitions. V2 reports add five checks;
Phase 1 reports retain their original metric set and fields apart from evaluator version.

| Metric | Numerator / denominator | Aggregate and null behavior |
| --- | --- | --- |
| `citation_in_context` | Citations with an ID supplied to generation / all citations | Micro mean over citations from parsed v2 responses; null if none |
| `context_quote_validity` | Exact case-sensitive body substrings with a cited, supplied source / all evidence entries | Micro mean over evidence from parsed v2 responses; null if none |
| `current_citation_rate` | Citations with a known corpus source marked current / all citations | Unknown and archived IDs count as failures; null if none |
| `effective_context_coverage` | Gold relevant IDs in actual supplied context / gold relevant IDs | Macro mean over v2 cases with gold sources, including failed generations; unanswerable cases null |
| `generation_success_rate` | Completed, structurally parsed responses / captured request attempts | Mean over v2 records, including failures; null if none |

Context checks use the full historical context, even when replay's retrieval cutoff K is
lower than stored K. Raw Recall@K and MRR still use the replay cutoff. Context quotations
validate document bodies, not metadata or exclusion snapshots. A quote in a valid corpus
source absent from generation context fails the context check even if the original
corpus-wide quote check passes. Neither check establishes that a claim follows from a quote.

Generation failures are excluded from response-quality denominators rather than counted
as incorrect model abstentions. Always inspect request counts and generation success rate
alongside response metrics; otherwise successful-only metrics can conceal provider failure.
Missing citations/evidence require coverage gates as well as validity gates. Mixed-origin
files report origin `mixed` and retain provenance for each v2 record; new context metric
denominators include only v2 records. Avoid using mixed inputs as a model benchmark.

Thresholds retain explicit inclusive bounds. No thresholds means `not_evaluated`/`passed:
null`; an empty requested denominator fails even at minimum zero. CLI exit 1 denotes a
failed configured gate. Exit 2 denotes invalid usage/input or any captured generation
failure, and takes precedence over a gate exit. Reports still show the configured gate
status independently of operational success. Captures are retained even when exit 2 occurs.

## What this does not measure

No structural score measures free-form correctness, all unsupported claims, semantic
faithfulness, reference-fact coverage, prompt-injection resistance, personal eligibility
correctness or hallucination-free output. A test returns “200 hours per week” while quoting
the exact policy's “20 hours”; every structural metric passes. The report repeats this
limitation and leaves human scores null. See [the human rubric](human-review.md).

The corpus is small and synthetic, so even reviewed results are not a general model-quality
estimate. Live generation can vary across requests and provider model changes; deterministic
replay guarantees reproducible evaluation of fixed captured inputs, not reproducible model
output. No LLM-as-a-Judge, agents, chatbot, vector database or multi-provider framework is
part of this phase.
