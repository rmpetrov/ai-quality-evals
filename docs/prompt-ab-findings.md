# Local RAG prompt A/B findings

Explicit decision rules improved automatic answerability classification in this small
experiment, but the completed run-02 human review found lower factual correctness and
groundedness for v2. The release decision is to **keep `benefits_rag_v1` as the default
and retain `benefits_rag_v2` for research only**.

## Objective and hypothesis

The experiment investigated a response-consistency failure: an informative, supported
answer could declare `outcome: abstain`, sometimes while supplying citations and evidence.
The hypothesis was that clearer answer/abstain rules and explicit evidence requirements
would reduce these inconsistencies without degrading semantic correctness.

The original `benefits_rag_v1` was compared with experimental `benefits_rag_v2`. V2 asks
for a supported answer to use `answer`, requires citations and exact quotes for each
material claim and necessary source, and reserves `abstain` for missing support. Neither
prompt contains gold answers or case-specific solutions. The
[prompt-consistency procedure](prompt-consistency.md) records the templates, hashes and
repeatable manual A/B commands.

## Controlled local experiment

Both prompt versions ran through local Ollama using `qwen2.5:3b`. Two A/B pairs, labeled
01 and 02, each evaluated the same four cases once per prompt: eight responses per prompt
version, 16 responses in total. The earlier diagnostic captures that motivated the
experiment are not included in these counts.

| Control | Value |
| --- | --- |
| Model and provider | `qwen2.5:3b` through local Ollama |
| Selected cases, in request order | `case-02`, `case-18`, `case-17`, `case-22` |
| Retrieval | `top_k=3` |
| Output limit | `max_output_tokens=800` |
| Timeout | 120 seconds per HTTP operation |
| Retries | Zero |
| Requests | Four sequential generations per prompt per run |

Within each pair, questions, generation settings, retrieval results, supplied context,
corpus/dataset hashes and response schema matched. The requested change was the prompt
version. Original outputs and declared outcomes were preserved without repair.

## Automatic answerability results: both runs

`answerability_accuracy` compares the declared `answer`/`abstain` outcome with the dataset's
gold answerability label. It does not score whether the answer text is correct.

| Prompt | Correct classifications / responses | Accuracy |
| --- | ---: | ---: |
| `benefits_rag_v1` | 3/8 | 37.5% |
| `benefits_rag_v2` | 6/8 | 75% |

These are deterministic automatic results from replaying the captured responses for runs
01 and 02. They show better outcome classification for v2 on these observations, not
proof of improved factual correctness or a statistically established model improvement.

## Paired human semantic review: run 02 only

The completed run-02 review uses the five dimensions in the
[human review rubric](human-review.md), scored 0/1/2. These are manually authored judgments,
separate from the automatic metrics. Higher scores are better. Means below use only
cases with an applicable rating for both prompts.

| Dimension | v1 mean | v2 mean | Paired applicable cases |
| --- | ---: | ---: | ---: |
| `answer_correctness` | 2.0 | 1.0 | 4 |
| `appropriate_abstention` | 0.5 | 1.5 | 4 |
| `citation_support` | 1.6667 | 1.3333 | 3 |
| `groundedness` | 2.0 | 1.0 | 4 |
| `necessary_fact_coverage` | 2.0 | 1.25 | 4 |

Citation support excludes `case-22` because it did not have an applicable rating on both
sides; its means are rounded to four decimal places. The other dimensions use all four
paired cases. These scores are not eight-response semantic averages and do not establish
a completed paired human review of run 01.

V2's higher appropriate-abstention mean coexisted with lower correctness, groundedness,
fact coverage and citation support in this reviewed pair. The dimensions expose different
failure modes rather than a single combined quality score.

## Material findings in v2, run 02

**`case-18`: `evidence_contradiction`.** The response treated the dental annual maximum
and medical deductible as the same rule, while its own cited evidence explicitly
separated them. The sources and exact quotation could be valid even though the answer's
conclusion contradicted that evidence. This was a human semantic finding, not an automatic
contradiction detector.

**`case-22`: `unsupported_personal_claim`.** The response asserted that an individual's
reimbursement claim had been paid, relying on a general definition of the status
"Paid". A terminology document does not establish the status of a particular claim.
No individual payment record supported the assertion in the supplied context. Human
review identified this unsupported personal financial-status claim; citation existence
and quote validity could not establish the inference.

## Why structural results were insufficient

A valid citation establishes that a document exists; an exact quotation establishes
that text occurs in its attributed source. Context membership and current-policy checks
add provenance and currency, but none establishes that a material answer claim follows
from the evidence. Coverage checks establish presence, not sufficient semantic support.
Outcome classification can improve while an answer becomes factually worse.

The separate deterministic consistency rules can flag observable field conflicts such
as an abstention with citations or an answer without evidence. They do not determine
arbitrary factual correctness, evidence contradiction or personal-record availability.
The [semantic-evaluation workflow](semantic-evaluation.md) therefore keeps automatic
metrics, consistency findings and explicit human ratings in separate report sections.
Human scores were not silently added to automatic quality gates.

## Release decision

Keep v1 as the default and retain v2 as an explicitly selected research prompt. The
observed answerability gain does not justify promoting v2 given the reviewed contradiction
and unsupported personal claim. This decision preserves both immutable prompt versions
and historical replay. It does not certify v1 as error-free: its outcome-label consistency
remains a known limitation requiring further work.

## Provenance and limitations

The automatic counts were checked against the four local capture/report pairs under
`live-captures/prompt-ab-v1-01/`, `prompt-ab-v2-01/`, `prompt-ab-v1-02/` and
`prompt-ab-v2-02/`. The paired human figures and labels were checked against
`human-reviews/comparison-02-complete.json` and the two run-02 annotation sets, including
their capture-file and annotation hashes. These artifacts remain private and Git-ignored;
this document contains aggregate results and paraphrased findings, not full traces,
reviewer identities, review timestamps or credentials.

- **Sample size:** four selected questions repeated twice are a targeted regression probe,
  not eight distinct questions per prompt or a representative production benchmark.
  No statistical significance is claimed.
- **Stochasticity:** repeated model generations can vary despite matching requested
  settings. Two pairs do not characterize that variation or isolate every runtime factor.
- **Human subjectivity:** rubric scores, applicability and defect labels reflect manual
  judgments. These artifacts do not establish inter-rater agreement or an independent
  multi-reviewer consensus. Paired denominators must remain visible.
- **Reproducibility:** fixed capture bytes, original corpus/dataset bytes, prompt snapshots
  and hash-bound annotations permit deterministic offline replay and summary generation.
  They do not guarantee identical future generations from the same model tag; weights,
  runtime state and software versions can change. Reproducing semantic judgments also
  requires the original annotations. Private raw artifacts are not bundled with this
  portfolio document, so public readers cannot independently reconstruct the full review
  from the aggregate tables alone.
