# Phase 1 completion report

Verified locally on October 13, 2026, using Python 3.13.3 on macOS.

**Verdict: READY FOR INITIAL COMMIT.** Phase 1 is complete. No commit, push, pull request,
or remote repository was created. Local Git is initialized on `main` with no remotes.

## 1. Architecture overview

The standard-library runtime uses immutable dataclasses, strict input loaders, a real
lexical retriever, independent metric functions, and a small evaluation runner/CLI.

`Markdown corpus + question text -> BM25 -> ranked hits -> gold comparison -> JSON + gates`

`Stored synthetic retrieval/response traces -> validation -> deterministic checks -> JSON + gates`

Gold labels are visible only to validation and evaluation, never to retrieval. Retrieval
mode produces no response; replay mode runs no retrieval or generation. Corpus, case,
and trace SHA-256 hashes, evaluator version, parameters, denominators, and per-case
evidence make results auditable. Reports omit timestamps and absolute input paths.

## 2. Files created

40 publishable files were created, grouped below. Generated reports, wheels, caches,
build files, and virtual environments are excluded by `.gitignore`.

```text
.github/workflows/ci.yml
.gitignore
LICENSE
README.md
pyproject.toml
requirements-dev.lock
docs/phase1-report.md
scripts/check_offline.py
src/ai_quality_evals/
  __init__.py
  __main__.py
  cli.py
  metrics.py
  models.py
  py.typed
  retrieval.py
  runner.py
  validation.py
  data/
    cases.jsonl
    replays/good.jsonl
    replays/bad.jsonl
    kb/
      appeals.md
      claim-status.md
      claim-submission.md
      commuter.md
      dental-limits.md
      dependents.md
      eligibility.md
      enrollment-2025.md
      enrollment-2026.md
      life-events.md
      medical-limits.md
      support.md
      wellness-2025.md
      wellness-2026.md
tests/
  conftest.py
  test_cli.py
  test_metrics.py
  test_retrieval.py
  test_runner.py
  test_validation.py
```

## 3. Dataset categories and counts

The fictional JuniperVale Benefits corpus has 14 short policies: 12 current and two
archived. All policy documents and trace examples are synthetic and locally authored.

| Category | Cases |
| --- | ---: |
| Straightforward | 9 |
| Multi-document | 6 |
| Distractor | 5 |
| Unanswerable | 4 |
| Total | 24 |

Twenty cases have gold sources. Two distractor cases also require multiple documents.
There are six good and six defective synthetic replay records; each set covers the same
four answerable and two unanswerable questions. Reports explicitly list the 18 omitted
dataset cases for each replay run.

## 4. Retrieval implementation

BM25 uses positive IDF `ln(1 + (N - df + 0.5)/(df + 0.5))`, `k1=1.5`, and `b=0.75`.
The indexed text is the body including its title once. ASCII word/number tokens are
case-folded; distinct query terms contribute once in sorted order. Only positive scores
are returned. Full-precision score ties break by document ID; public scores round to
12 decimals after ranking. CLI K is configurable, with a default of three.

No query-to-document mapping, gold-derived hints, embeddings, stopword removal,
date filtering, or model call is used. Numeric term-frequency and length-normalization
tests verify the actual algorithm. The README specifies the complete formula.

## 5. Metrics and exact definitions

| Metric/check | Definition |
| --- | --- |
| Recall@K | Distinct relevant IDs in the first K results divided by distinct gold IDs; macro mean over answerable cases |
| MRR@K | Macro mean of `1/r` for the first relevant source within K, or zero when absent |
| Citation validity | Fraction of all citation entries whose IDs exist in the loaded corpus |
| Exact quotation validity | Fraction of evidence entries with a nonblank, case-sensitive exact substring in the specified body and that source ID in the response citations |
| Answerability accuracy | Fraction of traces whose declared `answer`/`abstain` outcome agrees with expected answerability |
| Citation coverage | Fraction of declared answers with at least one citation |
| Evidence coverage | Fraction of declared answers with at least one evidence entry |
| Schema integrity | Fail-fast validation of required fields, types, unique IDs, source references, score ordering, and exact gold evidence; not an aggregate quality score |

Empty denominators yield null. Unanswerable cases do not enter retrieval averages.
Structural metrics use micro denominators; category summaries preserve the same rules.
Citation diagnostics expose source status and membership in the stored retrieved context.

Existence and exact quotations do not measure semantic faithfulness. An archived exact
quote may pass while the response is wrong. A test explicitly demonstrates that a false
answer can pass both citation and quotation checks.

## 6. Example evaluation results

| Run | Cases | Recall@K | MRR@K | Citation validity | Exact quote validity | Answerability accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Real BM25, K=3 | 24 | 1.000 | 0.950 | null | null | null |
| Good synthetic replay, K=3 | 6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Defective synthetic replay, K=3 | 6 | 0.375 | 0.500 | 0.800 | 0.400 | 0.667 |
| Real BM25, K=1 | 24 | 0.700 | 0.900 | null | null | null |

Retrieval averages cover 20 cases. The two archived-policy distractors (`case-16` and
`case-17`) rank the current source second, explaining the 0.95 MRR at K=3.
The good replay has citation/evidence coverage 1.0; the defective replay has both 2/3.
These replay scores assess fixtures and evaluator behavior, not model performance.

Generated artifacts are available locally under `reports/`:

- `retrieval.json` and `retrieval-repeat.json`
- `replay-good.json` and `replay-bad.json`
- `replay-observation.json` with `gate_status: "not_evaluated"` and `passed: null`
- `retrieval-k1.json`
- `wheel-retrieval.json` from the fresh installation outside the repository

## 7. Automated verification

| Check | Result |
| --- | --- |
| Python 3.13 installation | Fresh development environment installed pinned tooling; regular wheel built successfully |
| Fresh runtime environment | Wheel installed with `--no-index --no-deps`; only the project and pip are installed |
| Wheel portability/data | `validate` and gated retrieval passed from `/private/tmp`, outside the repository |
| Dependency integrity | `pip check` passed in development and fresh runtime environments |
| Ruff | Formatting and lint passed |
| Strict mypy | Passed across 15 source/test/script Python files |
| pytest | **135 passed** after the gate-status review fix |
| Offline smoke checks | Passing retrieval/replay gates, observation-only defective replay, and both expected exit-1 failures verified |
| Determinism | Repeated reports, two hash seeds, and the fresh-wheel report are byte-identical |
| Report traceability | Questions, gold evidence, rankings/scores, response text, individual checks, denominators, and input hashes verified |
| Offline operation | Both modes pass with socket creation disabled in a test |
| Git exclusions | Reports, both virtual environments, `.env` files, build output, and egg metadata verified ignored |
| Credential/data review | No credential-pattern matches or environment files in publishable sources; data was authored synthetically |
| Workflow syntax | GitHub Actions YAML parses successfully |

GitHub Actions uses Python 3.13, pinned action commits, dependency installation, Ruff,
mypy, pytest, and offline smoke checks. Report upload runs even on a failed job.
The hosted workflow has not been run because no remote repository exists.

## 8. Failing quality-threshold demonstrations

The defective replay violates every minimum when all seven metrics require 1.0.
Its fabricated quotation in `case-11` produces `not_an_exact_substring`; the unknown
source in `case-01` fails citation validity. Answering unsupported personal-record and
relocation questions lowers answerability accuracy to 2/3.

```bash
ai-evals replay --traces src/ai_quality_evals/data/replays/bad.jsonl \
  --minimum exact_quote_validity=1 --output reports/replay-bad.json
# Exit 1: exact_quote_validity=0.4 < minimum 1.0

ai-evals retrieval --top-k 1 --minimum recall_at_k=0.95 \
  --output reports/retrieval-k1.json
# Exit 1: recall_at_k=0.7 < minimum 0.95
```

Both write complete failed reports. Invalid data/configuration instead exits 2. A requested
metric with no applicable observations also fails its gate. Without thresholds, a valid
run is observation only, with `gate_status: "not_evaluated"` and `passed: null`; a zero
exit does not certify answer correctness. Configured gates serialize `gate_status: "passed"`
with `passed: true`, or `gate_status: "failed"` with `passed: false`.

## 9. Remaining gaps

- Small authored data has no held-out statistical validation or production representativeness.
- No semantic answer grader, policy freshness selection, or guarantee of hallucination detection.
- No real response generation, live application traces, latency/token/cost measurement,
  or model-provider integration.
- Replays cover a deliberate subset and use synthetic scores; they do not measure an
  end-to-end RAG pipeline.
- Hosted CI execution remains pending creation of the remote repository after review.

## 10. Recommended next phase

Add one small, opt-in real RAG path with captured model, configuration, context, response,
and cost provenance. Keep live quality measurements separate from this offline evaluator
regression suite. Expand independently reviewed answer labels and difficult retrieval
cases before treating thresholds as production gates. Agent tools remain later work.

## 11. Verdict and stopping point

**READY FOR INITIAL COMMIT.** The requested Phase 1 foundation is implemented, tested,
documented, and left uncommitted for review. No Phase 2 integration was implemented.
