# Controlled prompt consistency experiment

The reviewed local v1 captures contain four completed `qwen2.5:3b` generations. Human
inspection found supported factual answers labeled `abstain` in cases 02, 18 and 17;
case 18 also omitted a necessary document's evidence. Case 22 appropriately abstained
because personal claim information was unavailable. Declared answerability accuracy was
1/4 (0.25), while retrieval found all expected sources for the three answerable cases.
This motivates a prompt consistency hypothesis; it is not evidence that v2 improves
the model. No v2 inference was performed during implementation.

## Controlled change and provenance

`benefits_rag_v2` gives explicit decision rules: supply a supported answer with outcome
`answer`, cite and quote every material claim and each necessary source, or abstain with
a missing-information explanation and empty citation/evidence arrays. It asks the model
to check that `outcome`, `text`, `citations` and `evidence` agree. It contains no gold
facts, case IDs, expected outcomes or case-specific examples.

The default remains `benefits_rag_v1`, with its exact original instructions. Both versions
use the same response schema. The immutable template hashes are:

```text
benefits_rag_v1: 008e7c5d94a0f33d64e85f78964b68a9660ff11bfc53b3569f9295ce9ac36a58
benefits_rag_v2: 42b707506ede48a26939e828384fd8dd68b6655de35101847b35df6b81a7196f
```

Each capture retains its prompt version/hash/instructions/schema, exact question/context,
requested settings, model metadata and original model output. Replay recognizes both
templates and rejects changed instructions or schemas even if their hash is recomputed.
Prompt version v2 does not change capture schema version 2. Legacy schema-1 traces and
schema-2 captures made with prompt v1 remain supported.

Retrieval, archived-context filtering, providers, response parsing, metric definitions
and evaluator thresholds are unchanged. The evaluator preserves inconsistent outcomes
and missing evidence as model defects; it never rewrites them. There is no automated
semantic judge. Tests and smoke checks use simulated outputs or offline replay only.

## Manual A/B runs

Run from the repository root using the existing Python 3.13 `.venv` and local Ollama
setup in [local-ollama.md](local-ollama.md). `PYTHONPATH=src` selects this checkout rather
than a previously installed package. These commands perform real local inference and
are deliberately manual. The model weights must already be installed and the loopback
server running with cloud disabled. Keep the same weights, server configuration, corpus,
dataset and checkout between A and B; do not download or replace the model between runs.

The reviewed captures used the following controls; hold all of them constant:

| Control | Value |
| --- | --- |
| Provider/model | Ollama / `qwen2.5:3b` |
| Cases, in request order | `case-02`, `case-18`, `case-17`, `case-22` |
| Maximum requests | 4, sequential, zero retries |
| Raw retrieval top-k | 3 |
| Maximum output tokens | 800 |
| Timeout | 120 seconds per HTTP operation |

Use two fresh output directories. The CLI refuses existing captures/reports. For another
pair, change both directory suffixes together (for example `-02`) while retaining every
other setting. Capture each run separately; do not overwrite or relabel the original v1
captures. The only requested generation change below is the prompt version.

```bash
# A: fresh v1 control
PYTHONPATH=src .venv/bin/python -m ai_quality_evals live \
  --provider ollama --allow-live --model qwen2.5:3b \
  --prompt-version benefits_rag_v1 \
  --max-cases 4 --case case-02 --case case-18 --case case-17 --case case-22 \
  --top-k 3 --max-output-tokens 800 --timeout 120 \
  --captures live-captures/prompt-ab-v1-01/traces.jsonl

# B: v2 experiment
PYTHONPATH=src .venv/bin/python -m ai_quality_evals live \
  --provider ollama --allow-live --model qwen2.5:3b \
  --prompt-version benefits_rag_v2 \
  --max-cases 4 --case case-02 --case case-18 --case case-17 --case case-22 \
  --top-k 3 --max-output-tokens 800 --timeout 120 \
  --captures live-captures/prompt-ab-v2-01/traces.jsonl
```

Each command writes `report.json` beside its capture. No thresholds are requested: the
reports are observation-only and exit 0 alone does not demonstrate answer correctness.
If a batch fails, its partial captures remain, exit is 2, and omitted cases are reported.
Compare only completed four-case runs; keep failed attempts and use fresh directories
for any deliberately rerun pair.

## Offline replay and human comparison

After capturing both runs, Ollama can be stopped. These commands make no model requests:

```bash
PYTHONPATH=src .venv/bin/python -m ai_quality_evals validate \
  --traces live-captures/prompt-ab-v1-01/traces.jsonl
PYTHONPATH=src .venv/bin/python -m ai_quality_evals validate \
  --traces live-captures/prompt-ab-v2-01/traces.jsonl

PYTHONPATH=src .venv/bin/python -m ai_quality_evals replay \
  --traces live-captures/prompt-ab-v1-01/traces.jsonl --top-k 3 \
  --output live-captures/prompt-ab-v1-01/replay.json
PYTHONPATH=src .venv/bin/python -m ai_quality_evals replay \
  --traces live-captures/prompt-ab-v2-01/traces.jsonl --top-k 3 \
  --output live-captures/prompt-ab-v2-01/replay.json
```

Check that both captures contain all four IDs with `generation.status = completed`, the
expected prompt hashes, identical settings, corpus/dataset hashes, retrieval hits and
actual context. Compare returned model metadata as well. Only the prompt template/hash
should differ in generation inputs. Model outputs, tokens and measured latency may vary.

Compare `answerability_accuracy` and the per-case raw outcomes first. Then review
citation/evidence coverage, exact/context quote validity, current citations and generation
success rate. In this sample, a consistent supported response should use `answer` for
cases 02, 18 and 17, while case 22 should continue to abstain with empty citations/evidence.
These are review expectations, never provider input.

Use the [human review rubric](human-review.md) to check factual support and every material
claim's evidence against the supplied current documents. For case 18, verify that both
necessary sources appear in citations and exact supporting quotes; a valid single quote
does not establish evidence completeness. Review case 22 for unsupported personal facts.
Inspect contradictions among all four response fields directly in the captures.

Report A/B counts and per-case regressions alongside the metric values. Structural gates
cannot establish semantic support or multi-document completeness. Four cases are a
targeted regression probe, not a general quality estimate. Repeated manual pairs can
expose model variation; keep their controls and artifacts unchanged and separate.
