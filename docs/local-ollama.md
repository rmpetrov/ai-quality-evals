# Local RAG evaluation with Ollama

The optional `live --provider ollama` path uses the native local API and the existing
BM25 retrieval, `exclude_archived_v1` context policy, `benefits_rag_v1` prompt, strict
response schema, deterministic metrics and v2 offline replay. It requires no OpenAI SDK,
`OPENAI_API_KEY`, Ollama account or paid API. The default `live` provider remains OpenAI.

## Apple Silicon setup

Install Ollama using the [official macOS installer](https://ollama.com/download/mac).
The [macOS guide](https://docs.ollama.com/macos) documents Apple M-series support and
macOS requirements. Install this repository into your Python 3.13 environment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

Quit any running Ollama menu-bar application before starting this dedicated server in a
separate terminal. Disable cloud features in the **server's** environment:

```bash
OLLAMA_NO_CLOUD=1 OLLAMA_HOST=127.0.0.1:11434 ollama serve
```

Alternatively, configure `disable_ollama_cloud: true` in your existing
`~/.ollama/server.json` and restart the application, as described in the
[Ollama cloud-disable instructions](https://docs.ollama.com/faq#how-do-i-disable-ollamas-cloud-features).
Setting a variable only for `ai-evals` does not configure an already running server.

Download the [qwen2.5:3b local model](https://ollama.com/library/qwen2.5:3b) manually in
another terminal. This setup step needs Internet access and downloads model weights;
the evaluation code never downloads or replaces a model:

```bash
ollama pull qwen2.5:3b
ollama list
```

## Explicit, bounded evaluation

With the server running and weights installed, explicitly authorize a small local batch:

```bash
ai-evals live --provider ollama --allow-live --model qwen2.5:3b \
  --max-cases 3 --case case-02 --case case-17 --case case-21 \
  --timeout 120 --max-output-tokens 800 \
  --captures live-captures/qwen-local-run/traces.jsonl
```

Each selected case gets one metadata check and at most one generation request. The default
limit is three cases; the hard maximum is 24. Explicit selections must fit the limit.
Requests are sequential, retries are zero, and the batch stops on the first failure.
Output tokens are limited to 16–4096 via `options.num_predict`. `--timeout` is a finite,
positive timeout in seconds for each HTTP operation, including model checks. The default
is 30 seconds; cold loading may need longer. This socket timeout is not a strict batch
deadline and cannot guarantee cancellation of server-side work after a timeout.

The CLI writes `report.json` beside the capture. Both paths must be under a Git-ignored
`live-captures/` directory and must not already exist. Omit `--captures` for a unique run
directory. Every result, including a handled failure, is saved before the next case.
Choose a fresh path for subsequent runs.

The transport connects only to `127.0.0.1:11434`, with no configurable remote endpoint,
proxy support or redirect following. It ignores `OLLAMA_HOST`, `OPENAI_BASE_URL` and proxy
variables in the evaluator process. Before each chat it calls local `/api/show` to check
installed model metadata and reject remote/cloud-backed aliases. Cloud IDs, URL/registry
IDs and unverifiable model weights fail visibly. Missing models require manual installation;
there is no `/api/pull`, cloud fallback or call to the OpenAI provider.

The [native chat request](https://docs.ollama.com/api/chat) sends the unchanged prompt
instructions as a system message and the exact question/retrieved context as a user message.
It passes the existing JSON schema directly in `format`, with `stream: false`. The same
strict parser validates the resulting `message.content`; malformed output is captured,
never repaired or replaced with a fixture. Token-limit truncation is an incomplete response.

## Provenance and offline replay

Real local attempts use `provider: "ollama"`, `trace_origin: "live_capture"` and
`execution: "live"`. An injected test transport uses `synthetic_fixture`/`simulated`.
The capture retains requested/returned model IDs, raw generation text, status/errors,
prompt/context snapshots and measured latency. Latency includes the metadata check and
chat request. Ollama supplies no OpenAI response/request ID, so those fields stay null.

Reported `prompt_eval_count` and `eval_count` become input/output tokens. Total tokens
are their sum when both are available. Missing individual counts and an unavailable
total stay null; if neither count is reported, usage is null. V2 nullable token counts
are accepted only for provider `ollama`; the existing OpenAI capture contract stays strict.
No text-based token estimates or prices are calculated.

Stop Ollama and replay entirely offline, without a key or SDK:

```bash
ai-evals validate --traces live-captures/qwen-local-run/traces.jsonl
ai-evals replay --traces live-captures/qwen-local-run/traces.jsonl \
  --minimum generation_success_rate=1 --minimum citation_in_context=1 \
  --minimum context_quote_validity=1 --minimum current_citation_rate=1 \
  --output live-captures/qwen-local-run/replay.json
```

Exit codes remain 0 for observation/pass, 1 for a failed configured metric gate, and 2
for invalid configuration/input or a captured generation failure. Failed generations are
not model abstentions. Replay preserves historical output, usage and latency and verifies
the original corpus/dataset and prompt/context integrity.

Structural metrics do not establish semantic correctness or absence of hallucinations.
Use the [human review rubric](human-review.md) for actual outputs. No local model benchmark
is claimed by implementation tests: all test responses are fabricated, sockets are blocked,
and both existing GitHub Actions jobs remain offline. Implementation verification performs
no model downloads or genuine inference.
