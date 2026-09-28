# docuzent – Document‑Intelligence at the Edge

A Rust-based local Retrieval‑Augmented Generation (RAG) engine that ingests any file format, builds context‑size hierarchies, and serves queries to LLMs (Ollama, GPT‑4o, etc.) without leaving the machine.

---

## Quick Start

```bash
# Install Rust if you don't have it
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone & build
git clone https://github.com/no-mans-code/docuzent.git
cd docuzent
cargo build --release

# One-off Docling conversion
./target/release/docuzent ingest ./data --output ./out

# Ask questions about one or more documents, loaded as one corpus (any
# format Docling parses, or .zips of them) - --mode defaults to adaptive
./target/release/docuzent ask ./report.pdf --model qwen2.5:3b
./target/release/docuzent ask ./ch1.pdf ./ch2.pdf --model qwen2.5:3b --mode swap

# Already-text files (plain text, markdown, source code, config) skip
# Docling entirely with --text-only - see "Text-only ingestion" below
./target/release/docuzent ask ./src/main.rs --text-only --model qwen2.5:3b

# Benchmark the on-disk context cache's real effect on one document
./target/release/docuzent bench ./report.pdf --model qwen2.5:3b

# Or use the browser UI instead of the CLI - model and cache mode are
# switchable at runtime from the page itself, --model/--mode are only the
# initial pick
cargo run -p docuzent-web -- --model qwen2.5:3b --mode adaptive --port 3800
# then open http://localhost:3800 - upload document(s), ask them questions
```

Requires a running local Ollama (`ollama serve`) with the chosen model pulled, and Docling set up per [Docling Setup](#docling-setup) below. The folder-watching, multi-document RAG mode described further down is on hold; `ask`/`bench` (single-document mode) are what's built and verified today.

---

## Features

| Feature | Description |
|---------|-------------|
| **Local ingestion** | Any format Docling parses - PDF, DOCX, Markdown, plain text, images (OCR), audio/video (ASR) - or a `.zip` of them, extracted one level deep. |
| **VRAM-aware context sizing** | The default context length is capped by the model's real trained context (not just its RoPE-extrapolated nominal window) and real free VRAM (real GGUF architecture metadata, not guessed) - controllable via `--context-length` or the web UI's slider. See "VRAM-aware context sizing" below. |
| **Text-only ingestion** | Plain text/source-code files skip Docling entirely via `--text-only` / `text_only: true` - reuses the same caching/chunking machinery, just a different front end. |
| **MCP server** | `docuzent-mcp` exposes document Q&A as a tool for coding agents (Claude Code, etc.) over stdio - see "MCP server" below. A Docling-free Docker image (`Dockerfile.mcp`, 149MB) is available for MCP-only use. |
| **Fits-or-map-reduce** | A document that fits the model's context window is answered in one call; a bigger one is split and answered via question-aware map-reduce. |
| **On-disk context cache** | `no-mans-code/kvcache` - a generic, size-bounded, LRU-evicted cache, keyed by `{model}|{context length}|{document hash}`. Tunable capacity, default `min(disk/2, 50GB)`. Loading a genuinely different document set evicts the previous one's entry immediately, not left to LRU aging. |
| **Docling parse cache** | A second, separate on-disk cache (10GB default) for Docling's own parsing output, keyed per source file - independent of model or mode, since parsing is the same work regardless of what's later done with the text. |
| **Three ingestion modes** | `swap` (always reuse the disk cache when available), `raw` (never reuse anything, in-memory or disk - every question pays a full cold reprocess), `adaptive` (default - predicts whether reuse or a cold reprocess will be faster and picks per-request; see "Adaptive Mode" below). |
| **Model agnostic** | Connects to any local LLM via the Ollama API. Selectable at runtime from the browser UI, not just at startup. |
| **Benchmarked, not claimed** | `docuzent bench` measures the cache's real effect on your own machine and model - see "Measured Performance" below. |
| **Atomiser** | Separate, reusable crate (`no-mans-code/atomiser`) for query/task decomposition - not yet wired in here, built for the multi-document RAG mode once it resumes. Tracked for possible transparent use in simple mode too - see open issues. |

---

## Existing Tools Comparison

Below is a high‑level comparison of the most relevant open‑source tools that can help build a local Retrieval‑Augmented Generation (RAG) engine. The table is intentionally concise; feel free to extend it as new projects emerge.

| Tool | Focus | Language | Key Features | Notes |
|------|-------|----------|--------------|-------|
| **LlamaIndex** (LangChain/ChatGLM) | RAG framework | Python | Document loaders, chunking, embeddings, retrieval, prompt templating | Great for rapid prototyping; heavy Python dependency |
| **LangChain** | General LLM orchestration | Python | Multi‑step pipelines, embeddings, vector stores, memory | Mature, but Python‑centric |
| **Qdrant** | Vector store | Rust/Go/JavaScript | Disk‑backed embeddings, filtering, search, API | Native Rust, good for Rust ecosystems |
| **Milvus** | Vector database | Rust, C++, Python | Distributed, high‑scale, ANN search | Enterprise‑grade, heavier setup |
| **Weaviate** | Vector + metadata store | Go | Schema‑based, GraphQL API | Easy to deploy, but heavier runtime |
| **Ollama** | Local LLM + embeddings | Rust | Model hosting, embedding API, lightweight | Excellent for on‑device inference |
| **SvelteKit + Pinecone** | Full stack | JavaScript | Frontend + vector DB | Good for quick web demos |
| **Docling** | Document ingestion & transformation | Python | Multi‑format ingestion, OCR, ASR, chunking | The core of this project's ingestion, shelled out to as a subprocess (no native Rust bindings exist) |

---

## Measured Performance (Single-Document Mode)

Correcting an earlier draft of this section that claimed Docling itself is "Rust-based" - it is a Python library; `docuzent` shells out to its CLI (see [Docling Setup](#docling-setup)). Real measurements, not marketing claims:

- **Context reuse, immediate** (same process, sub-second gap): a follow-up Ollama call reusing its `context` token array showed `prompt_eval_duration` roughly **10x lower** than resending the same ~2500-token document cold (0.30s → 0.02s).
- **Context reuse, cross-session** (`docuzent bench`, the actual scenario disk persistence is for - a genuinely cold session, then a fresh session reusing what it persisted): **no measurable benefit** in the real run - 135ms warm vs. 103ms cold, and per-token throughput was identical between the two, meaning no computation was actually skipped. See `Blueprint.md` → "Single-Document Q&A" for the full honest writeup, including why, and what a real fix would require.
- The on-disk cache (`no-mans-code/kvcache`) is verified to persist and reload correctly across a fresh `Session`/`Cache::open` - that mechanism works exactly as designed. Whether it delivers a speedup depends on Ollama's own internal state, which this project cannot control or extend past what its public API exposes.

Run `docuzent bench <your-file>` yourself for a live measurement on your own machine and model - the numbers above are one real run, not a guarantee.

---

## Adaptive Mode

`--mode adaptive` (the default) doesn't just always reuse the disk cache the way `swap` does - it predicts, per request, whether reusing a cached context or a fresh cold reprocess will actually be faster, and picks whichever wins. The prediction logic is [`no-mans-code/ollama-kv-profiler`](https://github.com/no-mans-code/ollama-kv-profiler)'s own `predictor` module, pulled in as a git dependency rather than reimplemented - that project exists specifically to answer "swap vs. reingest, which wins, and by how much" empirically, and this is that research turned into a real decision inside the product.

Unlike the profiler's own benchmarking (which affords ~10 extra calibration calls per run since it's a research tool), a real product shouldn't pay that cost on every document load. Instead, `docuzent-core::speed_profile::SpeedProfile` maintains a small, persisted, self-calibrating running average per `(model, context length)` - `prefill_tokens_per_sec` and `eval_tokens_per_sec` update from every real ingest call's own timing, `expected_answer_tokens`/`fixed_overhead_ms` update from every real call that reuses a context, all as a free side effect of normal use. The very first time a model is used there's no data yet, so adaptive mode falls back to a cold prime and starts building the profile from that real call - no wasted round trips either way.

Each `AskResponse`/CLI answer reports the real decision made (`adaptive_decision`: predicted swap ms vs. predicted raw ms, and which was chosen) so it's never a black box.

---

## Map-reduce: per-chunk caching, and a real context-size finding

Map-reduce (the path for a document too big for one context window - see `chunks_mapped`/`used_map_reduce` in `AnswerReport`) now honors `Mode` per chunk, not just for the whole document: each chunk gets its own disk-cache entry keyed by its own content hash, checked/predicted through the same swap/raw/adaptive logic as the single-chunk path. Only the chunk's *primed* state is ever persisted - never a state that reflects a specific question's answer, so a second, completely different question about the same large document reuses every chunk's cached context rather than re-processing the document cold again. Confirmed with a real run: asking a second, unrelated question about the same 10-file, 200K-character real document showed 6 of 7 chunks reused from disk with zero re-priming, only the one chunk whose boundary shifted needed a fresh prime.

`--map-reduce-context-fraction` (default `0.25`) controls how much of a model's *nominal* context window a single chunk is sized to use - deliberately not the model's full window. This exists because of a real, confirmed failure, not a theoretical concern: `devstral-small-2:24b` reports a nominal context of 393,216 tokens, but its own metadata (`mistral3.rope.scaling.original_context_length`) shows it was actually trained on 8,192 tokens - the rest is RoPE-scaling extrapolation. Trusting the nominal number sized a single chunk to the model's *entire* 393K window, which fit an entire real 200K-character document into one prime call - and that call failed outright (Ollama became unreachable and the model was no longer resident afterward), not just "produced a worse answer." Overriding `--context-length` to the model's real trained size (8192) restored normal map-reduce behavior immediately.

### A real, and counter-intuitive, chunk-size-vs-accuracy result

Tested a genuinely compound question ("What energy resources are described, how are they connected to the manufacturing industries discussed, and what role does the transport network play in connecting them?") against the same real 10-file NCERT document, varying only `--map-reduce-context-fraction`:

| Model | Fraction | Chunks | Answer quality (real output) |
|---|---|---|---|
| `qwen3:0.6b` | 0.25 (narrow) | 5 | Vague, generic, ends with a confusing stray "NONE" - "energy resources such as minerals and energy sources related to manufacturing... All relevant points are addressed. **Answer**: NONE." |
| `qwen3:0.6b` | 0.75 (wide) | 2 | Specific, coherent, correct - "coal, iron ore, and cement... railways, pipelines, highways... steel, cement, aluminum... critical for... economic growth and regional integration." |
| `qwen2.5:3b` | 0.25 (narrow) | 7 | A false negative - "does not contain information about energy resources... does not mention... ties to manufacturing industries" (the document does cover this; the model just couldn't see it from fragmented chunks) |

The starting hypothesis going into this (`max_context // 4`, narrower is safer) was **wrong for this question shape**. Narrower chunks fragmented the document across more, smaller pieces - for a compound question spanning three related concepts, a single chunk was less likely to contain enough of the connective material to answer well, regardless of model size (the same narrow-fraction failure mode showed up in both the 0.6B *and* 3B models). Wider chunks let even the smallest model tested see enough surrounding context to synthesize a correct, coherent answer.

**Honest scope of this finding**: one real question, one real document, two model sizes fully compared (a third, 24B, was attempted but proved impractically slow on this hardware even at a realistic context size - see below) - a real, directionally useful signal, not a statistically validated default. The likely explanation (compound/multi-concept questions need chunks that can hold multiple related concepts together; narrow, single-concept-sized chunks may be fine or even better for narrow fact-lookup questions) is itself untested and worth checking before changing the shipped default. Tracked for further, broader validation in [issue #24](https://github.com/no-mans-code/docuzent/issues/24).

**A separate, real finding about model size and hardware**: `devstral-small-2:24b`, even at its realistic 8192-token context, took long enough on this hardware (a 16GB GPU, meaning a 24B model spills significantly into system RAM) that a single real map-reduce question became impractical for interactive use - real evidence that "bigger model" isn't a free win for local, latency-sensitive use without matching hardware, independent of the context-size question above.

### A real bug this same testing surfaced: reasoning models can return an empty answer

`qwen3:0.6b` is a "thinking"-capable model - Ollama reports its internal reasoning in a separate `thinking` field, distinct from `response` (the visible answer). Every call is capped at a fixed generation budget (`num_predict`, 1536 tokens) so a broad question can't run unbounded - but a real compound question showed the model could spend its *entire* budget reasoning and never reach a visible answer at all: `response` came back empty while `thinking` held a real, if incomplete, trace. `GenerateResponse::text()` now falls back to a labeled excerpt of `thinking` when `response` is empty but the model produced something - never a silently blank answer when the model genuinely said something.

---

## Text-only ingestion (skip Docling for plain text/code)

`Session::load_text_files` (CLI: `docuzent ask <files> --text-only`) reads each file's raw UTF-8 content directly and never calls Docling - for `.txt`, `.md`, `.log`, config files, and source code, which are already text and don't need Docling's document-structure parsing. It shares every downstream mechanism with the normal Docling path: `.zip` extraction, the order-independent combined content hash (so the same set of files always keys identically regardless of load order), chunking, and the on-disk LLM-context cache/eviction. Only the ingestion front end differs - no syntax-aware extraction (functions, symbols) happens here, that's left to whatever consumes this mode. See [issue #27](https://github.com/no-mans-code/docuzent/issues/27).

---

## MCP server: document Q&A as a tool for coding agents

`docuzent-mcp` exposes the same `Session`-backed Q&A as an MCP tool over stdio, so a coding agent (Claude Code, etc.) can delegate "read this and answer a specific question" to a local Ollama model instead of reading the whole file into its own context - only the answer text crosses back. A second question about the same document set is typically far cheaper than the first: the server keeps one `Session` (`Mode::Adaptive`) alive for its whole process lifetime and reuses the on-disk context cache instead of reprocessing from scratch, and that reuse survives across server restarts too (it's the same disk cache `ask`/`bench` use).

```bash
cargo build --release -p docuzent-mcp
claude mcp add --transport stdio docuzent -- ./target/release/docuzent-mcp
```

Configured via environment variables (an MCP client launches the server directly, with no natural place for CLI flags): `DOCUZENT_MODEL` (default `qwen2.5:3b`), `DOCUZENT_HOST` (default `http://localhost:11434`), `DOCUZENT_CACHE`/`DOCUZENT_DOCLING_CACHE` (default under `.docuzent-cache/`), `DOCUZENT_MAP_REDUCE_CONTEXT_FRACTION`, `DOCUZENT_CONTEXT_LENGTH` (override the VRAM-aware safe default - see "VRAM-aware context sizing" below), `DOCUZENT_MAX_CONCURRENT_REQUESTS` (default `1` - see "Concurrent requests" below).

The one tool, **`ask_document(paths: string[], question: string, text_only?: bool)`**:
- `paths` - one or more files (or `.zip`s of them) loaded as a single combined corpus.
- `question` - must be real and specific; a bare `"summarize"` or an empty string is rejected outright, since an unfocused ask defeats question-aware map-reduce chunking and risks silently dropping whatever detail the caller actually needed.
- `text_only` - skip Docling for plain-text/source files (see above); needed before code files can go through this server sensibly.

The response reports:
- `cache_tier` - `"cold"`, `"adaptive-reuse"`, or `"adaptive-reuse-partial (n/m chunks)"` for a map-reduced document - so the calling agent can see the real cost characteristics rather than guessing from latency. Verified for real: a fresh ~67KB text file's first question comes back `"cold"`; the exact same file, in a brand-new server process, on its next question comes back `"adaptive-reuse"` - genuine cross-process disk-cache reuse, not a cosmetic label. See [issue #28](https://github.com/no-mans-code/docuzent/issues/28).
- `ram_offload_warning` - ground truth from Ollama's own `/api/ps`, present only when the model is measurably spilling into system RAM (see "VRAM-aware context sizing" below).
- `queued_ms` - how long this specific call actually waited for a free slot behind other in-flight calls (see "Concurrent requests" below).

### A Docling-free image, for MCP-only use

`Dockerfile.mcp` builds `docuzent-mcp` (and `docuzent-cli`) with **no Python, no Docling, no torch at all** - just the Rust workspace. Real, measured: 149MB and a ~32s build, vs. the main `Dockerfile`'s multi-GB Python/torch runtime. It cannot ingest real documents (PDF/DOCX/scans - that's Docling-only), but everything through `text_only: true` works identically, since that path never touches Docling. For an agent that only ever asks about source code or plain text, this is the image to use.

```bash
docker build -f Dockerfile.mcp -t docuzent-mcp .
docker run -i --rm -v docuzent-mcp-cache:/data -e DOCUZENT_MODEL=qwen2.5:3b docuzent-mcp
# wire into an MCP client, e.g.:
claude mcp add --transport stdio docuzent -- docker run -i --rm -v docuzent-mcp-cache:/data docuzent-mcp
```

`host.docker.internal` (Docker Desktop's DNS name for the host machine) is the default `DOCUZENT_HOST`, matching the main `Dockerfile`'s convention - override with `-e DOCUZENT_HOST=...` on Linux Docker. Verified for real: built the image, ran it against a mounted file and the host's real Ollama, got a correct answer with `chars_loaded` matching the raw file exactly. See [issue #31](https://github.com/no-mans-code/docuzent/issues/31).

---

## VRAM-aware context sizing (`docuzent_core::vram`)

The default context length is no longer just "whatever `/api/show` reports as the model's nominal window" - that number can be actively unsafe to use as-is, for two independent reasons this module accounts for:

1. **RoPE-scaling extrapolation.** `devstral-small-2:24b` reports a nominal 393,216-token context, but its own metadata (`mistral3.rope.scaling.original_context_length`) shows it was actually trained on 8,192 - the rest is extrapolation. Trusting the nominal number crashed Ollama outright in earlier testing (see the map-reduce section above). The safe default is now capped at the model's real trained context when one is reported.
2. **VRAM.** Real KV-cache size is computed from the model's actual GGUF architecture metadata - `block_count`, `attention.head_count_kv`, and, since it matters, `attention.key_length` directly rather than the derived `embedding_length / head_count` approximation (`ollama_kv_profiler::ollama::Client::architecture_info` falls back to that approximation, and it's a measurable **25% underestimate for devstral/mistral3** - real 128 vs. derived 160 - and a **2.9x underestimate for gemma4** - real 512 vs. derived 176; `docuzent_core::model_info::ModelInfo::attention_key_length` reads the real field directly when the model reports it). Combined with the model's real on-disk weight size and real free VRAM (`nvidia-smi`, NVIDIA-only for now), the default context length is capped to whatever actually fits within 85% of free VRAM.

Real, live result on this project's own dev machine (an RTX 5080, 16GB): `devstral-small-2:24b`'s ~15.2GB of weights alone exceed the safety-margin budget at the time of testing - the safe default correctly floors at 4,096 tokens (map-reduce then handles anything larger) rather than claiming the model's real 8,192-token trained context is safe when, on this exact card, it measurably wasn't. This matches the earlier, separately-documented finding that a 24B model on a 16GB card spills into system RAM even at a "realistic" context size.

**Ground truth, not just prediction**: after a real call, `docuzent_core::vram::real_vram_fraction` reads Ollama's own `/api/ps` for the fraction of the model actually resident in VRAM right now - surfaced as `ram_offload_warning` in `docuzent-web`'s `/model-info`/`/ask` responses and `docuzent-mcp`'s `ask_document` response whenever a model is measurably spilling to system RAM. An explicit override (`--context-length` / the web UI's slider / `DOCUZENT_CONTEXT_LENGTH`) always bypasses every check here - this only changes what happens with no override given. See [issue #30](https://github.com/no-mans-code/docuzent/issues/30).

### Context-size slider (web UI)

The context-size field in the web UI is a slider, not a fixed number - bounded by the model's real nominal window, defaulting to the safe value above, with a live red/green estimate (using the exact same formula the backend uses, via `GET /context-estimate?model=...`) as it's dragged: model weight size + KV-cache-at-this-size + a fixed overhead margin, compared against real free VRAM. Dragging it past the safe boundary shows the real estimated GB and turns the slider red before you even click Apply. See [issue #29](https://github.com/no-mans-code/docuzent/issues/29).

### Ingestion progress: a real estimate, honestly labeled

Ollama's API exposes no live prefill-progress signal - there is no partial-progress event during prompt evaluation. `GET /progress` (polled by the web UI while `/ask` is in flight) computes an *estimate*: `elapsed time × this session's own real measured prefill_tokens_per_sec` (from `docuzent_core::speed_profile::SpeedProfile` - real wall-clock timing from prior calls, so it already reflects any real slowdown from RAM offload, not a theoretical rate), clamped to the document's estimated total token count. The first-ever call for a `(model, context length)` pair has no measured rate yet - the UI shows an explicit indeterminate state then, never a fabricated number.

---

## Concurrent requests: explicit, fair, and observable (not yet parallel)

Traced directly from the code: `docuzent-web` and `docuzent-mcp` each hold exactly one active `Session` (one document, one model) behind a lock - by design, since loading a new document evicts the previous one. Before this, concurrent requests just contended for that lock with no fairness guarantee, no configurable limit, and no visibility into how long a request actually waited.

Now: a `tokio::sync::Semaphore`-based admission gate (`--max-concurrent-requests` / `DOCUZENT_MAX_CONCURRENT_REQUESTS`, default `1`) makes this explicit and FIFO-fair, and every response reports real, measured `queued_ms` - how long *this* request waited, not a guess. Real, verified: two overlapping `ask_document` calls against the same MCP server showed one return with `queued_ms: 0` and the other with `queued_ms: 7203` - a genuine multi-second wait behind the first call's real Ollama round trip, not a fabricated number.

**This does not add real parallelism** - raising the limit above 1 without a real multi-session pool just means concurrent callers stomp on the same document (repeated cache-eviction thrashing from `Session::load_documents`'s "evict the previous document" behavior), so the default stays 1. A real multi-session pool - admission-controlled by the VRAM estimator above, so as many models as actually fit in free VRAM run truly in parallel - is designed but not yet built; not pub/sub (that's a fan-out pattern, not a request/response admission-control one) but a bounded worker pool behind this same fair queue. See [issue #32](https://github.com/no-mans-code/docuzent/issues/32).

---

## Docling Setup

Docling (the ingestion engine) is a Python package with no native Rust bindings, so `docuzent` shells out to its CLI. Set up a local venv once:

```bash
py -m venv .venv
./.venv/Scripts/pip install -r requirements.txt
```

`docuzent` auto-detects `.venv/Scripts/docling.exe` when run from the repo root or a workspace member directory. Override with `--docling-bin <path>` or the `DOCLING_BIN` env var if needed.

```bash
cargo run -p docuzent-cli -- ingest temp-test --output out --to json
```

---

## Configuration

Create a `config.toml` in the project root or pass a path with `--config`:

```toml
# Max disk space for KV cache (GB)
cache_size_gb = 100

# Hierarchy levels (bytes)
chunk_sizes = [1024, 10000, 20000, 50000]

# Default model endpoint
model = "ollama:llama3.2"

# Ollama API URL
ollama_host = "http://localhost:11434"

# Stats UI port (if enabled)
stats_port = 8080
```

---

## Build & Release

```bash
cargo build --release
```

Binaries will be in `target/release/`.

---

## Documentation

- Architecture diagram: `docs/architecture.mermaid`
- API reference: `docs/api.md`
- Contribution guide: `CONTRIBUTING.md`

---

## Contributing

Pull requests are welcome! Please run `cargo fmt && cargo clippy` before submitting.

---

## License

MIT © 2026 no‑mans‑code
