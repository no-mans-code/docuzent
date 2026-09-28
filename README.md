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
| **Model size inference** | Queries Ollama's `/api/show` for the real context window of whichever model you pick - no manual sizing. |
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
