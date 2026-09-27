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

# Ask questions about a single document (any format Docling parses, or a .zip)
./target/release/docuzent ask ./report.pdf --model qwen2.5:3b

# Benchmark the on-disk context cache's real effect on one document
./target/release/docuzent bench ./report.pdf --model qwen2.5:3b
```

Requires a running local Ollama (`ollama serve`) with the chosen model pulled, and Docling set up per [Docling Setup](#docling-setup) below. The folder-watching, multi-document RAG mode described further down is on hold; `ask`/`bench` (single-document mode) are what's built and verified today.

---

## Features

| Feature | Description |
|---------|-------------|
| **Local ingestion** | Any format Docling parses - PDF, DOCX, Markdown, plain text, images (OCR), audio/video (ASR) - or a `.zip` of them, extracted one level deep. |
| **Model size inference** | Queries Ollama's `/api/show` for the real context window of whichever model you pick - no manual sizing. |
| **Fits-or-map-reduce** | A document that fits the model's context window is answered in one call; a bigger one is split and answered via question-aware map-reduce. |
| **On-disk context cache** | `no-mans-code/kvcache` - a generic, size-bounded, LRU-evicted cache, keyed by `{model}|{context length}|{document hash}`. Tunable capacity, default `min(disk/2, 50GB)`. |
| **Model agnostic** | Connects to any local LLM via the Ollama API. |
| **Benchmarked, not claimed** | `docuzent bench` measures the cache's real effect on your own machine and model - see "Measured Performance" below. |
| **Atomiser** | Separate, reusable crate (`no-mans-code/atomiser`) for query/task decomposition - not yet wired in here, built for the multi-document RAG mode once it resumes. |

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
