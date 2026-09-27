# docuzent � Document-Intelligence at the Edge

A Rust-based local Retrieval-Augmented Generation (RAG) engine that ingests any file format, builds context-size hierarchies, and serves queries to LLMs (Ollama, GPT-4o, etc.) without leaving the machine.

---

## ?? Quick Start

```bash
# Install Rust if you don�t have it
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone & build
git clone https://github.com/no-mans-code/docuzent.git
cd docuzent
cargo build --release

# Ingest a folder
./target/release/docuzent ingest ./data --config config.toml

# Query a document
./target/release/docuzent query "Explain vector embeddings." --model ollama:gpt4o
```

---

## ?? Features

| Feature | Description |
|---------|-------------|
| **Local ingestion** | Supports PDFs, DOCX, Markdown, plain text, images (OCR). |
| **Chunking** | Token-based chunking with overlap to preserve context. |
| **Hierarchical context** | Automatic fallback from small to large chunk sizes (1?k ? 10?k ? 20?k ? 50?k). |
| **On-disk KV cache** | LRU eviction, configurable disk quota (default 100?GB). |
| **Model agnostic** | Connects to any local LLM via Ollama API; adapters for others. |
| **Metrics & Stats** | CLI & optional web UI showing ingestion progress, cache hit/miss rates, document health. |
| **Extensible** | Plug-in system for new document types or LLM providers. |


## ?? Existing Tools Comparison

Below is a high‑level comparison of the most relevant open‑source tools that can help build a local Retrieval‑Augmented Generation (RAG) engine.  The table is intentionally concise; feel free to extend it as new projects emerge.

| Tool | Focus | Language | Key Features | Notes |
|------|-------|----------|--------------|-------|
| **LlamaIndex** (LangChain/ChatGLM) | RAG framework | Python | Document loaders, chunking, embeddings, retrieval, prompt templating | Great for rapid prototyping; heavy Python dependency
| **LangChain** | General LLM orchestration | Python | Multi‑step pipelines, embeddings, vector stores, memory | Mature, but Python‑centric
| **Qdrant** | Vector store | Rust/Go/JavaScript | Disk‑backed embeddings, filtering, search, API | Native Rust, good for Rust ecosystems
| **Milvus** | Vector database | Rust, C++, Python | Distributed, high‑scale, ANN search | Enterprise‑grade, heavier setup
| **Weaviate** | Vector + metadata store | Go | Schema‑based, GraphQL API | Easy to deploy, but heavier runtime
| **Ollama** | Local LLM + embeddings | Rust | Model hosting, embedding API, lightweight | Excellent for on‑device inference
| **SvelteKit + Pinecone** | Full stack | JavaScript | Frontend + vector DB | Good for quick web demos
| **Docling** | Document ingestion & transformation | Rust | Multi‑format ingestion, OCR, chunking, embeddings | The core of this project, used in the repository

---
## ?? Configuration

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


## ?? Build & Release

```bash
cargo build --release
```

Binaries will be in `target/release/`.
---

## ?? Documentation

- Architecture diagram: `docs/architecture.mermaid`
- API reference: `docs/api.md`
- Contribution guide: `CONTRIBUTING.md`


## ?? Contributing

Pull requests are welcome! Please run `cargo fmt && cargo clippy` before submitting.
---

## ?? License

MIT � 2026 no-mans-code
