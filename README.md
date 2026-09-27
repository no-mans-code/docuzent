# docuzent – Document-Intelligence at the Edge

A Rust-based local Retrieval-Augmented Generation (RAG) engine that ingests any file format, builds context-size hierarchies, and serves queries to LLMs (Ollama, GPT-4o, etc.) without leaving the machine.

---

## ?? Quick Start

```bash
# Install Rust if you don’t have it
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

---

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

---

## ?? Contributing

Pull requests are welcome! Please run `cargo fmt && cargo clippy` before submitting.

---

## ?? License

MIT © 2026 no-mans-code
