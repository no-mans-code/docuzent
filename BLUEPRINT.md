# docuzent Architecture Blueprint

## Overview

docuzent is a local Retrieval‑Augmented Generation (RAG) engine written in Rust. Its core responsibilities are:

1. **Ingest** – Read various document formats and extract raw text.
2. **Chunk** – Split text into overlapping token chunks.
3. **Hierarchy** – Build a multi‑level context stack (1 k → 10 k → 20 k → 50 k).
4. **KV Cache** – Store and retrieve pre‑processed chunks on disk with an LRU policy (max 100 GB).
5. **Query Resolver** – Compose the best‑fit chunk(s) for a user query and forward to a local LLM.
6. **Metrics** – Expose CLI & optional web UI for ingestion stats and cache hit‑rate.

## Architecture Diagram

```mermaid
graph TD
  A[Document Source] --> B[Ingestion & Extraction]
  B --> C[Chunking]
  C --> D[Hierarchy Builder]
  D --> E[KV Cache Layer]
  E --> F[Query Resolver]
  F --> G[LLM (Ollama / GPT‑4o)]
  G --> H[Response]
  E --> I[Metrics & Stats UI]
```

## Component Details

| Component       | Responsibility                                      | Key Parameters                                     |
|-----------------|-----------------------------------------------------|----------------------------------------------------|
| **Ingestion**   | PDF, DOCX, Markdown, Text, OCR (Tesseract)          | File type detection, OCR confidence threshold      |
| **Chunker**     | Token‑based chunking (OpenAI tokenizer) with 256‑token overlap | Chunk size limits per hierarchy level              |
| **Hierarchy Builder** | Builds context stack; chooses level based on model capability | `chunk_sizes` from config                         |
| **KV Cache**    | LRU eviction; 100 GB max; async write                | `cache_size_gb`, `lru_capacity`, disk path        |
| **Query Resolver** | Retrieves cache hits; fallback to next level; calls LLM | `model`, `ollama_host`, `llm_timeout`              |
| **Metrics**     | CLI counters, cache hit/miss, ingestion throughput   | `stats_port`, `enable_web_ui`                     |

## Assumptions & Constraints

- **Disk Space** – KV cache is capped at 100 GB; any additional data is pruned using LRU.
- **LLM Compatibility** – Works with Ollama endpoints; adapters can be added for OpenAI, Provider, etc.
- **Local Operation** – No external API keys or internet access required.
- **Rust 1.75+** – Required for async I/O and tokenization crates.

## Next Steps

* Implement ingestion pipeline (PDF, text).
* Add chunking with token counting.
* Build hierarchy and LRU cache.
* Wire up query resolver to Ollama.
* Add CLI & optional web UI.
