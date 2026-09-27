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

## Workspace layout

`doc-intelligence` is a Cargo workspace:
- `docuzent-core/` - lib: Docling wrapper (`ingest`), page-aware chunker (`chunk`), embedder trait + Ollama impl (`embed`), storage (`store`), ingest orchestration (`pipeline`), and the single-document Q&A pieces - `hash`, `archive`, `model_info`, `generate` (Ollama `/api/generate` + its resumable `context`), `session` (the fits-or-map-reduce orchestrator).
- `docuzent-cli/` - bin `docuzent`: subcommands `ingest` (Docling conversion), `ask` (single-document Q&A, interactive or `--question`), `bench` (measures the context cache's real effect).
- `docuzent-web/` - planned: axum + askama + SSE UI, not yet built.

Two pieces are **not** in this workspace - separate, reusable crates/repos, since each is meant for other projects too, not just this one:
- `no-mans-code/atomiser` - query/task decomposition (one-shot plan graph + stack-based/ADaPT-style iterative). Not yet wired into docuzent-core; will be once the paused RAG mode's multi-hop query answering resumes.
- `no-mans-code/kvcache` - the generic on-disk LRU byte-blob cache backing the single-document Q&A mode's context persistence.

## Component Details

| Component       | Responsibility                                      | Key Parameters                                     |
|-----------------|-----------------------------------------------------|----------------------------------------------------|
| **Ingestion**   | PDF, DOCX, Markdown, Text, OCR - via Docling CLI, shelled out to from Rust (validated against `temp-test/`) | `--device` (`auto`|`cpu`|`cuda`|`mps`|`xpu`, default `auto` - GPU used when present, `--no-gpu` forces CPU) |
| **Chunker**     | Groups Docling's `texts[]` by `prov[].page_no` into one big chunk per page (splitting only if a page overflows the cap) - "big chunks that fit in memory," not small token windows | `max_chars` (default 4000) |
| **Embedder**    | `Embedder` trait + `OllamaEmbedder` (`nomic-embed-text` on `localhost:11434` by default) | `host`, `model` |
| **Store**       | Embedded `redb` (pure Rust, no C toolchain needed - this machine has none) holding chunks (text + embedding + page), per-document mtime/size (skips unchanged files), and a corpus version counter bumped only on real change | db path |
| **Retrieval**   | Brute-force cosine similarity over every stored chunk - adequate at local-folder scale, and this project is accuracy-focused, not latency-focused | `top_k` |
| **Atomiser**    | Classifies/decomposes a query into a dependency-ordered sub-query graph when it is multi-hop - separate crate, see [Query Atomisation](#query-atomisation--multi-hop-resolution) below | see `no-mans-code/atomiser` |
| **Map-reduce (RAG mode)** | Not yet built - question-aware map (extract cited facts per chunk) + reduce (merge, recursively if still oversized), for the multi-document corpus mode | - |
| **Session / map-reduce (simple mode)** | `docuzent-core::session::Session` - fits-in-one-chunk-or-map-reduce for one document, question-aware, built and verified. See [Single-Document Q&A](#single-document-qa-simple-mode) below | `--model`, `--host`, `--cache` |
| **Context cache** | `no-mans-code/kvcache` (generic on-disk LRU) persisting Ollama's `context` per `{model}\|{context_length}\|{doc_hash}` - correct and disk-verified, but see the honesty note below on what it does and doesn't actually speed up | `--cache`, capacity default `min(disk/2, 50GB)` |

## Query Atomisation & Multi-Hop Resolution

Some queries are compound: "what is the capital of ABC, and what is its history?" requires resolving "capital of ABC → XYZ" before "history of XYZ" can even be asked. Ported from a proven design in `flowera`'s `internal/atomizer` (Go), reimplemented in Rust with RAG-shaped step types in place of its tool tags.

**Three layers, cost increasing, cheapest first:**

1. **Classify** — deterministic regex/pattern matchers tag a query (or sub-query) with a type (`fact-lookup`, `history`, `comparison`, `summary`, …). No model call.
2. **Triage** — deterministic: is the whole query single-hop (one retrieval, one answer) or compound (needs decomposition)? No model call for the common case.
3. **Decompose** — only reached for compound queries; one model call writes a numbered sub-query graph, each node optionally declaring `needs: [ids]` on earlier nodes. The resolver runs nodes in dependency order, substituting earlier answers into later sub-queries before retrieval. Same shape as flowera's `Plan`/`Step`/`Needs` graph, and matches the published "Self-Ask" / "IRCoT" pattern for multi-hop RAG (interleave retrieval with reasoning, reuse each answer in the next lookup).

**Determinism** — the requirement is: *the same query returns the same result unless the corpus changed*. Two of the three layers above are already deterministic (pure pattern matching). The one exception, the Decompose model call, is made deterministic in practice by:
- Running it at temperature 0 (not flowera's 0.15 — full reproducibility matters more here than occasional better phrasing).
- Caching the full pipeline output — decomposition graph, retrieved chunk IDs, and final answer — keyed on `(exact query text, corpus version hash)`. An unchanged corpus + identical query is a cache hit: no model call, no retrieval, bit-identical answer. Ingesting or removing a document bumps the corpus hash and invalidates affected entries.

**Context-budget adaptive retrieval** — before ranking, compare the token size of the full candidate corpus (or corpus subset for the query's topic) against the target model's context window minus prompt/chat overhead:
- **Fits** → skip ranking, stuff the full text into context ("full-context mode"). Example: 5 history books that all fit → load them all.
- **Doesn't fit** → fall back to relevance-ranked chunk retrieval bounded by the remaining token budget.

This mirrors recent "adaptive"/"self-routing" RAG work that chooses between full-context and retrieval based on a size check rather than always retrieving.

## Single-Document Q&A ("simple mode")

The full folder-watching RAG system above is on hold while this simpler, parallel feature is built: upload one document (or a `.zip` of them), ask it questions, done. No corpus, no vector store - just Docling + Ollama + a real cache.

**Flow:**
1. **Model size inference** - `docuzent-core::model_info` queries Ollama's `/api/show` for the chosen model and reads its real context window from `model_info["<family>.context_length"]` (the key's family prefix varies - `qwen2.*`, `llama.*`, etc. - so it's found by suffix, not hardcoded per family). No manual context-size configuration needed.
2. **Upload** - any file Docling parses (PDF, DOCX, images via OCR, **audio/video via Docling's own ASR pipeline**), or a `.zip`. A zip is extracted exactly **one level deep** (`docuzent-core::archive::extract_one_level`) - a nested zip inside it is written to disk as plain bytes but never opened as an archive, which is the standard defense against a zip bomb without needing to track decompressed-size budgets.
3. **Fits-or-map-reduce** - the combined parsed text is compared against the model's real context window (minus a 20% reserve for prompt/question/answer overhead). Fits → one chunk, one prime call. Doesn't fit → question-aware map-reduce: each piece is asked "extract only what's relevant to this question" (or `NONE`), then a reduce call answers from the extracted facts alone.
4. **Context caching** - see below.

**On being honest about "KV cache":** Ollama's public API does not expose a model's real attention key/value tensors - there is no endpoint to dump or restore them. What it does expose is a `context` field: a plain token-id array `/api/generate` returns and accepts, meant for continuing a conversation without resending everything said so far.

**What the real measurements actually show** (both against `qwen2.5:3b`, context length 32768) - a materially mixed result, reported honestly rather than the rosier one-sided version an early test suggested:
- **Immediate reuse** (same process, sub-second gap, no other request in between): a follow-up call passing `context` back showed `prompt_eval_duration` roughly **10x lower** than a cold call re-sending the same ~2500-token document (0.30s down to 0.02s). Real, measured, large.
- **Cross-session reuse** (`docuzent bench`, a genuinely cold session followed by a fresh `Session`/`Cache::open` on the same file - the actual scenario disk persistence exists for): the "warm" run's answer call took **135ms** of `prompt_eval_duration` against 847 tokens; the cold run's prime call took **103ms** against 633 tokens - essentially identical throughput (6.3 vs 6.1 tokens/ms), meaning **no cache-skipping happened at all**. The ~30s gap (dominated by Docling re-parsing the document in the second session) was apparently enough for whatever internal state made the immediate case fast to be gone.

**Conclusion:** `context` reuse is real but short-lived - it rides on Ollama/llama.cpp's own internal, ephemeral, RAM-only slot cache, which this crate cannot extend, persist, or control. Persisting `context` to disk still round-trips correctly (verified: a fresh `Session` on the same cache file loads `warm_from_disk: true` and skips the priming call) and remains architecturally the right piece to have, but through Ollama's public HTTP API alone, it is not a reliable performance win across any real gap - only across a gap so short Ollama would likely have kept it warm anyway. A genuine cross-session speedup would need to bypass Ollama's HTTP layer entirely (llama.cpp's own `--prompt-cache`/session-slot file support, or a custom runtime with direct KV-tensor access) - a materially bigger effort, not attempted here.

**Two tiers** (`docuzent-core::session::Session`):
- **In memory**, for whichever one document is currently loaded. Loading a *different* document drops it immediately, freeing that memory - this process never holds more than one document's context resident at once.
- **On disk**, via the separate [`kvcache`](https://github.com/no-mans-code/kvcache) crate: a generic, size-bounded, LRU-evicted byte-blob cache (pure Rust, `redb`-backed). Keyed by `{model}|{context_length}|{document_sha256}`, so a different model or a re-pulled model with a different context length gets its own entry rather than colliding. Capacity defaults to `min(available disk space / 2, 50 GB)` (`kvcache::default_capacity_bytes`), tunable at program start. Verified to survive a fresh `Session`/`Cache::open` on the same file (standing in for a process restart) - the second session's first question skips priming entirely.

**Benchmark**: `docuzent bench <file>` runs a genuinely cold session (fresh cache file) then a fresh session reusing what the cold run persisted, and reports both runs' real `prompt_eval_duration` plus the measured speedup - not a claimed number, a computed one from that specific document and model.

## Assumptions & Constraints

- **Disk Space** – KV cache is capped at 100 GB; any additional data is pruned using LRU.
- **LLM Compatibility** – Works with Ollama endpoints; adapters can be added for OpenAI, Provider, etc.
- **Local Operation** – No external API keys or internet access required.
- **Rust 1.75+** – Required for async I/O and tokenization crates.

## Next Steps

* [x] Ingestion pipeline via Docling - see `docuzent-core::ingest`, validated against `temp-test/`.
* [x] Workspace restructure (`docuzent-core` lib + `docuzent-cli` bin).
* [x] Atomiser crate (`no-mans-code/atomiser`): classify + compound-detection + one-shot decompose, ported from flowera and generalized. Stack-based/ADaPT-style escalation tracked separately, not built yet.
* [x] Page-aware big-chunk chunker, Ollama embedder, `redb` store with mtime-based skip-if-unchanged and a corpus version counter - verified end-to-end (real Docling + real Ollama) against a `temp-test/` subset. (Folder-watching RAG mode: on hold.)
* [x] Atomiser: stack-based (ADaPT-style) iterative decomposition added alongside one-shot - `next_action` parser + `Stack`/`Frame`/`run` driver, 23/23 tests.
* [x] Single-document Q&A ("simple mode"): `hash`, `archive` (1-level zip), `model_info` (context-size inference via Ollama), `generate` (Ollama `/api/generate` + `context`), `session` (fits-or-map-reduce + 2-tier caching) - all in `docuzent-core`; `docuzent ask`/`docuzent bench` in `docuzent-cli`. New `kvcache` crate/repo for the generic on-disk LRU cache. Verified end-to-end with real Docling + real Ollama, including disk persistence surviving a fresh `Session`. **Honest finding**: cross-session context reuse showed no measurable speedup in the real `bench` run (see "On being honest about KV cache" above) - the mechanism is correct, the performance win is not guaranteed through Ollama's public API.
* [ ] Map-reduce answer engine for the multi-document RAG mode (question-aware map + recursive reduce, citations) - distinct from simple mode's own map-reduce, which is already built.
* [ ] Context-budget check routing between full-context and retrieval mode (RAG mode).
* [ ] Query/answer cache keyed on `(query, corpus version)` (RAG mode).
* [ ] Folder watcher (`notify` crate) wired to re-ingest on change (RAG mode).
* [ ] Web UI (`docuzent-web`: axum + askama + SSE) - would serve both modes.
