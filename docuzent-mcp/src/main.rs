//! An MCP server exposing docuzent's document Q&A as a tool for coding
//! agents (see https://github.com/no-mans-code/docuzent/issues/28).
//!
//! Why this exists: a coding agent reading a file directly pays for every
//! token of it in its own context, and pays again on every repeat
//! question. `ask_document` delegates "read this and answer a specific
//! question" to a local Ollama model instead - only the answer text
//! crosses back into the caller's context, and a second question about
//! the same document set is typically far cheaper than the first, because
//! this server keeps one [`Session`] alive for its whole process lifetime
//! and reuses its on-disk, per-document Ollama context cache
//! ([`docuzent_core::session::Mode::Adaptive`]) rather than cold-
//! reprocessing the document on every call.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use docuzent_core::docling_cache::DoclingCache;
use docuzent_core::generate::OllamaClient;
use docuzent_core::session::{Mode, Session, DEFAULT_MAP_REDUCE_CONTEXT_FRACTION};
use kvcache::Cache;
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::{tool, tool_handler, tool_router, transport::stdio, ErrorData as McpError, ServerHandler, ServiceExt};
use tracing_subscriber::EnvFilter;

/// Below this many words, a question is almost certainly too unfocused
/// for question-aware map-reduce chunking to do its job - see the
/// `#[tool(description = ...)]` on [`DocuzentTools::ask_document`] and
/// issue #28's "reject...an overly generic one (e.g. bare 'summarize')".
/// Not a rigorous NLP check, just a cheap guard against the specific
/// failure mode the issue names: a one- or two-word ask like "summarize"
/// or "explain this" gives map-reduce nothing to target, so a large
/// document's chunks each independently guess what's "relevant" and the
/// caller silently loses whatever the guesses disagreed on.
const MIN_QUESTION_WORDS: usize = 3;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct AskDocumentRequest {
    /// One or more local file paths to load as a single combined corpus
    /// (any format Docling can parse, or `.zip` archives of them,
    /// extracted one level deep). Multiple paths are combined, not
    /// answered separately.
    pub paths: Vec<String>,
    /// A real, specific question about the document(s) - e.g. "what does
    /// the `resolve_context` function do and what does it return?", not
    /// a bare "summarize". A generic ask defeats question-aware map-reduce
    /// chunking and risks silently dropping the detail you actually
    /// needed; see this tool's description.
    pub question: String,
    /// Set this for plain-text or source-code files (.txt, .md, .log,
    /// config files, source code) - skips Docling entirely and reads
    /// `paths` as raw UTF-8 text instead, which is both unnecessary and
    /// the wrong ingestion path for something that's already text. Leave
    /// false for real documents (PDFs, Office formats, scans).
    #[serde(default)]
    pub text_only: bool,
}

#[derive(Debug, serde::Serialize)]
struct AskDocumentResponse {
    answer: String,
    /// Which cache tier actually served this answer, so the calling
    /// agent can see the real cost characteristics rather than guessing
    /// from latency alone: "cold" (this content, at this chunk boundary,
    /// was reprocessed from scratch), "adaptive-reuse" (the on-disk
    /// Ollama context cache was reused - see
    /// [`docuzent_core::session::Mode::Adaptive`]), or, for a map-reduced
    /// document, "adaptive-reuse-partial" when only some chunks were
    /// cache-served.
    cache_tier: String,
    used_map_reduce: bool,
    chunks_mapped: usize,
    chars_loaded: usize,
    /// Ground truth from Ollama's own `/api/ps` (not an estimate): the
    /// fraction of the model currently sitting in VRAM, taken right after
    /// this call. `None` when it can't be measured (no NVIDIA GPU). Below
    /// ~0.98, the model is being partly served from system RAM - real,
    /// measured evidence of the slowdown this causes, not a guess. See
    /// https://github.com/no-mans-code/docuzent/issues/30.
    #[serde(skip_serializing_if = "Option::is_none")]
    ram_offload_warning: Option<String>,
    /// Real, measured milliseconds this call waited for a free slot
    /// behind `DOCUZENT_MAX_CONCURRENT_REQUESTS` other in-flight calls -
    /// `0` when uncontended. See
    /// https://github.com/no-mans-code/docuzent/issues/32.
    queued_ms: u64,
    /// Source file names in the loaded corpus - always present, so the
    /// calling agent always knows which files an answer was actually
    /// based on. See https://github.com/no-mans-code/docuzent/issues/34.
    source_files: Vec<String>,
    /// Per-chunk attribution for a map-reduced answer - which chunk(s)
    /// contributed a real extraction, and the extraction itself. Empty
    /// on the single-chunk path (see `source_files` instead).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    sources: Vec<docuzent_core::session::AnswerSource>,
}

#[derive(Clone)]
struct DocuzentTools {
    session: Arc<Mutex<Session<OllamaClient>>>,
    host: String,
    model: String,
    /// Fair (FIFO), explicit admission gate - see
    /// `DOCUZENT_MAX_CONCURRENT_REQUESTS` and
    /// https://github.com/no-mans-code/docuzent/issues/32. There is
    /// exactly one active document/model at a time by design (`session`
    /// above is one `Session`), so this defaults to 1 permit: it exists
    /// to make today's serialization fair and observable (real
    /// `queued_ms` per call), not to add real parallelism, which would
    /// need a real multi-session pool (tracked, not built yet).
    request_gate: Arc<tokio::sync::Semaphore>,
    // Read by #[tool_router]'s generated dispatch code, not by name in
    // this file - matches the upstream rmcp examples' own convention.
    #[allow(dead_code)]
    tool_router: ToolRouter<DocuzentTools>,
}

/// Below this fraction resident in VRAM, worth surfacing as a real,
/// measured warning rather than staying silent - small numerical slop
/// under 1.0 from measurement rounding shouldn't itself count as offload.
const RAM_OFFLOAD_WARNING_THRESHOLD: f64 = 0.97;

/// Rejects an empty or too-generic question. See [`MIN_QUESTION_WORDS`].
fn validate_question(question: &str) -> Result<(), String> {
    let trimmed = question.trim();
    if trimmed.is_empty() {
        return Err("`question` is empty - ask_document needs a real, specific question about the document(s), not just a load.".to_string());
    }
    if trimmed.split_whitespace().count() < MIN_QUESTION_WORDS {
        return Err(format!(
            "`{trimmed}` is too generic for question-aware chunking on a large document - ask about a specific fact, function, section, or comparison instead of a bare command like \"summarize\"."
        ));
    }
    Ok(())
}

#[tool_router]
impl DocuzentTools {
    #[tool(
        description = "Answer one specific, focused question about one or more local documents (PDFs/Office formats/etc. via Docling, plain-text or source code via text_only, or .zip archives) without pulling the document's full contents into your own context - only the answer text is returned. Requires a real, specific question (not a bare \"summarize\") - see the paths/question/text_only argument docs. A second question about the same document set is typically far cheaper than the first: this server keeps an on-disk, per-document Ollama context cache and reuses it when that's predicted to be faster (see cache_tier in the response) instead of reprocessing the document from scratch every time."
    )]
    async fn ask_document(&self, Parameters(req): Parameters<AskDocumentRequest>) -> Result<CallToolResult, McpError> {
        if req.paths.is_empty() {
            return Ok(CallToolResult::error(vec![ContentBlock::text("`paths` must contain at least one file.")]));
        }
        if let Err(msg) = validate_question(&req.question) {
            return Ok(CallToolResult::error(vec![ContentBlock::text(msg)]));
        }

        let queue_start = std::time::Instant::now();
        let _permit = self.request_gate.acquire().await.expect("request_gate semaphore is never closed");
        let queued_ms = queue_start.elapsed().as_millis() as u64;

        let paths: Vec<PathBuf> = req.paths.iter().map(PathBuf::from).collect();
        let mut session = self.session.lock().expect("session mutex poisoned by a prior panic");

        let load = if req.text_only { session.load_text_files(&paths) } else { session.load_documents(&paths) };
        let load = match load {
            Ok(l) => l,
            Err(e) => return Ok(CallToolResult::error(vec![ContentBlock::text(format!("failed to load document(s): {e:#}"))])),
        };

        let report = match session.ask(&req.question) {
            Ok(r) => r,
            Err(e) => return Ok(CallToolResult::error(vec![ContentBlock::text(format!("failed to answer: {e:#}"))])),
        };
        drop(session);

        let ram_offload_warning = docuzent_core::vram::real_vram_fraction(&self.host, &self.model).and_then(|fraction| {
            (fraction < RAM_OFFLOAD_WARNING_THRESHOLD).then(|| {
                format!(
                    "RAM OFFLOAD: only {:.0}% of `{}` is resident in VRAM right now (measured via Ollama's /api/ps) - the rest is running from system RAM, which is why this is slower than usual.",
                    fraction * 100.0,
                    self.model
                )
            })
        });

        let cache_tier = if report.used_map_reduce {
            let total = report.chunk_adaptive_decisions.len();
            let reused = report.chunk_adaptive_decisions.iter().filter(|d| d.chose_swap).count();
            match (total, reused) {
                (0, _) => "cold".to_string(),
                (t, r) if r == t => "adaptive-reuse".to_string(),
                (_, 0) => "cold".to_string(),
                (t, r) => format!("adaptive-reuse-partial ({r}/{t} chunks)"),
            }
        } else {
            match report.adaptive_decision {
                Some(d) if d.chose_swap => "adaptive-reuse".to_string(),
                _ => "cold".to_string(),
            }
        };

        let response = AskDocumentResponse {
            answer: report.answer,
            cache_tier,
            used_map_reduce: report.used_map_reduce,
            chunks_mapped: report.chunks_mapped,
            chars_loaded: load.chars,
            ram_offload_warning,
            queued_ms,
            source_files: report.source_files,
            sources: report.sources,
        };
        let text = serde_json::to_string_pretty(&response)
            .map_err(|e| McpError::internal_error(format!("failed to serialize response: {e}"), None))?;
        Ok(CallToolResult::success(vec![ContentBlock::text(text)]))
    }
}

#[tool_handler]
impl ServerHandler for DocuzentTools {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::from_build_env())
            .with_instructions(
                "Answers specific questions about local documents (or plain-text/code files, via text_only) using a local Ollama model, so you don't have to read the whole file into your own context. Ask a real, focused question - not a bare \"summarize\".",
            )
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .with_writer(std::io::stderr) // stdout is the MCP transport - never write logs there
        .with_ansi(false)
        .init();

    let model = env_or("DOCUZENT_MODEL", "qwen2.5:3b");
    let host = env_or("DOCUZENT_HOST", "http://localhost:11434");
    let cache_path = PathBuf::from(env_or("DOCUZENT_CACHE", ".docuzent-cache/context.redb"));
    let docling_cache_path = PathBuf::from(env_or("DOCUZENT_DOCLING_CACHE", ".docuzent-cache/docling.redb"));
    let map_reduce_context_fraction: f32 =
        env_or("DOCUZENT_MAP_REDUCE_CONTEXT_FRACTION", &DEFAULT_MAP_REDUCE_CONTEXT_FRACTION.to_string())
            .parse()
            .context("DOCUZENT_MAP_REDUCE_CONTEXT_FRACTION must be a number")?;
    // Defaults to 1: exactly one document/model is active at a time by
    // design (one `Session`), so raising this without a real multi-
    // session pool (tracked, not built yet - see
    // https://github.com/no-mans-code/docuzent/issues/32) just means
    // concurrent tool calls stomp on the same document. The gate exists
    // to make today's serialization fair (FIFO) and observable
    // (`queued_ms` on every response), not to add real parallelism.
    let max_concurrent_requests: usize = env_or("DOCUZENT_MAX_CONCURRENT_REQUESTS", "1")
        .parse()
        .context("DOCUZENT_MAX_CONCURRENT_REQUESTS must be a number")?;

    tracing::info!(%model, %host, "starting docuzent-mcp");

    let context_length_override: Option<u32> = match std::env::var("DOCUZENT_CONTEXT_LENGTH") {
        Ok(v) => Some(v.parse().context("DOCUZENT_CONTEXT_LENGTH must be a number")?),
        Err(_) => None,
    };
    // Defaults to a VRAM/trained-context-aware safe size rather than the
    // model's raw nominal window - see docuzent_core::vram's module docs
    // for the real, already-observed failure this prevents (a 24B model's
    // RoPE-extrapolated nominal context crashing Ollama outright).
    let (context_length, estimate) = docuzent_core::vram::resolve_context_length(&host, &model, context_length_override)
        .with_context(|| format!("failed to size context window for `{model}` - is `ollama serve` running and has `{model}` been pulled?"))?;
    tracing::info!(
        context_length,
        nominal_context_length = estimate.nominal_context_length,
        original_context_length = estimate.original_context_length,
        vram_free_bytes = estimate.vram_free_bytes,
        overridden = context_length_override.is_some(),
        "context window sized"
    );

    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let capacity = kvcache::default_capacity_bytes(&cache_path)?;
    let disk_cache = Cache::open(&cache_path, capacity)?;

    if let Some(parent) = docling_cache_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let docling_cache = DoclingCache::open(&docling_cache_path, docuzent_core::docling_cache::DEFAULT_CAPACITY_BYTES)?;

    let session = Session::open(
        OllamaClient::new(&host, &model, context_length),
        &model,
        context_length,
        std::env::temp_dir().join("docuzent-mcp"),
        Mode::Adaptive,
        map_reduce_context_fraction,
        disk_cache,
        docling_cache,
    )?;

    let tools = DocuzentTools {
        session: Arc::new(Mutex::new(session)),
        host: host.clone(),
        model: model.clone(),
        request_gate: Arc::new(tokio::sync::Semaphore::new(max_concurrent_requests.max(1))),
        tool_router: DocuzentTools::tool_router(),
    };

    let service = tools.serve(stdio()).await.inspect_err(|e| {
        tracing::error!(?e, "serving error");
    })?;
    service.waiting().await?;
    Ok(())
}
