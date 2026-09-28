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
use docuzent_core::session::{infer_context_length, Mode, Session, DEFAULT_MAP_REDUCE_CONTEXT_FRACTION};
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
}

#[derive(Clone)]
struct DocuzentTools {
    session: Arc<Mutex<Session<OllamaClient>>>,
    // Read by #[tool_router]'s generated dispatch code, not by name in
    // this file - matches the upstream rmcp examples' own convention.
    #[allow(dead_code)]
    tool_router: ToolRouter<DocuzentTools>,
}

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
    fn ask_document(&self, Parameters(req): Parameters<AskDocumentRequest>) -> Result<CallToolResult, McpError> {
        if req.paths.is_empty() {
            return Ok(CallToolResult::error(vec![ContentBlock::text("`paths` must contain at least one file.")]));
        }
        if let Err(msg) = validate_question(&req.question) {
            return Ok(CallToolResult::error(vec![ContentBlock::text(msg)]));
        }

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

    tracing::info!(%model, %host, "starting docuzent-mcp");

    let context_length = infer_context_length(&host, &model)
        .with_context(|| format!("failed to infer context size for `{model}` - is `ollama serve` running and has `{model}` been pulled?"))?;

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

    let tools = DocuzentTools { session: Arc::new(Mutex::new(session)), tool_router: DocuzentTools::tool_router() };

    let service = tools.serve(stdio()).await.inspect_err(|e| {
        tracing::error!(?e, "serving error");
    })?;
    service.waiting().await?;
    Ok(())
}
