use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use tokio_stream::{Stream, StreamExt};

mod workspaces;
use workspaces::WorkspaceManager;
use clap::Parser;
use docuzent_core::docling_cache::DoclingCache;
use docuzent_core::generate::OllamaClient;
use docuzent_core::session::{Mode, Session};
use docuzent_core::vram;
use kvcache::Cache;
use serde::{Deserialize, Serialize};

/// axum's own default (2MB) is meant for typical JSON APIs, not multi-file
/// document uploads - a single real PDF, let alone several, routinely
/// exceeds it. This is generous rather than exact since it's the only
/// thing standing between a local, single-user tool and an oversized
/// upload, not a hard resource limit that needs tuning.
const MAX_UPLOAD_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// docuzent-web - a local browser UI for document Q&A. Entirely optional:
/// `docuzent-cli` is fully capable standalone without this running at all.
#[derive(Parser)]
struct Cli {
    /// Ollama model to start with - switchable at runtime from the UI
    /// (see `GET /models`, `POST /model`), this is only the initial pick
    #[arg(long, default_value = "qwen2.5:3b")]
    model: String,
    /// Ollama host
    #[arg(long, default_value = "http://localhost:11434")]
    host: String,
    /// How the LLM-context cache is used: swap, raw, or adaptive
    #[arg(long, default_value = "adaptive")]
    mode: String,
    /// Override the model's inferred context window (tokens)
    #[arg(long)]
    context_length: Option<u32>,
    /// Fraction of the model's context window a single map-reduce chunk
    /// is sized to use - see the README's "Adaptive Mode" section and
    /// https://github.com/no-mans-code/docuzent/issues/24
    #[arg(long, default_value_t = docuzent_core::session::DEFAULT_MAP_REDUCE_CONTEXT_FRACTION)]
    map_reduce_context_fraction: f32,
    /// Where the on-disk LLM-context cache lives
    #[arg(long, default_value = ".docuzent-cache/context.redb")]
    cache: PathBuf,
    /// Where the on-disk Docling parse cache lives
    #[arg(long, default_value = ".docuzent-cache/docling.redb")]
    docling_cache: PathBuf,
    /// Port to listen on
    #[arg(long, default_value_t = 3000)]
    port: u16,
    /// Maximum number of session requests (/load, /ask) allowed to run at
    /// once - a fair, FIFO-queued admission gate (`tokio::sync::Semaphore`),
    /// not an accident of mutex contention. Defaults to 1: there is
    /// exactly one active document/model at a time by design (loading a
    /// new document evicts the previous one - see `Session::load_documents`),
    /// so raising this without also building a real multi-session pool
    /// (tracked separately, not yet built - see
    /// https://github.com/no-mans-code/docuzent/issues/32) just means
    /// concurrent callers stomp on the same document: real thrashing
    /// (repeated cache eviction), not real parallelism.
    #[arg(long, default_value_t = 1)]
    max_concurrent_requests: usize,
}

/// Everything that changes together when the active model (or mode)
/// changes - held as one `Option` behind one lock so a switch can drop
/// the old `Session` (and its cache file handles) before opening new
/// ones on the same cache paths, rather than briefly holding two open
/// handles on the same redb file at once (which the CLI's own `bench`
/// command already avoids the same way - see its explicit `drop` before
/// reopening).
struct SessionState {
    session: Session<OllamaClient>,
    model: String,
    mode_label: String,
    context_length: u32,
    map_reduce_context_fraction: f32,
    /// The real VRAM picture behind `context_length` - see
    /// `docuzent_core::vram`. Kept alongside the session so `/model-info`
    /// can report it without a second round-trip to Ollama.
    vram_estimate: vram::ModelVramEstimate,
    /// The exact paths behind the last successful `/load` call - so
    /// "save this as a workspace" (see `workspaces.rs`) has something
    /// real to record, since `Session` itself only keeps base file names
    /// (`source_files`), not full paths.
    last_loaded_paths: Vec<PathBuf>,
    last_loaded_text_only: bool,
}

/// An *estimated* ingestion-progress snapshot, set right before a
/// potentially-slow prime call starts and cleared when it finishes - see
/// `GET /progress`. Real-data-driven (from `Session`'s own measured speed
/// profile), never fabricated: `tokens_per_sec` is `0.0` when nothing has
/// been measured yet for this `(model, context_length)`, in which case
/// the frontend shows an indeterminate state rather than a fake rate.
struct ProgressSnapshot {
    total_estimated_tokens: u64,
    tokens_per_sec: f64,
    started_at: std::time::Instant,
    label: String,
}

struct AppState {
    state: Mutex<Option<SessionState>>,
    progress: Mutex<Option<ProgressSnapshot>>,
    /// Fair (FIFO), explicit admission gate for `/load` and `/ask` - see
    /// `Cli::max_concurrent_requests` and
    /// https://github.com/no-mans-code/docuzent/issues/32. Replaces
    /// relying on `std::sync::Mutex` contention alone, which makes no
    /// fairness guarantee.
    request_gate: tokio::sync::Semaphore,
    host: String,
    cache_path: PathBuf,
    docling_cache_path: PathBuf,
    workspaces: WorkspaceManager,
}

/// Waits for a free slot in `gate`, returning the held permit alongside
/// how long this call actually waited - real, measured, not guessed. The
/// permit must be kept alive (bound to a variable, not `_`) until the
/// gated work is done.
async fn acquire_gate(gate: &tokio::sync::Semaphore) -> (tokio::sync::SemaphorePermit<'_>, u64) {
    let start = std::time::Instant::now();
    let permit = gate.acquire().await.expect("request_gate semaphore is never closed");
    (permit, start.elapsed().as_millis() as u64)
}

#[allow(clippy::too_many_arguments)]
fn open_session_state(
    host: &str,
    model: &str,
    mode_label: &str,
    context_length_override: Option<u32>,
    map_reduce_context_fraction: f32,
    cache_path: &PathBuf,
    docling_cache_path: &PathBuf,
) -> Result<SessionState> {
    let mode = Mode::parse(mode_label)?;
    // Defaults to a VRAM/trained-context-aware safe size, not the model's
    // raw nominal window - see docuzent_core::vram's module docs for the
    // real, already-observed failure this prevents (a 24B model's RoPE-
    // extrapolated nominal context crashing Ollama outright). An explicit
    // override (CLI flag, or the web UI's slider) always wins.
    let (context_length, vram_estimate) = vram::resolve_context_length(host, model, context_length_override)
        .with_context(|| format!("failed to size context window for `{model}` - is `ollama serve` running and has it been pulled?"))?;
    println!("Model `{model}` context window: {context_length} tokens (nominal {}, safe default {})", vram_estimate.nominal_context_length, vram_estimate.safe_context_length);

    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let capacity = kvcache::default_capacity_bytes(cache_path)?;
    let cache = Cache::open(cache_path, capacity)?;

    if let Some(parent) = docling_cache_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let docling_cache = DoclingCache::open(docling_cache_path, docuzent_core::docling_cache::DEFAULT_CAPACITY_BYTES)?;

    let upload_dir = std::env::temp_dir().join("docuzent-web-uploads");
    std::fs::create_dir_all(&upload_dir).ok();

    let session = Session::open(OllamaClient::new(host, model, context_length), model, context_length, upload_dir, mode, map_reduce_context_fraction, cache, docling_cache)?;
    Ok(SessionState {
        session,
        model: model.to_string(),
        mode_label: mode_label.to_string(),
        context_length,
        map_reduce_context_fraction,
        vram_estimate,
        last_loaded_paths: Vec::new(),
        last_loaded_text_only: false,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("Context cache: {} (capacity {:.1} GB)", cli.cache.display(), kvcache::default_capacity_bytes(&cli.cache)? as f64 / 1e9);
    println!("Docling parse cache: {} (capacity {:.1} GB)", cli.docling_cache.display(), docuzent_core::docling_cache::DEFAULT_CAPACITY_BYTES as f64 / 1e9);

    let session_state = open_session_state(&cli.host, &cli.model, &cli.mode, cli.context_length, cli.map_reduce_context_fraction, &cli.cache, &cli.docling_cache)?;

    let state = std::sync::Arc::new(AppState {
        state: Mutex::new(Some(session_state)),
        progress: Mutex::new(None),
        request_gate: tokio::sync::Semaphore::new(cli.max_concurrent_requests.max(1)),
        host: cli.host.clone(),
        cache_path: cli.cache.clone(),
        docling_cache_path: cli.docling_cache.clone(),
        workspaces: WorkspaceManager::new(cli.cache.parent().unwrap_or(std::path::Path::new(".docuzent-cache")).join("workspaces.json")),
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/model-info", get(model_info_handler))
        .route("/models", get(models_handler))
        .route("/context-estimate", get(context_estimate_handler))
        .route("/model", post(switch_model_handler))
        .route("/load", post(load_handler))
        .route("/ask", post(ask_handler))
        .route("/ask-stream", post(ask_stream_handler))
        .route("/progress", get(progress_handler))
        .route("/workspaces", get(list_workspaces_handler).post(save_workspace_handler))
        .route("/workspaces/load", post(load_workspace_handler))
        .route("/workspaces/delete", post(delete_workspace_handler))
        .layer(DefaultBodyLimit::max(MAX_UPLOAD_BYTES))
        .with_state(state);

    // 0.0.0.0, not 127.0.0.1 - loopback-only is invisible to Docker's
    // port forwarding (it connects to the container's external-facing
    // interface, not its loopback), which would otherwise accept the
    // container and the port mapping while every real request got an
    // empty reply. Binding all interfaces is the standard, safe choice
    // for a server that may run in a container - it doesn't change
    // anything about direct localhost access when run natively.
    let addr = format!("0.0.0.0:{}", cli.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.with_context(|| format!("failed to bind {addr}"))?;
    println!("Listening on http://localhost:{}", cli.port);
    axum::serve(listener, app).await?;

    Ok(())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

/// The VRAM picture for one model - shared shape across `/model-info`,
/// `POST /model`'s response, and `GET /context-estimate`, so the frontend
/// slider logic is the same regardless of which endpoint produced it (the
/// currently active session's real numbers, vs. a candidate model the
/// user is considering switching to).
#[derive(Serialize, Clone)]
struct VramFields {
    nominal_context_length: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    original_context_length: Option<u32>,
    safe_context_length: u32,
    weight_bytes: u64,
    kv_bytes_per_token: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    vram_free_bytes: Option<u64>,
    /// So the frontend's live slider estimate (`weight_bytes +
    /// kv_bytes_per_token * tokens + overhead_bytes`) matches the
    /// backend's own formula exactly, rather than a second hardcoded copy
    /// of the same constant drifting out of sync with it.
    overhead_bytes: u64,
    safety_fraction: f64,
}

impl From<&vram::ModelVramEstimate> for VramFields {
    fn from(e: &vram::ModelVramEstimate) -> Self {
        Self {
            nominal_context_length: e.nominal_context_length,
            original_context_length: e.original_context_length,
            safe_context_length: e.safe_context_length,
            weight_bytes: e.weight_bytes,
            kv_bytes_per_token: e.kv_bytes_per_token,
            vram_free_bytes: e.vram_free_bytes,
            overhead_bytes: vram::FIXED_OVERHEAD_BYTES,
            safety_fraction: vram::DEFAULT_VRAM_SAFETY_FRACTION,
        }
    }
}

/// Ground truth (not an estimate), from Ollama's own `/api/ps`: `None`
/// when it can't be measured, or when there's nothing to warn about.
fn ram_offload_warning(host: &str, model: &str) -> Option<String> {
    let fraction = vram::real_vram_fraction(host, model)?;
    (fraction < RAM_OFFLOAD_WARNING_THRESHOLD).then(|| {
        format!(
            "{:.0}% of `{model}` is resident in VRAM right now (measured, not estimated) - the rest is running from system RAM, which is why generation is slower than usual.",
            fraction * 100.0
        )
    })
}

/// Below this fraction resident in VRAM, worth surfacing as a real,
/// measured warning - small numerical slop under 1.0 from measurement
/// rounding shouldn't itself count as offload.
const RAM_OFFLOAD_WARNING_THRESHOLD: f64 = 0.97;

#[derive(Serialize)]
struct ModelInfoResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    map_reduce_context_fraction: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vram: Option<VramFields>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ram_offload_warning: Option<String>,
}

async fn model_info_handler(State(state): State<std::sync::Arc<AppState>>) -> impl IntoResponse {
    let (model, host, snapshot) = {
        let guard = state.state.lock().unwrap();
        match guard.as_ref() {
            Some(s) => (s.model.clone(), state.host.clone(), Some((s.mode_label.clone(), s.context_length, s.map_reduce_context_fraction, VramFields::from(&s.vram_estimate)))),
            None => (String::new(), state.host.clone(), None),
        }
    };
    let Some((mode_label, context_length, map_reduce_context_fraction, vram_fields)) = snapshot else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ModelInfoResponse { ok: false, error: Some("session is switching models right now".to_string()), model: None, mode: None, context_length: None, map_reduce_context_fraction: None, cache_path: None, vram: None, ram_offload_warning: None }),
        );
    };
    // A real, live /api/ps check - cheap and local, but still a blocking
    // HTTP call, so it's kept off the async runtime's worker thread.
    let model_for_warning = model.clone();
    let warning = tokio::task::spawn_blocking(move || ram_offload_warning(&host, &model_for_warning)).await.ok().flatten();
    (
        StatusCode::OK,
        Json(ModelInfoResponse {
            ok: true,
            error: None,
            model: Some(model),
            mode: Some(mode_label),
            context_length: Some(context_length),
            map_reduce_context_fraction: Some(map_reduce_context_fraction),
            cache_path: Some(state.cache_path.display().to_string()),
            vram: Some(vram_fields),
            ram_offload_warning: warning,
        }),
    )
}

#[derive(Serialize)]
struct ModelsResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    models: Option<Vec<String>>,
}

/// Lists this Ollama server's actually-pulled models, so the UI offers a
/// real choice rather than whatever was fixed at startup.
async fn models_handler(State(state): State<std::sync::Arc<AppState>>) -> impl IntoResponse {
    let host = state.host.clone();
    let result = tokio::task::spawn_blocking(move || docuzent_core::model_info::list_models(&host)).await;
    match result {
        Ok(Ok(models)) => (StatusCode::OK, Json(ModelsResponse { ok: true, error: None, models: Some(models) })),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ModelsResponse { ok: false, error: Some(e.to_string()), models: None })),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ModelsResponse { ok: false, error: Some(e.to_string()), models: None })),
    }
}

#[derive(Deserialize)]
struct ContextEstimateQuery {
    model: String,
}

#[derive(Serialize)]
struct ContextEstimateResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vram: Option<VramFields>,
}

/// The VRAM picture for a *candidate* model - independent of whichever
/// session is currently active, so the web UI's context-size slider can
/// show real bounds/estimates for a model the user is considering
/// switching to, before they click Apply. See
/// https://github.com/no-mans-code/docuzent/issues/29.
async fn context_estimate_handler(State(state): State<std::sync::Arc<AppState>>, axum::extract::Query(q): axum::extract::Query<ContextEstimateQuery>) -> impl IntoResponse {
    let host = state.host.clone();
    let model = q.model.clone();
    let result = tokio::task::spawn_blocking(move || vram::estimate_for_model(&host, &model)).await;
    match result {
        Ok(Ok(estimate)) => (StatusCode::OK, Json(ContextEstimateResponse { ok: true, error: None, model: Some(q.model), vram: Some(VramFields::from(&estimate)) })),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ContextEstimateResponse { ok: false, error: Some(e.to_string()), model: None, vram: None })),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ContextEstimateResponse { ok: false, error: Some(e.to_string()), model: None, vram: None })),
    }
}

#[derive(Serialize)]
struct ProgressResponse {
    active: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    estimated_tokens_total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    estimated_tokens_done: Option<u64>,
    /// `0.0` when nothing has been measured for this `(model,
    /// context_length)` yet - the frontend shows an indeterminate state
    /// rather than pretending to know a rate, per
    /// https://github.com/no-mans-code/docuzent/issues/30.
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_per_sec: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
}

/// Ollama's API exposes no live prefill-progress signal - this is an
/// *estimate*, computed from `elapsed × the session's own real measured
/// throughput` (see `Session::estimated_prefill_tokens_per_sec`), polled
/// by the frontend while a `/load` or `/ask` request that might be doing
/// a slow cold prime is in flight.
async fn progress_handler(State(state): State<std::sync::Arc<AppState>>) -> impl IntoResponse {
    let guard = state.progress.lock().unwrap();
    match guard.as_ref() {
        Some(p) => {
            let elapsed = p.started_at.elapsed().as_secs_f64();
            let estimated_tokens_done = if p.tokens_per_sec > 0.0 { ((elapsed * p.tokens_per_sec) as u64).min(p.total_estimated_tokens) } else { 0 };
            Json(ProgressResponse {
                active: true,
                estimated_tokens_total: Some(p.total_estimated_tokens),
                estimated_tokens_done: Some(estimated_tokens_done),
                tokens_per_sec: Some(p.tokens_per_sec),
                label: Some(p.label.clone()),
            })
        }
        None => Json(ProgressResponse { active: false, estimated_tokens_total: None, estimated_tokens_done: None, tokens_per_sec: None, label: None }),
    }
}

#[derive(Deserialize)]
struct SwitchModelRequest {
    model: String,
    #[serde(default)]
    mode: Option<String>,
    #[serde(default)]
    context_length: Option<u32>,
    #[serde(default)]
    map_reduce_context_fraction: Option<f32>,
}

#[derive(Serialize)]
struct SwitchModelResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    map_reduce_context_fraction: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vram: Option<VramFields>,
}

fn switch_error(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<SwitchModelResponse>) {
    (status, Json(SwitchModelResponse { ok: false, error: Some(message.into()), model: None, mode: None, context_length: None, map_reduce_context_fraction: None, vram: None }))
}

/// Switches the active model (and optionally mode/context length/
/// map-reduce fraction) at runtime - drops the old `Session` (and its
/// cache file handles) before opening new ones on the same cache paths,
/// so the on-disk caches carry over rather than starting fresh per model.
async fn switch_model_handler(State(state): State<std::sync::Arc<AppState>>, Json(req): Json<SwitchModelRequest>) -> impl IntoResponse {
    let (mode_label, previous_fraction) = {
        let guard = state.state.lock().unwrap();
        let previous = guard.as_ref();
        (
            req.mode.clone().or_else(|| previous.map(|s| s.mode_label.clone())).unwrap_or_else(|| "adaptive".to_string()),
            previous.map(|s| s.map_reduce_context_fraction).unwrap_or(docuzent_core::session::DEFAULT_MAP_REDUCE_CONTEXT_FRACTION),
        )
    };
    let map_reduce_context_fraction = req.map_reduce_context_fraction.unwrap_or(previous_fraction);

    // Drop the old session (releasing its cache file handles) before
    // opening new ones on the same paths - see `SessionState`'s doc comment.
    state.state.lock().unwrap().take();

    let host = state.host.clone();
    let cache_path = state.cache_path.clone();
    let docling_cache_path = state.docling_cache_path.clone();
    let model = req.model.clone();
    let result = tokio::task::spawn_blocking(move || open_session_state(&host, &model, &mode_label, req.context_length, map_reduce_context_fraction, &cache_path, &docling_cache_path)).await;

    match result {
        Ok(Ok(new_state)) => {
            let response = SwitchModelResponse {
                ok: true,
                error: None,
                model: Some(new_state.model.clone()),
                mode: Some(new_state.mode_label.clone()),
                context_length: Some(new_state.context_length),
                map_reduce_context_fraction: Some(new_state.map_reduce_context_fraction),
                vram: Some(VramFields::from(&new_state.vram_estimate)),
            };
            *state.state.lock().unwrap() = Some(new_state);
            (StatusCode::OK, Json(response))
        }
        Ok(Err(e)) => switch_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => switch_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Serialize)]
struct LoadResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chars: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fits_in_one_chunk: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warm_from_disk: Option<bool>,
    /// Real, measured milliseconds this request waited for a free slot
    /// behind `--max-concurrent-requests` other in-flight requests - `0`
    /// when uncontended. See
    /// https://github.com/no-mans-code/docuzent/issues/32.
    queued_ms: u64,
}

fn load_error(status: StatusCode, message: impl Into<String>, queued_ms: u64) -> (StatusCode, Json<LoadResponse>) {
    (status, Json(LoadResponse { ok: false, error: Some(message.into()), chars: None, fits_in_one_chunk: None, warm_from_disk: None, queued_ms }))
}

/// Accepts one or more files in a single multipart request, saving each
/// and loading them together as one combined corpus.
async fn load_handler(
    State(state): State<std::sync::Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let (_permit, queued_ms) = acquire_gate(&state.request_gate).await;
    let upload_dir = std::env::temp_dir().join("docuzent-web-uploads");
    std::fs::create_dir_all(&upload_dir).ok();

    let mut dests = Vec::new();
    let mut text_only = false;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return load_error(StatusCode::BAD_REQUEST, e.to_string(), queued_ms),
        };
        // A plain form field (no filename) named `text_only` toggles
        // skipping Docling entirely, for files that are already text
        // (.txt, .md, .log, config, source code) - see
        // https://github.com/no-mans-code/docuzent/issues/27.
        if field.name() == Some("text_only") && field.file_name().is_none() {
            let value = field.text().await.unwrap_or_default();
            text_only = value == "true" || value == "1" || value == "on";
            continue;
        }
        let original_name = field.file_name().unwrap_or("upload").to_string();
        let safe_name = sanitize_filename(&original_name);
        let bytes = match field.bytes().await {
            Ok(b) => b,
            Err(e) => return load_error(StatusCode::BAD_REQUEST, e.to_string(), queued_ms),
        };
        let dest = upload_dir.join(&safe_name);
        if let Err(e) = std::fs::write(&dest, &bytes) {
            return load_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), queued_ms);
        }
        dests.push(dest);
    }

    if dests.is_empty() {
        return load_error(StatusCode::BAD_REQUEST, "no file in request", queued_ms);
    }

    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            let mut guard = state.state.lock().unwrap();
            let Some(session_state) = guard.as_mut() else {
                anyhow::bail!("session is switching models right now - try again in a moment");
            };
            let result = if text_only { session_state.session.load_text_files(&dests) } else { session_state.session.load_documents(&dests) };
            if result.is_ok() {
                session_state.last_loaded_paths = dests;
                session_state.last_loaded_text_only = text_only;
            }
            result
        }
    })
    .await;

    match result {
        Ok(Ok(report)) => (
            StatusCode::OK,
            Json(LoadResponse {
                ok: true,
                error: None,
                chars: Some(report.chars),
                fits_in_one_chunk: Some(report.fits_in_one_chunk),
                warm_from_disk: Some(report.warm_from_disk),
                queued_ms,
            }),
        ),
        Ok(Err(e)) => load_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), queued_ms),
        Err(e) => load_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), queued_ms),
    }
}

#[derive(Deserialize)]
struct AskRequest {
    question: String,
}

#[derive(Serialize)]
struct AskResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    answer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    used_map_reduce: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chunks_mapped: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    adaptive_decision: Option<docuzent_core::session::AdaptiveDecision>,
    #[serde(skip_serializing_if = "Option::is_none")]
    timings: Option<Vec<docuzent_core::session::CallTiming>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ram_offload_warning: Option<String>,
    /// Real, measured milliseconds this request waited for a free slot
    /// behind `--max-concurrent-requests` other in-flight requests - `0`
    /// when uncontended. See
    /// https://github.com/no-mans-code/docuzent/issues/32.
    queued_ms: u64,
    /// Source file names in the loaded corpus - always present when
    /// `ok`. See https://github.com/no-mans-code/docuzent/issues/34.
    #[serde(skip_serializing_if = "Option::is_none")]
    source_files: Option<Vec<String>>,
    /// Per-chunk attribution for a map-reduced answer - empty on the
    /// single-chunk path (see `source_files` instead).
    #[serde(skip_serializing_if = "Option::is_none")]
    sources: Option<Vec<docuzent_core::session::AnswerSource>>,
}

/// The `AskResponse` shape for an error, shared by `/ask` and
/// `/ask-stream` so both build it identically.
fn ask_error_body(message: &str, queued_ms: u64) -> AskResponse {
    AskResponse {
        ok: false,
        error: Some(message.to_string()),
        answer: None,
        used_map_reduce: None,
        chunks_mapped: None,
        adaptive_decision: None,
        timings: None,
        ram_offload_warning: None,
        queued_ms,
        source_files: None,
        sources: None,
    }
}

fn ask_error(status: StatusCode, message: impl Into<String>, queued_ms: u64) -> (StatusCode, Json<AskResponse>) {
    (status, Json(ask_error_body(&message.into(), queued_ms)))
}

async fn ask_handler(State(state): State<std::sync::Arc<AppState>>, Json(req): Json<AskRequest>) -> impl IntoResponse {
    let (_permit, queued_ms) = acquire_gate(&state.request_gate).await;
    let model = {
        let guard = state.state.lock().unwrap();
        let Some(session_state) = guard.as_ref() else {
            return ask_error(StatusCode::SERVICE_UNAVAILABLE, "session is switching models right now - try again in a moment", queued_ms);
        };
        // An *estimate*, from this session's own real measured throughput
        // (0.0 if nothing measured yet for this model/context length) -
        // see `ProgressSnapshot`'s doc comment and
        // https://github.com/no-mans-code/docuzent/issues/30. Total is
        // whatever's currently loaded, not necessarily what this
        // particular question's map-reduce chunking will touch, but it's
        // the best estimate available before the real call starts.
        let total_estimated_tokens = (session_state.session.current_text_len_chars().unwrap_or(0) / docuzent_core::session::CHARS_PER_TOKEN).max(1) as u64;
        let tokens_per_sec = session_state.session.estimated_prefill_tokens_per_sec().unwrap_or(0.0);
        *state.progress.lock().unwrap() = Some(ProgressSnapshot { total_estimated_tokens, tokens_per_sec, started_at: std::time::Instant::now(), label: "answering".to_string() });
        session_state.model.clone()
    };
    let host = state.host.clone();

    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            let mut guard = state.state.lock().unwrap();
            let Some(session_state) = guard.as_mut() else {
                anyhow::bail!("session is switching models right now - try again in a moment");
            };
            session_state.session.ask(&req.question)
        }
    })
    .await;

    *state.progress.lock().unwrap() = None;

    match result {
        Ok(Ok(report)) => {
            let warning = tokio::task::spawn_blocking(move || ram_offload_warning(&host, &model)).await.ok().flatten();
            (
                StatusCode::OK,
                Json(AskResponse {
                    ok: true,
                    error: None,
                    answer: Some(report.answer),
                    used_map_reduce: Some(report.used_map_reduce),
                    chunks_mapped: Some(report.chunks_mapped),
                    adaptive_decision: report.adaptive_decision,
                    timings: Some(report.timings),
                    ram_offload_warning: warning,
                    queued_ms,
                    source_files: Some(report.source_files),
                    sources: Some(report.sources),
                }),
            )
        }
        Ok(Err(e)) => ask_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), queued_ms),
        Err(e) => ask_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), queued_ms),
    }
}

/// Same as `/ask`, but streams the final answer token-by-token over
/// Server-Sent Events as it's actually generated (Ollama's real
/// `stream: true`, not an estimate) instead of blocking until the whole
/// answer is ready - see
/// https://github.com/no-mans-code/docuzent/issues/35. Each event's
/// `data` is a JSON object: `{"token": "..."}` for a fragment, or the
/// full `AskResponse` shape (with `"ok"`/`"answer"`/etc.) as the final
/// event, distinguishable by the absence of a `"token"` key.
async fn ask_stream_handler(State(state): State<std::sync::Arc<AppState>>, Json(req): Json<AskRequest>) -> Sse<impl Stream<Item = Result<Event, std::convert::Infallible>>> {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();

    tokio::spawn(async move {
        let queue_start = std::time::Instant::now();
        let _permit = state.request_gate.acquire().await;
        let queued_ms = queue_start.elapsed().as_millis() as u64;

        let model = {
            let guard = state.state.lock().unwrap();
            let Some(session_state) = guard.as_ref() else {
                let _ = tx.send(serde_json::to_string(&ask_error_body("session is switching models right now - try again in a moment", queued_ms)).unwrap());
                return;
            };
            let total_estimated_tokens = (session_state.session.current_text_len_chars().unwrap_or(0) / docuzent_core::session::CHARS_PER_TOKEN).max(1) as u64;
            let tokens_per_sec = session_state.session.estimated_prefill_tokens_per_sec().unwrap_or(0.0);
            *state.progress.lock().unwrap() = Some(ProgressSnapshot { total_estimated_tokens, tokens_per_sec, started_at: std::time::Instant::now(), label: "answering".to_string() });
            session_state.model.clone()
        };
        let host = state.host.clone();

        let question = req.question.clone();
        let result = tokio::task::spawn_blocking({
            let state = state.clone();
            let tx = tx.clone();
            move || {
                let mut guard = state.state.lock().unwrap();
                let Some(session_state) = guard.as_mut() else {
                    anyhow::bail!("session is switching models right now - try again in a moment");
                };
                session_state.session.ask_streaming(&question, &mut |fragment| {
                    let _ = tx.send(serde_json::json!({ "token": fragment }).to_string());
                })
            }
        })
        .await;

        *state.progress.lock().unwrap() = None;

        let final_body = match result {
            Ok(Ok(report)) => {
                let warning = tokio::task::spawn_blocking(move || ram_offload_warning(&host, &model)).await.ok().flatten();
                serde_json::to_string(&AskResponse {
                    ok: true,
                    error: None,
                    answer: Some(report.answer),
                    used_map_reduce: Some(report.used_map_reduce),
                    chunks_mapped: Some(report.chunks_mapped),
                    adaptive_decision: report.adaptive_decision,
                    timings: Some(report.timings),
                    ram_offload_warning: warning,
                    queued_ms,
                    source_files: Some(report.source_files),
                    sources: Some(report.sources),
                })
            }
            Ok(Err(e)) => serde_json::to_string(&ask_error_body(&e.to_string(), queued_ms)),
            Err(e) => serde_json::to_string(&ask_error_body(&e.to_string(), queued_ms)),
        };
        let _ = tx.send(final_body.unwrap_or_else(|e| format!(r#"{{"ok":false,"error":"failed to serialize response: {e}"}}"#)));
        // `tx` drops here (its earlier clone already dropped when the
        // spawn_blocking closure returned), closing the channel and
        // ending the SSE stream.
    });

    let stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx).map(|body| Ok(Event::default().data(body)));
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Keeps just the base filename, stripped of any path components - the
/// only sanitization that matters when the result is joined onto a fixed
/// upload directory.
fn sanitize_filename(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    if base.is_empty() {
        "upload".to_string()
    } else {
        base.to_string()
    }
}

#[derive(Serialize)]
struct WorkspacesResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    workspaces: Option<std::collections::BTreeMap<String, workspaces::Workspace>>,
}

async fn list_workspaces_handler(State(state): State<std::sync::Arc<AppState>>) -> impl IntoResponse {
    match state.workspaces.list() {
        Ok(ws) => (StatusCode::OK, Json(WorkspacesResponse { ok: true, error: None, workspaces: Some(ws) })),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(WorkspacesResponse { ok: false, error: Some(e.to_string()), workspaces: None })),
    }
}

#[derive(Deserialize)]
struct SaveWorkspaceRequest {
    name: String,
}

#[derive(Serialize)]
struct SimpleResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

/// Saves whichever files were loaded by the *last successful* `/load`
/// call, along with the currently active model/mode/context length, as
/// a named workspace - a thin record over what already exists (see
/// `workspaces.rs`), not a copy of the document content itself.
async fn save_workspace_handler(State(state): State<std::sync::Arc<AppState>>, Json(req): Json<SaveWorkspaceRequest>) -> impl IntoResponse {
    let Some(name) = workspaces::sanitize_name(&req.name) else {
        return (StatusCode::BAD_REQUEST, Json(SimpleResponse { ok: false, error: Some("workspace name is empty or invalid".to_string()) }));
    };
    let guard = state.state.lock().unwrap();
    let Some(session_state) = guard.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(SimpleResponse { ok: false, error: Some("session is switching models right now".to_string()) }));
    };
    if session_state.last_loaded_paths.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(SimpleResponse { ok: false, error: Some("load a document first, then save it as a workspace".to_string()) }));
    }
    let workspace = workspaces::Workspace {
        paths: session_state.last_loaded_paths.iter().map(|p| p.display().to_string()).collect(),
        text_only: session_state.last_loaded_text_only,
        model: session_state.model.clone(),
        mode: session_state.mode_label.clone(),
        context_length: session_state.context_length,
        map_reduce_context_fraction: session_state.map_reduce_context_fraction,
        updated_at: chrono_now_rfc3339(),
    };
    drop(guard);
    match state.workspaces.save_workspace(&name, workspace) {
        Ok(()) => (StatusCode::OK, Json(SimpleResponse { ok: true, error: None })),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(SimpleResponse { ok: false, error: Some(e.to_string()) })),
    }
}

/// A minimal RFC 3339 timestamp without pulling in a `chrono`/`time`
/// dependency just for this - good enough for "when was this last
/// saved," which is display-only, never parsed back for logic.
fn chrono_now_rfc3339() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    format!("unix:{secs}")
}

#[derive(Deserialize)]
struct LoadWorkspaceRequest {
    name: String,
}

#[derive(Serialize)]
struct LoadWorkspaceResponse {
    ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chars: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fits_in_one_chunk: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    warm_from_disk: Option<bool>,
}

/// Reloads a saved workspace's files into the *currently active* session
/// - deliberately does not also switch model/mode (that stays a
/// separate, explicit `POST /model` call), keeping this a thin reload,
/// not a hidden model switch. If the on-disk context cache still has
/// this exact document (likely, since nothing evicts it on a timer),
/// this is a real cache hit, not just upload convenience.
async fn load_workspace_handler(State(state): State<std::sync::Arc<AppState>>, Json(req): Json<LoadWorkspaceRequest>) -> impl IntoResponse {
    let workspace = match state.workspaces.get(&req.name) {
        Ok(Some(w)) => w,
        Ok(None) => return (StatusCode::NOT_FOUND, Json(LoadWorkspaceResponse { ok: false, error: Some(format!("no workspace named `{}`", req.name)), chars: None, fits_in_one_chunk: None, warm_from_disk: None })),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(LoadWorkspaceResponse { ok: false, error: Some(e.to_string()), chars: None, fits_in_one_chunk: None, warm_from_disk: None })),
    };
    let paths: Vec<PathBuf> = workspace.paths.iter().map(PathBuf::from).collect();
    let missing: Vec<&PathBuf> = paths.iter().filter(|p| !p.exists()).collect();
    if !missing.is_empty() {
        let names = missing.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ");
        return (
            StatusCode::GONE,
            Json(LoadWorkspaceResponse { ok: false, error: Some(format!("workspace file(s) no longer exist: {names} - they may have been cleared from the upload temp directory")), chars: None, fits_in_one_chunk: None, warm_from_disk: None }),
        );
    }

    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            let mut guard = state.state.lock().unwrap();
            let Some(session_state) = guard.as_mut() else {
                anyhow::bail!("session is switching models right now - try again in a moment");
            };
            let result = if workspace.text_only { session_state.session.load_text_files(&paths) } else { session_state.session.load_documents(&paths) };
            if result.is_ok() {
                session_state.last_loaded_paths = paths;
                session_state.last_loaded_text_only = workspace.text_only;
            }
            result
        }
    })
    .await;

    match result {
        Ok(Ok(report)) => (StatusCode::OK, Json(LoadWorkspaceResponse { ok: true, error: None, chars: Some(report.chars), fits_in_one_chunk: Some(report.fits_in_one_chunk), warm_from_disk: Some(report.warm_from_disk) })),
        Ok(Err(e)) => (StatusCode::INTERNAL_SERVER_ERROR, Json(LoadWorkspaceResponse { ok: false, error: Some(e.to_string()), chars: None, fits_in_one_chunk: None, warm_from_disk: None })),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(LoadWorkspaceResponse { ok: false, error: Some(e.to_string()), chars: None, fits_in_one_chunk: None, warm_from_disk: None })),
    }
}

async fn delete_workspace_handler(State(state): State<std::sync::Arc<AppState>>, Json(req): Json<LoadWorkspaceRequest>) -> impl IntoResponse {
    match state.workspaces.delete(&req.name) {
        Ok(true) => (StatusCode::OK, Json(SimpleResponse { ok: true, error: None })),
        Ok(false) => (StatusCode::NOT_FOUND, Json(SimpleResponse { ok: false, error: Some(format!("no workspace named `{}`", req.name)) })),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(SimpleResponse { ok: false, error: Some(e.to_string()) })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The core claim behind issue #32: a second caller genuinely waits
    /// for a busy gate, and the wait is *measured*, not guessed. No live
    /// model needed - this exercises `acquire_gate` directly against a
    /// real `tokio::sync::Semaphore`.
    #[tokio::test]
    async fn acquire_gate_reports_a_real_wait_time_when_contended() {
        let gate = tokio::sync::Semaphore::new(1);
        let (first_permit, first_wait) = acquire_gate(&gate).await;
        assert_eq!(first_wait, 0, "the first caller on an empty gate never waits");

        const HOLD_MS: u64 = 200;
        let waiter = acquire_gate(&gate);
        let holder = async {
            tokio::time::sleep(std::time::Duration::from_millis(HOLD_MS)).await;
            drop(first_permit);
        };
        let ((_second_permit, second_wait), _) = tokio::join!(waiter, holder);

        // Real measured wait, not exact to the millisecond - allow slack
        // for scheduler jitter in both directions.
        assert!(second_wait >= HOLD_MS.saturating_sub(50), "expected to wait close to {HOLD_MS}ms, actually waited {second_wait}ms");
    }

    #[tokio::test]
    async fn acquire_gate_is_immediate_when_the_gate_has_room() {
        let gate = tokio::sync::Semaphore::new(2);
        let (_a, wait_a) = acquire_gate(&gate).await;
        let (_b, wait_b) = acquire_gate(&gate).await;
        assert_eq!(wait_a, 0);
        assert_eq!(wait_b, 0, "a second permit within capacity must not wait on the first");
    }

    /// Same shape as `AppState::request_gate`'s default (see
    /// `Cli::max_concurrent_requests`'s doc comment) - with capacity 1,
    /// a and b and c only ever run one at a time, in submission order:
    /// c can't jump ahead of b just because a happens to release first.
    /// The real total wall time (a's hold, then b's hold, then c
    /// finally getting in) is the honest proof, not any single queued_ms
    /// reading in isolation.
    #[tokio::test]
    async fn three_callers_on_a_single_permit_run_strictly_one_at_a_time() {
        let gate = tokio::sync::Semaphore::new(1);
        const HOLD_MS: u64 = 100;
        let start = std::time::Instant::now();

        let (permit_a, wait_a) = acquire_gate(&gate).await;
        assert_eq!(wait_a, 0);

        let b = async {
            let (permit_b, wait_b) = acquire_gate(&gate).await;
            assert!(wait_b > 0, "b must have waited for a to release");
            tokio::time::sleep(std::time::Duration::from_millis(HOLD_MS)).await;
            drop(permit_b);
        };
        let c = async {
            let (permit_c, wait_c) = acquire_gate(&gate).await;
            // c must wait for both a's hold and b's hold, since only one
            // permit exists - roughly 2x HOLD_MS, not just b's alone.
            assert!(wait_c >= (HOLD_MS * 2).saturating_sub(60), "expected c to wait for both a and b (~{}ms), actually waited {wait_c}ms", HOLD_MS * 2);
            drop(permit_c);
        };
        let release_a = async {
            tokio::time::sleep(std::time::Duration::from_millis(HOLD_MS)).await;
            drop(permit_a);
        };
        tokio::join!(b, c, release_a);

        let total = start.elapsed().as_millis() as u64;
        assert!(total >= (HOLD_MS * 2).saturating_sub(60), "three sequential holds of {HOLD_MS}ms each should take at least ~{}ms total, took {total}ms", HOLD_MS * 2);
    }
}
