use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result};
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use docuzent_core::docling_cache::DoclingCache;
use docuzent_core::generate::OllamaClient;
use docuzent_core::session::{infer_context_length, Mode, Session};
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
}

struct AppState {
    state: Mutex<Option<SessionState>>,
    host: String,
    cache_path: PathBuf,
    docling_cache_path: PathBuf,
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
    let context_length = match context_length_override {
        Some(n) => n,
        None => infer_context_length(host, model)
            .with_context(|| format!("failed to infer context size for `{model}` - is `ollama serve` running and has it been pulled?"))?,
    };
    println!("Model `{model}` context window: {context_length} tokens");

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
    Ok(SessionState { session, model: model.to_string(), mode_label: mode_label.to_string(), context_length, map_reduce_context_fraction })
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("Context cache: {} (capacity {:.1} GB)", cli.cache.display(), kvcache::default_capacity_bytes(&cli.cache)? as f64 / 1e9);
    println!("Docling parse cache: {} (capacity {:.1} GB)", cli.docling_cache.display(), docuzent_core::docling_cache::DEFAULT_CAPACITY_BYTES as f64 / 1e9);

    let session_state = open_session_state(&cli.host, &cli.model, &cli.mode, cli.context_length, cli.map_reduce_context_fraction, &cli.cache, &cli.docling_cache)?;

    let state = std::sync::Arc::new(AppState {
        state: Mutex::new(Some(session_state)),
        host: cli.host.clone(),
        cache_path: cli.cache.clone(),
        docling_cache_path: cli.docling_cache.clone(),
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/model-info", get(model_info_handler))
        .route("/models", get(models_handler))
        .route("/model", post(switch_model_handler))
        .route("/load", post(load_handler))
        .route("/ask", post(ask_handler))
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
}

async fn model_info_handler(State(state): State<std::sync::Arc<AppState>>) -> impl IntoResponse {
    let guard = state.state.lock().unwrap();
    match guard.as_ref() {
        Some(s) => (
            StatusCode::OK,
            Json(ModelInfoResponse {
                ok: true,
                error: None,
                model: Some(s.model.clone()),
                mode: Some(s.mode_label.clone()),
                context_length: Some(s.context_length),
                map_reduce_context_fraction: Some(s.map_reduce_context_fraction),
                cache_path: Some(state.cache_path.display().to_string()),
            }),
        ),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ModelInfoResponse { ok: false, error: Some("session is switching models right now".to_string()), model: None, mode: None, context_length: None, map_reduce_context_fraction: None, cache_path: None }),
        ),
    }
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
}

fn switch_error(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<SwitchModelResponse>) {
    (status, Json(SwitchModelResponse { ok: false, error: Some(message.into()), model: None, mode: None, context_length: None, map_reduce_context_fraction: None }))
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
}

fn load_error(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<LoadResponse>) {
    (status, Json(LoadResponse { ok: false, error: Some(message.into()), chars: None, fits_in_one_chunk: None, warm_from_disk: None }))
}

/// Accepts one or more files in a single multipart request, saving each
/// and loading them together as one combined corpus.
async fn load_handler(
    State(state): State<std::sync::Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    let upload_dir = std::env::temp_dir().join("docuzent-web-uploads");
    std::fs::create_dir_all(&upload_dir).ok();

    let mut dests = Vec::new();
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => return load_error(StatusCode::BAD_REQUEST, e.to_string()),
        };
        let original_name = field.file_name().unwrap_or("upload").to_string();
        let safe_name = sanitize_filename(&original_name);
        let bytes = match field.bytes().await {
            Ok(b) => b,
            Err(e) => return load_error(StatusCode::BAD_REQUEST, e.to_string()),
        };
        let dest = upload_dir.join(&safe_name);
        if let Err(e) = std::fs::write(&dest, &bytes) {
            return load_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
        }
        dests.push(dest);
    }

    if dests.is_empty() {
        return load_error(StatusCode::BAD_REQUEST, "no file in request");
    }

    let result = tokio::task::spawn_blocking({
        let state = state.clone();
        move || {
            let mut guard = state.state.lock().unwrap();
            let Some(session_state) = guard.as_mut() else {
                anyhow::bail!("session is switching models right now - try again in a moment");
            };
            session_state.session.load_documents(&dests)
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
            }),
        ),
        Ok(Err(e)) => load_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => load_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
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
}

fn ask_error(status: StatusCode, message: impl Into<String>) -> (StatusCode, Json<AskResponse>) {
    (
        status,
        Json(AskResponse { ok: false, error: Some(message.into()), answer: None, used_map_reduce: None, chunks_mapped: None, adaptive_decision: None, timings: None }),
    )
}

async fn ask_handler(State(state): State<std::sync::Arc<AppState>>, Json(req): Json<AskRequest>) -> impl IntoResponse {
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

    match result {
        Ok(Ok(report)) => (
            StatusCode::OK,
            Json(AskResponse {
                ok: true,
                error: None,
                answer: Some(report.answer),
                used_map_reduce: Some(report.used_map_reduce),
                chunks_mapped: Some(report.chunks_mapped),
                adaptive_decision: report.adaptive_decision,
                timings: Some(report.timings),
            }),
        ),
        Ok(Err(e)) => ask_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        Err(e) => ask_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
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
