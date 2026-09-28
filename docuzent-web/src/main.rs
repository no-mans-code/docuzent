use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{Context, Result};
use axum::extract::{Multipart, State};
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

/// docuzent-web - a local browser UI for document Q&A. Entirely optional:
/// `docuzent-cli` is fully capable standalone without this running at all.
#[derive(Parser)]
struct Cli {
    /// Ollama model to use
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

struct AppState {
    session: Mutex<Session<OllamaClient>>,
    model: String,
    mode: String,
    context_length: u32,
    cache_path: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let mode = Mode::parse(&cli.mode)?;

    let context_length = match cli.context_length {
        Some(n) => n,
        None => infer_context_length(&cli.host, &cli.model).with_context(|| {
            format!("failed to infer context size for `{}` - is `ollama serve` running and has it been pulled?", cli.model)
        })?,
    };
    println!("Model `{}` context window: {} tokens", cli.model, context_length);

    if let Some(parent) = cli.cache.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let capacity = kvcache::default_capacity_bytes(&cli.cache)?;
    println!("Context cache: {} (capacity {:.1} GB)", cli.cache.display(), capacity as f64 / 1e9);
    let cache = Cache::open(&cli.cache, capacity)?;

    if let Some(parent) = cli.docling_cache.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let docling_cache = DoclingCache::open(&cli.docling_cache, docuzent_core::docling_cache::DEFAULT_CAPACITY_BYTES)?;

    let upload_dir = std::env::temp_dir().join("docuzent-web-uploads");
    std::fs::create_dir_all(&upload_dir).ok();

    let session = Session::open(OllamaClient::new(&cli.host, &cli.model, context_length), &cli.model, context_length, upload_dir, mode, cache, docling_cache)?;

    let state = std::sync::Arc::new(AppState {
        session: Mutex::new(session),
        model: cli.model.clone(),
        mode: cli.mode.clone(),
        context_length,
        cache_path: cli.cache.clone(),
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/model-info", get(model_info_handler))
        .route("/load", post(load_handler))
        .route("/ask", post(ask_handler))
        .with_state(state);

    let addr = format!("127.0.0.1:{}", cli.port);
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
    model: String,
    mode: String,
    context_length: u32,
    cache_path: String,
}

async fn model_info_handler(State(state): State<std::sync::Arc<AppState>>) -> Json<ModelInfoResponse> {
    Json(ModelInfoResponse {
        model: state.model.clone(),
        mode: state.mode.clone(),
        context_length: state.context_length,
        cache_path: state.cache_path.display().to_string(),
    })
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
            let mut session = state.session.lock().unwrap();
            session.load_documents(&dests)
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
            let mut session = state.session.lock().unwrap();
            session.ask(&req.question)
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
