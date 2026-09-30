use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use docuzent_core::docling_cache::DoclingCache;
use docuzent_core::generate::OllamaClient;
use docuzent_core::ingest::{self, IngestOptions};
use docuzent_core::session::{Mode, Session};
use docuzent_core::vram::ModelVramEstimate;
use kvcache::Cache;

/// docuzent - local document-intelligence CLI
#[derive(Parser)]
#[command(name = "docuzent")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Convert files via Docling.
    Ingest {
        /// Directory (or file) to ingest via Docling
        #[arg(default_value = "temp-test")]
        source: PathBuf,
        /// Directory where converted output is written
        #[arg(long, default_value = "temp-test-out")]
        output: PathBuf,
        /// Docling output format
        #[arg(long, default_value = "json")]
        to: String,
        /// Path to the docling executable. Defaults to $DOCLING_BIN, then a
        /// project-local .venv, then "docling" on PATH.
        #[arg(long)]
        docling_bin: Option<String>,
        /// Accelerator device: auto (GPU if present, else CPU), cpu, cuda, mps, xpu.
        #[arg(long, default_value = "auto")]
        device: String,
        /// Force CPU even if a GPU is present. Shorthand for --device cpu.
        #[arg(long)]
        no_gpu: bool,
    },
    /// Ask one or more questions about one or more documents, loaded
    /// together as a single combined corpus (any format Docling parses,
    /// or .zips of them, extracted one level deep). The model's context
    /// window is inferred automatically unless overridden; a combined
    /// corpus that fits is answered in one call, a bigger one via
    /// map-reduce.
    Ask {
        /// The document(s) to load (or .zips containing documents) -
        /// multiple files are combined into one corpus
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Ollama model to use
        #[arg(long, default_value = "qwen2.5:3b")]
        model: String,
        /// Ollama host
        #[arg(long, default_value = "http://localhost:11434")]
        host: String,
        /// How the LLM-context cache is used: swap (always reuse when
        /// available), raw (never reuse - always a full cold reprocess),
        /// adaptive (predict which is faster and pick per-request)
        #[arg(long, default_value = "adaptive")]
        mode: String,
        /// Override the model's inferred context window (tokens).
        /// Defaults to the model's real max, read from Ollama.
        #[arg(long)]
        context_length: Option<u32>,
        /// Fraction of the model's context window a single map-reduce
        /// chunk is sized to use - see the README's "Adaptive Mode"
        /// section and https://github.com/no-mans-code/docuzent/issues/24
        /// for why this isn't just the model's full nominal window
        #[arg(long, default_value_t = docuzent_core::session::DEFAULT_MAP_REDUCE_CONTEXT_FRACTION)]
        map_reduce_context_fraction: f32,
        /// Where the on-disk LLM-context cache lives
        #[arg(long, default_value = ".docuzent-cache/context.redb")]
        cache: PathBuf,
        /// Where the on-disk Docling parse cache lives
        #[arg(long, default_value = ".docuzent-cache/docling.redb")]
        docling_cache: PathBuf,
        /// Ask exactly this one question and exit, instead of an interactive loop
        #[arg(long)]
        question: Option<String>,
        /// Load `files` directly as raw UTF-8 text, skipping Docling
        /// entirely - for files that are already text (.txt, .md, .log,
        /// config files, source code) that don't need Docling's
        /// document-structure parsing. See
        /// https://github.com/no-mans-code/docuzent/issues/27
        #[arg(long)]
        text_only: bool,
    },
    /// Benchmarks the on-disk LLM-context cache's real effect: a cold
    /// session (no cache) vs. a fresh session reusing what the cold one
    /// persisted. Always uses Mode::Swap - it exists to measure the cache
    /// mechanism itself, not to compare modes.
    Bench {
        /// The document to benchmark against
        file: PathBuf,
        #[arg(long, default_value = "qwen2.5:3b")]
        model: String,
        #[arg(long, default_value = "http://localhost:11434")]
        host: String,
        #[arg(long, default_value = "What is this document about?")]
        question: String,
    },
    /// Ask a long document (a book) a question in one of the four reading modes - RAG, expanded RAG, RAG pointing at
    /// saved KV parts, or saved KV parts only (see docs/READING_MODES.md). Needs a llama.cpp server for the modes
    /// that save KV states; what it learns (saved parts, indexes) is kept and reused.
    Read {
        /// The document
        file: PathBuf,
        /// The question
        question: String,
        /// rag, rag-expanded, rag-kv or kv
        #[arg(long, default_value = "rag-kv")]
        mode: String,
        /// Cut the index where the model says scenes and topics change (default: at paragraphs)
        #[arg(long)]
        guided: bool,
        /// A llama.cpp server (it saves KV states - needed for rag-kv, kv and expansions)
        #[arg(long, default_value = "http://localhost:8081")]
        llm_url: String,
        /// Use an Ollama server instead (no saved KV states)
        #[arg(long)]
        ollama_url: Option<String>,
        #[arg(long, default_value = "qwen3:14b")]
        ollama_model: String,
        /// The llama.cpp server's --slot-save-path
        #[arg(long, default_value = ".docuzent-kv")]
        kv_dir: PathBuf,
        #[arg(long, default_value_t = 40)]
        kv_budget_gb: u64,
        /// An OpenAI-compatible embeddings endpoint (Ollama, or llama.cpp --embeddings); none = search by words only
        #[arg(long)]
        embed_url: Option<String>,
        #[arg(long, default_value = "nomic-embed-text")]
        embed_model: String,
        /// Where indexes are kept
        #[arg(long, default_value = ".docuzent-read")]
        work: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Ingest { source, output, to, docling_bin, device, no_gpu } => {
            run_ingest(source, output, to, docling_bin, device, no_gpu)
        }
        Command::Ask { files, model, host, mode, context_length, map_reduce_context_fraction, cache, docling_cache, question, text_only } => {
            run_ask(files, model, host, mode, context_length, map_reduce_context_fraction, cache, docling_cache, question, text_only)
        }
        Command::Bench { file, model, host, question } => run_bench(file, model, host, question),
        Command::Read { file, question, mode, guided, llm_url, ollama_url, ollama_model, kv_dir, kv_budget_gb, embed_url, embed_model, work } => {
            run_read(ReadArgs { file, question, mode, guided, llm_url, ollama_url, ollama_model, kv_dir, kv_budget_gb, embed_url, embed_model, work })
        }
    }
}

struct ReadArgs {
    file: PathBuf,
    question: String,
    mode: String,
    guided: bool,
    llm_url: String,
    ollama_url: Option<String>,
    ollama_model: String,
    kv_dir: PathBuf,
    kv_budget_gb: u64,
    embed_url: Option<String>,
    embed_model: String,
    work: PathBuf,
}

fn run_read(a: ReadArgs) -> Result<()> {
    use std::sync::Arc;

    use docuzent_doc::kvpool::KvPool;
    use docuzent_llm::{Embedder, Engine, LlamaServer, OllamaServer, OpenAiEmbedder};
    use docuzent_read::{answer, read, Chunking, Document, Mode as ReadMode, ReadOptions, Shelf, Sources};

    let mode = ReadMode::parse(&a.mode)?;
    let (engine, model): (Arc<dyn Engine>, String) = match &a.ollama_url {
        Some(url) => (Arc::new(OllamaServer::connect(url, &a.ollama_model, 12288)?), a.ollama_model.clone()),
        None => {
            let s = LlamaServer::connect(&a.llm_url)?;
            let name = s.model_path().rsplit(['/', '\\']).next().unwrap_or("model").to_string();
            (Arc::new(s), name)
        }
    };
    // a scope of its own, so this never touches another app's saved states in a shared directory
    let model_id = format!("dz-{}", model.chars().filter(|c| c.is_ascii_alphanumeric()).take(9).collect::<String>().to_lowercase());
    let store = Arc::new(docuzent_kv::KvStore::open_scoped(&a.kv_dir, a.kv_budget_gb << 30, Some("dz"))?);
    let pool = KvPool::new(engine.clone(), store);
    let embedder = a.embed_url.as_deref().map(|u| OpenAiEmbedder::new(u, &a.embed_model));

    let book = docuzent_doc::extract::extract_book(&a.file, None)?;
    let named = pool.with_scratch(|llm| docuzent_doc::title::detect(llm, &a.file.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default(), &book.text))?;
    eprintln!("{}{} - {} characters", named.title, named.author.as_ref().map(|x| format!(" by {x}")).unwrap_or_default(), book.text.len());
    let doc = Document::from_text(&book.text, &named.title, named.author.as_deref(), &model_id, &|t| engine.count_tokens(t))?;
    let chunking = if a.guided { Chunking::Guided } else { Chunking::Plain };
    if mode.needs_saved_parts() || mode.needs_expansions() || a.guided {
        let saved = doc.save_parts(&pool, &mut |i, n| if n > 0 { eprint!("\rsaving part {} of {n}   ", i + 1) })?;
        if saved > 0 {
            eprintln!();
        }
    }
    let index = if mode.needs_index() {
        Some(doc.index(&pool, &a.work.join(&doc.id), chunking, mode.needs_expansions(), &model_id, embedder.as_ref().map(|e| e as &dyn Embedder), &mut |s| eprintln!("{s} ..."))?)
    } else {
        None
    };
    let started = std::time::Instant::now();
    let reading = read(
        mode,
        Sources { pool: &pool, corpus: &doc, indexes: index.iter().map(|ix| Shelf { index: ix, first_part: 0 }).collect(), embedder: embedder.as_ref().map(|e| e as &dyn Embedder) },
        &a.question,
        ReadOptions::default(),
        &|s| eprintln!("  {s}"),
    )?;
    let reply = pool.with_scratch(|llm| answer::answer(llm, &a.question, &reading.passages))?;
    println!("{reply}\n");
    for p in &reading.passages {
        println!("[part {}] {}", p.part, p.text.chars().take(300).collect::<String>().replace('\n', " "));
    }
    eprintln!("\n{} - {:.1} s ({} passages, {} parts read closely, {} chunks found)", mode.label(), started.elapsed().as_secs_f64(), reading.passages.len(), reading.read_closely, reading.retrieved);
    Ok(())
}

fn run_ingest(
    source: PathBuf,
    output: PathBuf,
    to: String,
    docling_bin: Option<String>,
    device: String,
    no_gpu: bool,
) -> Result<()> {
    let device = if no_gpu { "cpu".to_string() } else { device };

    println!(
        "Ingesting {} -> {} (format: {}, device: {})",
        source.display(),
        output.display(),
        to,
        device
    );

    let report = ingest::run(&IngestOptions { source, output: output.clone(), to, device, docling_bin })?;

    println!("Using docling at `{}`", report.docling_bin);
    println!("Done. {} file(s) written to {}:", report.produced_files.len(), output.display());
    for name in &report.produced_files {
        println!("  - {name}");
    }
    Ok(())
}

/// Reports the real numbers behind the context-length decision - not just
/// the final value, so a user sees *why* it's smaller than the model's
/// nominal window when it is, rather than silently getting a different
/// number than they might have expected. See
/// https://github.com/no-mans-code/docuzent/issues/30.
fn print_context_length_report(model: &str, context_length: u32, overridden: bool, estimate: &ModelVramEstimate) {
    if overridden {
        println!("Model `{model}` context window: {context_length} tokens (explicit override - nominal window is {})", estimate.nominal_context_length);
        return;
    }
    if context_length == estimate.nominal_context_length {
        println!("Model `{model}` context window: {context_length} tokens");
        return;
    }
    println!("Model `{model}` context window: {context_length} tokens (safe default, not the nominal {})", estimate.nominal_context_length);
    if let Some(trained) = estimate.original_context_length {
        if trained != estimate.nominal_context_length {
            println!("  - nominal window is a RoPE-scaling extrapolation; the model was actually trained on {trained} tokens");
        }
    }
    match estimate.vram_free_bytes {
        Some(free) if context_length < estimate.original_context_length.unwrap_or(estimate.nominal_context_length) => {
            println!(
                "  - capped further to fit free VRAM (~{:.1} GB free, model weights ~{:.1} GB) - large documents will use map-reduce",
                free as f64 / 1e9,
                estimate.weight_bytes as f64 / 1e9
            );
        }
        None => println!("  - free VRAM couldn't be measured (no NVIDIA GPU detected) - using the model's trained context as-is"),
        _ => {}
    }
    println!("  - override with --context-length to use a different value");
}

#[allow(clippy::too_many_arguments)]
fn open_session(
    model: &str,
    host: &str,
    mode: Mode,
    context_length_override: Option<u32>,
    map_reduce_context_fraction: f32,
    cache_path: &PathBuf,
    docling_cache_path: &PathBuf,
) -> Result<Session<OllamaClient>> {
    let (context_length, estimate) = docuzent_core::vram::resolve_context_length(host, model, context_length_override)
        .with_context(|| format!("failed to size context window for `{model}` - is `ollama serve` running and has `{model}` been pulled?"))?;
    print_context_length_report(model, context_length, context_length_override.is_some(), &estimate);

    if let Some(parent) = cache_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let capacity = kvcache::default_capacity_bytes(cache_path)?;
    println!("Context cache: {} (capacity {:.1} GB)", cache_path.display(), capacity as f64 / 1e9);
    let cache = Cache::open(cache_path, capacity)?;

    if let Some(parent) = docling_cache_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    println!(
        "Docling parse cache: {} (capacity {:.1} GB)",
        docling_cache_path.display(),
        docuzent_core::docling_cache::DEFAULT_CAPACITY_BYTES as f64 / 1e9
    );
    let docling_cache = DoclingCache::open(docling_cache_path, docuzent_core::docling_cache::DEFAULT_CAPACITY_BYTES)?;

    Session::open(
        OllamaClient::new(host, model, context_length),
        model,
        context_length,
        std::env::temp_dir().join("docuzent-ask"),
        mode,
        map_reduce_context_fraction,
        cache,
        docling_cache,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_ask(
    files: Vec<PathBuf>,
    model: String,
    host: String,
    mode: String,
    context_length: Option<u32>,
    map_reduce_context_fraction: f32,
    cache: PathBuf,
    docling_cache: PathBuf,
    question: Option<String>,
    text_only: bool,
) -> Result<()> {
    let mode = Mode::parse(&mode)?;
    let mut session = open_session(&model, &host, mode, context_length, map_reduce_context_fraction, &cache, &docling_cache)?;

    let load = if text_only { session.load_text_files(&files)? } else { session.load_documents(&files)? };
    println!(
        "Loaded {} file(s) ({} chars, {}, {})",
        files.len(),
        load.chars,
        if load.fits_in_one_chunk { "fits in one chunk" } else { "will use map-reduce" },
        if load.warm_from_disk { "a cached context is available on disk" } else { "cold - not seen before" }
    );

    if let Some(q) = question {
        answer_and_print(&mut session, &q)?;
        return Ok(());
    }

    println!("Ask questions about this document (blank line to quit):");
    let stdin = io::stdin();
    loop {
        print!("> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        answer_and_print(&mut session, line)?;
    }
    Ok(())
}

fn answer_and_print(session: &mut Session<OllamaClient>, question: &str) -> Result<()> {
    // Streams the final answer token-by-token as it's generated - real
    // streaming (Ollama's `stream: true`), not an estimate; see
    // https://github.com/no-mans-code/docuzent/issues/35.
    let report = session.ask_streaming(question, &mut |fragment| {
        print!("{fragment}");
        io::stdout().flush().ok();
    })?;
    println!();
    print!("  [{}", if report.used_map_reduce { format!("map-reduce over {} chunks", report.chunks_mapped) } else { "single chunk".to_string() });
    for t in &report.timings {
        print!(", {}: {:.0}ms prompt_eval ({} tok)", t.label, t.prompt_eval_duration_ms, t.prompt_eval_count);
    }
    if let Some(d) = report.adaptive_decision {
        print!(
            ", adaptive: predicted swap {:.0}ms vs raw {:.0}ms, chose {}",
            d.predicted_swap_ms,
            d.predicted_raw_ms,
            if d.chose_swap { "swap" } else { "raw" }
        );
    }
    println!("]");
    println!("  sources: {}", report.source_files.join(", "));
    for s in &report.sources {
        println!("    [chunk {}] {}", s.chunk_index, truncate_for_display(&s.excerpt, 160));
    }
    Ok(())
}

/// Keeps a citation excerpt printable on one line without dumping a
/// whole chunk's extraction into the terminal.
fn truncate_for_display(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(max_chars).collect::<String>())
    }
}

fn run_bench(file: PathBuf, model: String, host: String, question: String) -> Result<()> {
    let cold_cache_path = std::env::temp_dir().join(format!("docuzent-bench-{}.redb", std::process::id()));
    let cold_docling_cache_path = std::env::temp_dir().join(format!("docuzent-bench-docling-{}.redb", std::process::id()));
    let _ = std::fs::remove_file(&cold_cache_path); // guarantee a genuinely cold cache
    let _ = std::fs::remove_file(&cold_docling_cache_path);

    println!("=== Cold run (no cache) ===");
    let mut cold = open_session(&model, &host, Mode::Swap, None, docuzent_core::session::DEFAULT_MAP_REDUCE_CONTEXT_FRACTION, &cold_cache_path, &cold_docling_cache_path)?;
    let load = cold.load_document(&file)?;
    assert!(!load.warm_from_disk, "bench requires a genuinely fresh cache path");
    let cold_report = cold.ask(&question)?;
    for t in &cold_report.timings {
        println!("  {}: prompt_eval={:.0}ms ({} tok), eval={:.0}ms, wall={:.0}ms", t.label, t.prompt_eval_duration_ms, t.prompt_eval_count, t.eval_duration_ms, t.wall_ms);
    }
    drop(cold); // release the cache file before reopening it

    println!("=== Warm run (fresh session, same cache file - simulates a process restart) ===");
    let mut warm = open_session(&model, &host, Mode::Swap, None, docuzent_core::session::DEFAULT_MAP_REDUCE_CONTEXT_FRACTION, &cold_cache_path, &cold_docling_cache_path)?;
    let load = warm.load_document(&file)?;
    assert!(load.warm_from_disk, "the cold run above should have persisted a context for this document");
    let warm_report = warm.ask(&question)?;
    for t in &warm_report.timings {
        println!("  {}: prompt_eval={:.0}ms ({} tok), eval={:.0}ms, wall={:.0}ms", t.label, t.prompt_eval_duration_ms, t.prompt_eval_count, t.eval_duration_ms, t.wall_ms);
    }

    let cold_prime = cold_report.timings.iter().find(|t| t.label == "prime");
    if let Some(cold_prime) = cold_prime {
        let warm_answer = &warm_report.timings[0];
        let speedup = cold_prime.prompt_eval_duration_ms / warm_answer.prompt_eval_duration_ms.max(0.001);
        println!(
            "=== Result: priming cold took {:.0}ms of prompt_eval; the warm run's first answer took {:.0}ms - {:.1}x ===",
            cold_prime.prompt_eval_duration_ms, warm_answer.prompt_eval_duration_ms, speedup
        );
    }

    let _ = std::fs::remove_file(&cold_cache_path);
    let _ = std::fs::remove_file(&cold_docling_cache_path);
    Ok(())
}
