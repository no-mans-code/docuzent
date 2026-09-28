use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use docuzent_core::docling_cache::DoclingCache;
use docuzent_core::generate::OllamaClient;
use docuzent_core::ingest::{self, IngestOptions};
use docuzent_core::session::{infer_context_length, Mode, Session};
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
    }
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
    let context_length = match context_length_override {
        Some(n) => n,
        None => infer_context_length(host, model)
            .with_context(|| format!("failed to infer context size for `{model}` - is `ollama serve` running and has `{model}` been pulled?"))?,
    };
    println!("Model `{model}` context window: {context_length} tokens");

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
    let report = session.ask(question)?;
    println!("{}", report.answer);
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
    Ok(())
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
