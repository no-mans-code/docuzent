//! docuzent-eval - measures the four reading modes on a real book: accuracy, seconds per question, learning time
//! and disk.
//!
//! ```text
//! docuzent-eval --book book.txt --questions eval/sets/harry-potter.json \
//!   --llm-url http://llm:8080 --kv-dir /kv --embed-url http://host.docker.internal:11434 \
//!   --runs rag,rag:guided,rag-expanded,rag-kv,kv --work eval/work --out eval/results
//! ```
//!
//! A *run* is a mode, optionally with the chunking its index uses (`rag:guided`; plain by default). Every run
//! answers every question with the same plain answerer (`docuzent_read::answer`), so the mode is what differs.
//! Each answer is checked against the question's patterns: `must` (every one appears), `any_of` (at least one),
//! `never` (none - outside knowledge, spoilers, filler).
//!
//! What is learned (saved parts, indexes, expansions, embeddings) is kept in `--work` and in the KV store, with how
//! long it took, so a second evaluation measures the questions without learning the book again. The KV store is
//! *scoped* (`--scope`, default `eval`): it shares the model server's directory with any app using it, and can
//! never count, touch or evict that app's files.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser;
use docuzent_doc::kvpool::KvPool;
use docuzent_doc::{hash, title};
use docuzent_kv::KvStore;
use docuzent_llm::{Embedder, Engine, LlamaServer, OllamaServer, OpenAiEmbedder};
use docuzent_read::expand::{expand, guided_index};
use docuzent_read::{answer, check_context, read, Think, Chunking, Corpus, Document, Index, Mode, ReadOptions, Sources};
use regex::RegexBuilder;
use serde::{Deserialize, Serialize};

#[derive(Parser)]
#[command(name = "docuzent-eval", about = "Measure the four reading modes on a book")]
struct Cli {
    /// The book, as plain text (or any format docuzent reads)
    #[arg(long)]
    book: PathBuf,
    /// The question set (JSON; see eval/sets/)
    #[arg(long)]
    questions: PathBuf,
    /// The file name the book came in, to test title detection with (default: the --book file's name)
    #[arg(long)]
    file_name: Option<String>,
    /// A llama.cpp server (saves KV states: needed for rag-kv and kv)
    #[arg(long, env = "DOCUZENT_LLM_URL", default_value = "http://localhost:8081")]
    llm_url: String,
    /// Use an Ollama server instead (no saved KV states)
    #[arg(long)]
    ollama_url: Option<String>,
    #[arg(long, default_value = "qwen3:14b")]
    ollama_model: String,
    /// The KV store directory - the model server's --slot-save-path
    #[arg(long, default_value = "/kv")]
    kv_dir: PathBuf,
    #[arg(long, default_value = "eval")]
    scope: String,
    #[arg(long, default_value_t = 40)]
    kv_budget_gb: u64,
    /// An OpenAI-compatible embeddings endpoint (Ollama, or llama.cpp with --embeddings); none = words only
    #[arg(long)]
    embed_url: Option<String>,
    #[arg(long, default_value = "nomic-embed-text")]
    embed_model: String,
    /// Runs: mode[:plain|guided], comma-separated
    #[arg(long, default_value = "rag,rag:guided,rag-expanded,rag-kv,kv")]
    runs: String,
    /// Only these question numbers (1-based), comma-separated
    #[arg(long)]
    only: Option<String>,
    #[arg(long, default_value = "eval/work")]
    work: PathBuf,
    #[arg(long, default_value = "eval/results")]
    out: PathBuf,
}

#[derive(Deserialize)]
struct QuestionSet {
    book: String,
    #[serde(default)]
    expected_title: Option<String>,
    questions: Vec<Question>,
}

#[derive(Deserialize, Clone)]
struct Question {
    q: String,
    /// `tuning` (written after seeing failures) or `held-out` (written before any answer was seen)
    #[serde(default = "default_set")]
    set: String,
    #[serde(default)]
    must: Vec<String>,
    #[serde(default)]
    any_of: Vec<String>,
    #[serde(default)]
    never: Vec<String>,
    /// What a right answer looks like *off the leash* (the model may reason and use knowledge beyond the book), when
    /// that differs - for a question the book cannot answer, "the book does not say" is right on the leash, wrong off it.
    #[serde(default)]
    offleash: Option<Check>,
}

#[derive(Deserialize, Clone, Default)]
struct Check {
    #[serde(default)]
    must: Vec<String>,
    #[serde(default)]
    any_of: Vec<String>,
    #[serde(default)]
    never: Vec<String>,
}

fn default_set() -> String {
    "held-out".into()
}

#[derive(Serialize, Deserialize, Default, Clone)]
struct Learned {
    /// Seconds to read every part and save its KV state (0 when it was already saved).
    kv_s: Option<f64>,
    kv_bytes: u64,
    index_plain_s: Option<f64>,
    index_guided_s: Option<f64>,
    expand_s: Option<f64>,
    embed_s: std::collections::BTreeMap<String, f64>,
    index_bytes: std::collections::BTreeMap<String, u64>,
}

#[derive(Serialize, Deserialize, Default)]
#[serde(default)]
struct QuestionResult {
    run: String,
    n: usize,
    set: String,
    question: String,
    pass: bool,
    problems: Vec<String>,
    answer: String,
    seconds: f64,
    read_s: f64,
    answer_s: f64,
    parts_read: usize,
    chunks_retrieved: usize,
}

#[derive(Serialize, Deserialize, Default)]
#[serde(default)]
struct RunSummary {
    run: String,
    /// The model's plain generation speed (tokens/s) before and after the run: a run where it fell is one whose
    /// timings something else on the machine was eating into.
    speed_before: f64,
    speed_after: f64,
    mode: String,
    passed: usize,
    total: usize,
    held_out_passed: usize,
    held_out_total: usize,
    avg_s: f64,
    max_s: f64,
    learn_s: f64,
}

#[derive(Serialize, Deserialize, Default)]
#[serde(default)]
struct Report {
    book: String,
    expected_title: Option<String>,
    detected_title: Option<title::DocTitle>,
    chars: usize,
    parts: usize,
    model: String,
    embedder: Option<String>,
    learned: Learned,
    runs: Vec<RunSummary>,
    questions: Vec<QuestionResult>,
}

/// Plain generation speed, tokens per second: 64 tokens of nothing in particular.
fn speed(pool: &KvPool) -> f64 {
    pool.with_scratch(|llm| {
        let c = llm.complete(&docuzent_llm::chatml::ask("", "Count slowly from one upwards, in words."), &docuzent_llm::Sampling::precise(64), &mut |_| {})?;
        Ok(if c.gen_ms > 0.0 { c.gen_tokens as f64 / (c.gen_ms / 1000.0) } else { 0.0 })
    })
    .unwrap_or(0.0)
}

fn secs(t: Instant) -> f64 {
    t.elapsed().as_secs_f64()
}

fn check(q: &Question, answer: &str, offleash: bool) -> Vec<String> {
    let re = |p: &str| RegexBuilder::new(p).case_insensitive(true).build();
    let own = Check { must: q.must.clone(), any_of: q.any_of.clone(), never: q.never.clone() };
    let q = if offleash { q.offleash.as_ref().unwrap_or(&own) } else { &own };
    let mut problems = Vec::new();
    for p in &q.must {
        if !re(p).map(|r| r.is_match(answer)).unwrap_or(false) {
            problems.push(format!("missing /{p}/"));
        }
    }
    if !q.any_of.is_empty() && !q.any_of.iter().any(|p| re(p).map(|r| r.is_match(answer)).unwrap_or(false)) {
        problems.push(format!("none of {:?}", q.any_of));
    }
    for p in &q.never {
        if re(p).map(|r| r.is_match(answer)).unwrap_or(false) {
            problems.push(format!("must not say /{p}/"));
        }
    }
    problems
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let set: QuestionSet = serde_json::from_slice(&std::fs::read(&cli.questions).with_context(|| format!("no question set at {}", cli.questions.display()))?)?;
    let only: Option<Vec<usize>> = cli.only.as_deref().map(|s| s.split(',').filter_map(|n| n.trim().parse().ok()).collect());

    // the engine
    let (engine, model): (Arc<dyn Engine>, String) = match &cli.ollama_url {
        Some(url) => {
            let o = OllamaServer::connect(url, &cli.ollama_model, 12288)?;
            (Arc::new(o), format!("ollama:{}", cli.ollama_model))
        }
        None => {
            let s = LlamaServer::connect(&cli.llm_url)?;
            let name = s.model_path().rsplit('/').next().unwrap_or("model").to_string();
            (Arc::new(s), name)
        }
    };
    check_context(&model, engine.context_size())?;
    eprintln!("[eval] model {model}: a {}-token window", engine.context_size());
    // What saved states and indexes are filed under. An Ollama model's whole name - its first letters alone would
    // file qwen3:0.6b and qwen2.5:3b together; a weights file keeps the short id its saved states already have.
    let short: String = model.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>().to_lowercase();
    let model_id = if cli.ollama_url.is_some() { format!("{}-{short}", cli.scope) } else { format!("{}-{}", cli.scope, &short[..short.len().min(8)]) };
    let store = Arc::new(KvStore::open_scoped(&cli.kv_dir, cli.kv_budget_gb << 30, Some(&cli.scope))?);
    let pool = KvPool::new(engine.clone(), store.clone());
    let embedder: Option<Box<dyn Embedder>> = cli.embed_url.as_deref().map(|u| Box::new(OpenAiEmbedder::new(u, &cli.embed_model)) as Box<dyn Embedder>);

    // the book
    let book = docuzent_doc::extract::extract_book(&cli.book, None)?;
    let text = book.text;
    let doc_id: String = hash::hash_bytes(text.as_bytes()).chars().take(12).collect();
    let work = cli.work.join(&doc_id);
    std::fs::create_dir_all(&work)?;
    let file_name_for_title = cli.file_name.clone().unwrap_or_else(|| cli.book.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default());
    let detected = pool.with_scratch(|llm| title::detect(llm, &file_name_for_title, &text)).ok();
    eprintln!("[eval] {} - {} chars; title read as {:?}", set.book, text.len(), detected.as_ref().map(|t| (&t.title, &t.author)));

    let corpus = Document::from_text(&text, &set.book, None, &model_id, &|s| engine.count_tokens(s))?;
    eprintln!("[eval] {} parts", corpus.part_count());

    let learned_path = work.join(format!("learned-{model_id}.json"));
    let mut learned: Learned = std::fs::read(&learned_path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    let save_learned = |l: &Learned| std::fs::write(&learned_path, serde_json::to_vec_pretty(l).unwrap());

    // mode[:guided][:expanded][:offleash][:think|:nothink]
    let runs: Vec<(Mode, Chunking, bool, bool, Think, String)> = cli
        .runs
        .split(',')
        .map(|r| {
            let mut tokens = r.trim().split(':');
            let mode = Mode::parse(tokens.next().unwrap_or(""))?;
            let rest: Vec<&str> = tokens.collect();
            let chunking = if rest.contains(&"guided") { Chunking::Guided } else { Chunking::Plain };
            let expanded = mode.needs_expansions() || rest.contains(&"expanded");
            let offleash = rest.contains(&"offleash");
            let think = if rest.contains(&"think") { Think::Always } else if rest.contains(&"nothink") { Think::Never } else { Think::Auto };
            let mut name = mode.as_str().to_string();
            if chunking == Chunking::Guided {
                name.push_str(" (guided chunks)");
            }
            if expanded && !mode.needs_expansions() {
                name.push_str(" (expanded index)");
            }
            if offleash {
                name.push_str(" (off-leash)");
            }
            match think {
                Think::Always => name.push_str(" (always reasons)"),
                Think::Never => name.push_str(" (never reasons)"),
                Think::Auto => {}
            }
            Ok((mode, chunking, expanded, offleash, think, name))
        })
        .collect::<Result<_>>()?;

    // saved parts: every run that needs them, and guided chunking and expansions (which read each part resident)
    let need_kv = runs.iter().any(|(m, c, x, _, _, _)| m.needs_saved_parts() || *x || *c == Chunking::Guided);
    if need_kv {
        let missing: Vec<usize> = (0..corpus.part_count()).filter(|i| !store.contains(corpus.part_file(*i))).collect();
        if !missing.is_empty() {
            let t = Instant::now();
            for (n, i) in missing.iter().enumerate() {
                eprintln!("[eval] saving part {} of {} ({} to go)", i + 1, corpus.part_count(), missing.len() - n);
                pool.prime(corpus.part_file(*i), &corpus.id, &corpus.part_prefix(*i))?;
            }
            if missing.len() == corpus.part_count() {
                learned.kv_s = Some(secs(t));
            }
        }
        learned.kv_bytes = (0..corpus.part_count()).filter_map(|i| std::fs::metadata(cli.kv_dir.join(corpus.part_file(i))).ok()).map(|m| m.len()).sum();
        save_learned(&learned)?;
    }

    // indexes: one per (chunking, expanded), kept in the work directory
    let embed_id = embedder.as_ref().map(|e| e.id());
    let index_for = |chunking: Chunking, expanded: bool, learned: &mut Learned| -> Result<Index> {
        let key = format!("{}{}", if chunking == Chunking::Guided { "guided" } else { "plain" }, if expanded { "-expanded" } else { "" });
        let dir = work.join(format!("index-{key}-{model_id}"));
        if let Some(ix) = Index::load(&dir)? {
            if ix.embed_model == embed_id {
                return Ok(ix);
            }
        }
        let t = Instant::now();
        let mut ix = match chunking {
            Chunking::Plain => Index::plain(&(0..corpus.part_count()).map(|i| corpus.part_text(i)).collect::<Vec<_>>()),
            Chunking::Guided => guided_index(&pool, &corpus, &mut |i, n| eprintln!("[eval] guided chunking: part {} of {n}", i + 1))?,
        };
        match chunking {
            Chunking::Plain => learned.index_plain_s = Some(secs(t)),
            Chunking::Guided => learned.index_guided_s = Some(secs(t)),
        }
        if expanded {
            let t = Instant::now();
            expand(&pool, &corpus, &mut ix, &model_id, &mut |i, n| if i % 10 == 0 { eprintln!("[eval] expanding chunk {i} of {n}") })?;
            learned.expand_s = Some(secs(t));
        }
        if let Some(e) = embedder.as_deref() {
            let t = Instant::now();
            ix.embed(e, &mut |i, n| if i % 256 == 0 { eprintln!("[eval] embedding {i} of {n}") })?;
            learned.embed_s.insert(key.clone(), secs(t));
        }
        ix.save(&dir)?;
        learned.index_bytes.insert(key, Index::disk_bytes(&dir));
        Ok(ix)
    };

    let mut results: Vec<QuestionResult> = Vec::new();
    let mut summaries: Vec<RunSummary> = Vec::new();
    for (mode, chunking, expanded, offleash, think, name) in &runs {
        let index = if mode.needs_index() { Some(index_for(*chunking, *expanded, &mut learned)?) } else { None };
        save_learned(&learned)?;
        // what learning the book takes for this run: chunking (guided needs the model; plain is instant), expansions,
        // embeddings, and saved parts when the run reads them - or needed them to be made
        let key = format!("{}{}", if *chunking == Chunking::Guided { "guided" } else { "plain" }, if *expanded { "-expanded" } else { "" });
        let chunk_s = if *chunking == Chunking::Guided { learned.index_guided_s.unwrap_or(0.0) } else { learned.index_plain_s.unwrap_or(0.0) };
        let kv_s = if mode.needs_saved_parts() || *expanded || *chunking == Chunking::Guided { learned.kv_s.unwrap_or(0.0) } else { 0.0 };
        let learn_s = kv_s + if mode.needs_index() { chunk_s + if *expanded { learned.expand_s.unwrap_or(0.0) } else { 0.0 } + learned.embed_s.get(&key).copied().unwrap_or(0.0) } else { 0.0 };
        let speed_before = speed(&pool);
        let (mut passed, mut total, mut ho_p, mut ho_t, mut times) = (0, 0, 0, 0, Vec::new());
        for (n, q) in set.questions.iter().enumerate() {
            if only.as_ref().is_some_and(|o| !o.contains(&(n + 1))) {
                continue;
            }
            let t = Instant::now();
            let reading = read(*mode, Sources { pool: &pool, corpus: &corpus, indexes: index.as_ref().map(|i| vec![docuzent_read::Shelf { index: i, first_part: 0 }]).unwrap_or_default(), embedder: embedder.as_deref() }, &q.q, ReadOptions::default(), &|_| {})?;
            let read_s = secs(t);
            let ta = Instant::now();
            let ans = pool.with_scratch(|llm| answer::answer_with(llm, &q.q, &reading.passages, *offleash, *think))?;
            let answer_s = secs(ta);
            let seconds = secs(t);
            let problems = check(q, &ans, *offleash);
            let pass = problems.is_empty();
            total += 1;
            passed += pass as usize;
            if q.set == "held-out" {
                ho_t += 1;
                ho_p += pass as usize;
            }
            times.push(seconds);
            eprintln!("[eval] {name} #{} {} ({seconds:.0}s) {}", n + 1, if pass { "PASS" } else { "FAIL" }, q.q);
            results.push(QuestionResult { run: name.clone(), n: n + 1, set: q.set.clone(), question: q.q.clone(), pass, problems, answer: ans, seconds, read_s, answer_s, parts_read: reading.read_closely + reading.scores.len(), chunks_retrieved: reading.retrieved });
        }
        let avg = if times.is_empty() { 0.0 } else { times.iter().sum::<f64>() / times.len() as f64 };
        let max = times.iter().cloned().fold(0.0, f64::max);
        let speed_after = speed(&pool);
        eprintln!("[eval] {name}: {passed}/{total} (held out {ho_p}/{ho_t}), {avg:.0}s average; model at {speed_before:.0} -> {speed_after:.0} tokens/s");
        summaries.push(RunSummary { run: name.clone(), speed_before, speed_after, mode: mode.as_str().into(), passed, total, held_out_passed: ho_p, held_out_total: ho_t, avg_s: avg, max_s: max, learn_s });
    }

    let mut report = Report { book: set.book.clone(), expected_title: set.expected_title.clone(), detected_title: detected, chars: text.len(), parts: corpus.part_count(), model, embedder: embed_id, learned: learned.clone(), runs: summaries, questions: results };
    std::fs::create_dir_all(&cli.out)?;
    // One report per book and model: a later evaluation of some of the runs replaces those runs and keeps the rest.
    let slug = format!("{}--{}", slugify(&set.book), slugify(report.model.trim_end_matches(".gguf")));
    let json_path = cli.out.join(format!("{slug}.json"));
    if let Ok(bytes) = std::fs::read(&json_path) {
        let earlier: Report = serde_json::from_slice(&bytes).with_context(|| format!("{} is not a report - move it aside", json_path.display()))?;
        let fresh: Vec<String> = report.runs.iter().map(|r| r.run.clone()).collect();
        let mut runs: Vec<RunSummary> = earlier.runs.into_iter().filter(|r| !fresh.contains(&r.run)).collect();
        runs.append(&mut report.runs);
        let mut questions: Vec<QuestionResult> = earlier.questions.into_iter().filter(|q| !fresh.contains(&q.run)).collect();
        questions.append(&mut report.questions);
        report.runs = runs;
        report.questions = questions;
    }
    std::fs::write(&json_path, serde_json::to_vec_pretty(&report)?)?;
    std::fs::write(cli.out.join(format!("{slug}.md")), markdown(&report))?;
    println!("{}", markdown(&report));
    eprintln!("[eval] written to {}", json_path.display());
    Ok(())
}

fn slugify(s: &str) -> String {
    s.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() || c == '.' { c } else { '-' }).collect::<String>().split('-').filter(|s| !s.is_empty()).collect::<Vec<_>>().join("-")
}

fn markdown(r: &Report) -> String {
    let mut s = format!("## {}\n\n", r.book);
    s.push_str(&format!("{} characters, {} parts; model `{}`; embeddings `{}`.\n", r.chars, r.parts, r.model, r.embedder.as_deref().unwrap_or("none (words only)")));
    if let Some(t) = &r.detected_title {
        s.push_str(&format!("Title read from the first pages: **{}**{} (from the {}){}.\n", t.title, t.author.as_ref().map(|a| format!(" by {a}")).unwrap_or_default(), t.source, r.expected_title.as_ref().map(|e| format!("; expected \"{e}\"")).unwrap_or_default()));
    }
    s.push_str("\n| mode | accuracy | held out | seconds / question (avg, worst) | learning (s) | model speed (tokens/s, before -> after) |\n|---|---|---|---|---|---|\n");
    for run in &r.runs {
        let suspect = run.speed_after < run.speed_before * 0.6 || run.speed_before < 20.0;
        s.push_str(&format!("| {} | {}/{} | {}/{} | {:.0}, {:.0} | {:.0} | {:.0} -> {:.0}{} |\n", run.run, run.passed, run.total, run.held_out_passed, run.held_out_total, run.avg_s, run.max_s, run.learn_s, run.speed_before, run.speed_after, if suspect { " (timings suspect)" } else { "" }));
    }
    let l = &r.learned;
    s.push_str(&format!(
        "\nSaved KV states: {:.1} GB{}. Indexes: {}.\n",
        l.kv_bytes as f64 / 1e9,
        l.kv_s.map(|x| format!(" ({x:.0} s to save)")).unwrap_or_default(),
        l.index_bytes.iter().map(|(k, v)| format!("{k} {:.1} MB", *v as f64 / 1e6)).collect::<Vec<_>>().join(", ")
    ));
    let failures: Vec<&QuestionResult> = r.questions.iter().filter(|q| !q.pass).collect();
    if !failures.is_empty() {
        s.push_str("\nFailures:\n\n");
        for f in failures {
            s.push_str(&format!("- **{}** #{} {} - {}\n  > {}\n", f.run, f.n, f.question, f.problems.join("; "), f.answer.replace('\n', " ").chars().take(300).collect::<String>()));
        }
    }
    s
}
