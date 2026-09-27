//! Single-document Q&A: parse one document (or a zip of them, one level
//! deep) via Docling, decide whether it fits the model's context window as
//! one chunk or needs question-aware map-reduce, and persist Ollama's
//! resumable generation context (see [`crate::generate`] for what that
//! actually is) to disk per `(model, context_size, document_hash)` so a
//! second question about an unchanged document skips reprocessing it.
//!
//! Two tiers, matching the two different lifetimes these caches make
//! sense at:
//! - **In memory**, for whichever one document is currently loaded - freed
//!   the moment a *different* document is loaded, so this process never
//!   holds more than one document's context resident at a time.
//! - **On disk** (via `kvcache`), surviving across documents and process
//!   restarts, LRU-evicted once over its byte budget.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use kvcache::Cache;

use crate::archive;
use crate::chunk::{self, DEFAULT_MAX_CHARS};
use crate::generate::{GenerateResponse, Generator};
use crate::hash;
use crate::ingest::{self, IngestOptions};
use crate::model_info;

/// Fraction of the context window reserved for the prompt template, the
/// question, and the model's own answer - never spent on document text.
const OVERHEAD_FRACTION: f32 = 0.2;
/// Rough chars-per-token estimate for English prose - good enough for a
/// sizing decision, not for exact token accounting (Ollama's API does not
/// expose a tokenizer endpoint to do better without bundling one).
const CHARS_PER_TOKEN: usize = 4;

struct CurrentDoc {
    doc_hash: String,
    text: String,
    /// `Some` once we've primed the model with this document's text at
    /// least once (single-chunk path only - map-reduce has no single
    /// "whole document" context to prime).
    context: Option<Vec<i64>>,
}

pub struct Session<G: Generator> {
    generator: G,
    model: String,
    context_length: u32,
    work_dir: PathBuf,
    disk_cache: Cache,
    current: Option<CurrentDoc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoadReport {
    pub chars: usize,
    pub fits_in_one_chunk: bool,
    /// True if a previously-computed context for this exact
    /// (model, context_size, document) was found on disk and reused.
    pub warm_from_disk: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AnswerReport {
    pub answer: String,
    pub used_map_reduce: bool,
    pub chunks_mapped: usize,
    pub timings: Vec<CallTiming>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CallTiming {
    pub label: String,
    pub prompt_eval_count: u64,
    pub prompt_eval_duration_ms: f64,
    pub eval_duration_ms: f64,
    pub wall_ms: f64,
}

impl<G: Generator> Session<G> {
    pub fn open(
        generator: G,
        model: impl Into<String>,
        context_length: u32,
        work_dir: PathBuf,
        disk_cache: Cache,
    ) -> Self {
        Self { generator, model: model.into(), context_length, work_dir, disk_cache, current: None }
    }

    fn max_doc_chars(&self) -> usize {
        let usable_tokens = (self.context_length as f32 * (1.0 - OVERHEAD_FRACTION)) as usize;
        usable_tokens * CHARS_PER_TOKEN
    }

    fn cache_key(&self, doc_hash: &str) -> String {
        format!("{}|{}|{doc_hash}", self.model, self.context_length)
    }

    /// Parses `source` into its combined text and content hash. A `.zip`
    /// is extracted one level deep (see [`crate::archive`]); every other
    /// file is parsed as itself. Multiple files (from a zip) are
    /// concatenated, each under a `=== filename ===` header.
    fn parse_document(&self, source: &Path) -> Result<(String, String)> {
        let doc_hash = hash::hash_file(source)?;
        let is_zip = source
            .extension()
            .map(|e| e.eq_ignore_ascii_case("zip"))
            .unwrap_or(false);

        let files: Vec<PathBuf> = if is_zip {
            let extract_dir = self.work_dir.join(format!("extracted-{doc_hash}"));
            archive::extract_one_level(source, &extract_dir)?
                .into_iter()
                .filter(|p| !p.extension().map(|e| e.eq_ignore_ascii_case("zip")).unwrap_or(false))
                .collect()
        } else {
            vec![source.to_path_buf()]
        };

        let mut combined = String::new();
        for file in &files {
            let out_dir = self.work_dir.join(format!("docling-{doc_hash}"));
            let Ok(report) = ingest::run(&IngestOptions {
                source: file.clone(),
                output: out_dir.clone(),
                to: "json".to_string(),
                device: "auto".to_string(),
                docling_bin: None,
            }) else {
                continue; // not every file in a zip is necessarily a document Docling can open
            };
            let stem = file.file_stem().unwrap_or_default().to_string_lossy().into_owned();
            let Some(json_name) = report.produced_files.iter().find(|f| f.starts_with(&stem)) else {
                continue;
            };
            let json_path = out_dir.join(json_name);
            // DEFAULT_MAX_CHARS just bounds one internal page-chunk here;
            // we decide OUR OWN chunking for map-reduce separately below.
            let page_chunks = chunk::chunk_docling_json(&json_path, DEFAULT_MAX_CHARS)?;
            if !combined.is_empty() {
                combined.push_str("\n\n");
            }
            combined.push_str(&format!("=== {} ===\n", file.file_name().unwrap_or_default().to_string_lossy()));
            for c in page_chunks {
                combined.push_str(&c.text);
                combined.push('\n');
            }
        }

        Ok((combined, doc_hash))
    }

    /// Loads a new document: clears whatever was held in memory for the
    /// previous one (freeing it), parses `source`, and checks the on-disk
    /// cache for a context already computed for this exact
    /// (model, context_size, document).
    pub fn load_document(&mut self, source: &Path) -> Result<LoadReport> {
        self.current = None; // drop the previous document's in-memory context now, not later

        let (text, doc_hash) = self.parse_document(source)?;
        let chars = text.chars().count();
        let fits_in_one_chunk = chars <= self.max_doc_chars();

        let cached: Option<Vec<i64>> = self
            .disk_cache
            .get(&self.cache_key(&doc_hash))?
            .map(|bytes| serde_json::from_slice(&bytes))
            .transpose()?;
        let warm_from_disk = cached.is_some();

        self.current = Some(CurrentDoc { doc_hash, text, context: cached });
        Ok(LoadReport { chars, fits_in_one_chunk, warm_from_disk })
    }

    /// Answers `question` about whichever document was last loaded.
    pub fn ask(&mut self, question: &str) -> Result<AnswerReport> {
        let max_chars = self.max_doc_chars();
        let current = self.current.as_ref().context("no document loaded - call load_document first")?;
        let doc_hash = current.doc_hash.clone();
        let text = current.text.clone();
        let primed_context = current.context.clone();
        let mut timings = Vec::new();

        if text.chars().count() <= max_chars {
            let context = match primed_context {
                Some(c) => c,
                None => {
                    let (resp, timing) = timed(&self.generator, &format!(
                        "You will be asked questions about the following document. Read it, then wait for the question.\n\n{text}"
                    ), None, "prime")?;
                    timings.push(timing);
                    self.disk_cache.put(&self.cache_key(&doc_hash), &serde_json::to_vec(&resp.context)?)?;
                    if let Some(c) = self.current.as_mut() {
                        c.context = Some(resp.context.clone());
                    }
                    resp.context
                }
            };

            let (resp, timing) = timed(&self.generator, &format!("Question: {question}\nAnswer:"), Some(&context), "answer")?;
            timings.push(timing);
            self.disk_cache.put(&self.cache_key(&doc_hash), &serde_json::to_vec(&resp.context)?)?;
            if let Some(c) = self.current.as_mut() {
                c.context = Some(resp.context.clone());
            }

            Ok(AnswerReport { answer: resp.response.trim().to_string(), used_map_reduce: false, chunks_mapped: 0, timings })
        } else {
            // Too big for one chunk: question-aware map-reduce. Headroom
            // beyond max_chars/2 for the map prompt's own wording.
            let pieces = chunk::split_to_max(&text, max_chars / 2);
            let mut extracted = Vec::new();
            for (i, piece) in pieces.iter().enumerate() {
                let prompt = format!(
                    "From the excerpt below, extract only information relevant to answering this question: {question}\nIf nothing in this excerpt is relevant, reply with exactly: NONE\n\nExcerpt:\n{piece}"
                );
                let (resp, timing) = timed(&self.generator, &prompt, None, &format!("map[{i}]"))?;
                timings.push(timing);
                let text = resp.response.trim().to_string();
                if !text.eq_ignore_ascii_case("none") && !text.is_empty() {
                    extracted.push(text);
                }
            }

            let facts = extracted.join("\n---\n");
            let reduce_prompt = format!(
                "Answer the question using only the extracted facts below. If they do not contain the answer, say so.\n\nQuestion: {question}\n\nExtracted facts:\n{facts}\n\nAnswer:"
            );
            let (resp, timing) = timed(&self.generator, &reduce_prompt, None, "reduce")?;
            timings.push(timing);

            Ok(AnswerReport { answer: resp.response.trim().to_string(), used_map_reduce: true, chunks_mapped: pieces.len(), timings })
        }
    }
}

fn timed<G: Generator>(
    generator: &G,
    prompt: &str,
    context: Option<&[i64]>,
    label: &str,
) -> Result<(GenerateResponse, CallTiming)> {
    let t0 = std::time::Instant::now();
    let resp = generator.generate(prompt, context)?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let timing = CallTiming {
        label: label.to_string(),
        prompt_eval_count: resp.prompt_eval_count,
        prompt_eval_duration_ms: resp.prompt_eval_duration as f64 / 1e6,
        eval_duration_ms: resp.eval_duration as f64 / 1e6,
        wall_ms,
    };
    Ok((resp, timing))
}

/// Infers the context window to use from a running Ollama server's own
/// report of the model, rather than requiring it to be configured by hand.
pub fn infer_context_length(host: &str, model: &str) -> Result<u32> {
    Ok(model_info::model_info(host, model)?.context_length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct FakeGenerator {
        calls: RefCell<Vec<String>>,
        next_context: RefCell<i64>,
    }

    impl FakeGenerator {
        fn new() -> Self {
            Self { calls: RefCell::new(Vec::new()), next_context: RefCell::new(0) }
        }
    }

    impl Generator for FakeGenerator {
        fn generate(&self, prompt: &str, context: Option<&[i64]>) -> Result<GenerateResponse> {
            self.calls.borrow_mut().push(prompt.to_string());
            let mut n = self.next_context.borrow_mut();
            *n += 1;
            let mut new_context = context.map(|c| c.to_vec()).unwrap_or_default();
            new_context.push(*n);
            let response = if prompt.contains("Question") {
                "a real answer".to_string()
            } else if prompt.contains("Excerpt") {
                "NONE".to_string()
            } else {
                "primed".to_string()
            };
            Ok(GenerateResponse {
                response,
                context: new_context,
                prompt_eval_count: prompt.len() as u64,
                prompt_eval_duration: if context.is_some() { 1_000_000 } else { 100_000_000 },
                eval_count: 5,
                eval_duration: 1_000_000,
                total_duration: 0,
            })
        }
    }

    fn temp_cache() -> (Cache, std::path::PathBuf) {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("docuzent-core-session-test-{}-{n}.redb", std::process::id()));
        (Cache::open(&path, 10_000_000).unwrap(), path)
    }

    fn session_with_current(generator: FakeGenerator, text: &str, context_length: u32) -> (Session<FakeGenerator>, std::path::PathBuf) {
        let (cache, cache_path) = temp_cache();
        let mut session = Session::open(generator, "test-model", context_length, std::env::temp_dir(), cache);
        session.current = Some(CurrentDoc { doc_hash: "testhash".to_string(), text: text.to_string(), context: None });
        (session, cache_path)
    }

    #[test]
    fn ask_without_loading_a_document_fails() {
        let (cache, path) = temp_cache();
        let mut session = Session::open(FakeGenerator::new(), "m", 4096, std::env::temp_dir(), cache);
        assert!(session.ask("anything").is_err());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn small_document_uses_single_chunk_path_and_primes_once() {
        let (mut session, cache_path) = session_with_current(FakeGenerator::new(), "a short document", 100_000);
        let report = session.ask("what is this about?").unwrap();
        assert!(!report.used_map_reduce);
        assert_eq!(report.answer, "a real answer");
        // prime + answer = 2 calls, both recorded
        assert_eq!(report.timings.len(), 2);
        assert_eq!(report.timings[0].label, "prime");
        assert_eq!(report.timings[1].label, "answer");
        std::fs::remove_file(&cache_path).ok();
    }

    #[test]
    fn second_question_reuses_the_primed_context_without_repriming() {
        let (mut session, cache_path) = session_with_current(FakeGenerator::new(), "a short document", 100_000);
        session.ask("first question").unwrap();
        let report = session.ask("second question").unwrap();
        // Only "answer", no second "prime" - the in-memory context was reused.
        assert_eq!(report.timings.len(), 1);
        assert_eq!(report.timings[0].label, "answer");
        std::fs::remove_file(&cache_path).ok();
    }

    #[test]
    fn loading_a_new_document_clears_the_in_memory_context() {
        let (mut session, _cache_path) = session_with_current(FakeGenerator::new(), "a short document", 100_000);
        session.ask("first question").unwrap();
        assert!(session.current.as_ref().unwrap().context.is_some());

        // Simulate loading a different document by clearing directly -
        // load_document() itself needs a real Docling run, tested
        // separately in the integration test.
        session.current = None;
        assert!(session.ask("anything").is_err(), "no document loaded after clearing");
    }

    #[test]
    fn oversized_document_uses_map_reduce() {
        let big_text = "word ".repeat(100_000); // far larger than a tiny context window allows
        let (mut session, cache_path) = session_with_current(FakeGenerator::new(), &big_text, 128);
        let report = session.ask("what is this about?").unwrap();
        assert!(report.used_map_reduce);
        assert!(report.chunks_mapped > 1);
        // one map call per chunk, plus one reduce call
        assert_eq!(report.timings.len(), report.chunks_mapped + 1);
        assert_eq!(report.timings.last().unwrap().label, "reduce");
        std::fs::remove_file(&cache_path).ok();
    }
}
