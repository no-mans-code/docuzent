//! Single- or multi-document Q&A: parse one or more documents (zips
//! extracted one level deep) via Docling, decide whether the combined text
//! fits the model's context window as one chunk or needs question-aware
//! map-reduce, and manage Ollama's resumable generation context (see
//! [`crate::generate`] for what that actually is) under one of three
//! explicit modes.
//!
//! **Modes** ([`Mode`]):
//! - [`Mode::Swap`]: always check the on-disk LLM-context cache first and
//!   always persist to it - today's original behavior.
//! - [`Mode::Raw`]: never touch the disk cache, and never reuse context
//!   in-memory across questions either - every single question pays a
//!   full cold reprocess. The honest "no reuse at all" baseline, so
//!   swap-vs-raw is a real, measurable A/B inside the product rather than
//!   a fixed choice.
//! - [`Mode::Adaptive`]: the default. When a disk-cached context exists
//!   for a document, predicts whether reusing it or a fresh cold prime
//!   will actually be faster - using [`ollama_kv_profiler::predictor`],
//!   the same crossover logic validated empirically in that project, fed
//!   real numbers from a persisted, self-calibrating [`SpeedProfile`]
//!   rather than dedicated calibration round-trips (a real product
//!   shouldn't pay ~10 extra calls per document load just to decide this).
//!
//! **Two cache tiers**, serving different jobs:
//! - **In memory**, for whichever one document is currently loaded - freed
//!   the moment a *different* document is loaded.
//! - **On disk**: two separate caches. The LLM-context cache (`kvcache`,
//!   keyed by `(model, context_size, document_hash)`) persists Ollama's
//!   resumable context across process restarts. The [`docling_cache`]
//!   persists Docling's own (slow, GPU/CPU-bound) parsing output, keyed
//!   per source file - independent of model or mode, since parsing a
//!   file's text is the same work regardless of what's later done with
//!   it. Loading a genuinely *different* document set immediately evicts
//!   the previous one's LLM-context entry (not left to LRU aging) - see
//!   [`Session::load_documents`].

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use kvcache::Cache;
use ollama_kv_profiler::hardware;
use ollama_kv_profiler::predictor::{self, ModelSpeed, SystemSnapshot};
use serde::Serialize;

use crate::archive;
use crate::chunk::{self, DEFAULT_MAX_CHARS};
use crate::docling_cache::DoclingCache;
use crate::generate::{GenerateResponse, Generator};
use crate::hash;
use crate::ingest::{self, IngestOptions};
use crate::model_info;
use crate::speed_profile::SpeedProfile;

/// Fraction of the context window reserved for the prompt template, the
/// question, and the model's own answer - never spent on document text.
const OVERHEAD_FRACTION: f32 = 0.2;
/// Rough chars-per-token estimate for English prose - good enough for a
/// sizing decision, not for exact token accounting (Ollama's API does not
/// expose a tokenizer endpoint to do better without bundling one).
const CHARS_PER_TOKEN: usize = 4;
/// One-time real disk-bandwidth measurement size, matching
/// `ollama-kv-profiler`'s own usage.
const DISK_BENCH_BYTES: usize = 256 * 1024 * 1024;

/// How `Session` decides whether to reuse the LLM-context cache. See the
/// module doc comment for what each variant actually does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Swap,
    Raw,
    Adaptive,
}

impl Mode {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "swap" => Ok(Mode::Swap),
            "raw" => Ok(Mode::Raw),
            "adaptive" => Ok(Mode::Adaptive),
            other => anyhow::bail!("unknown mode `{other}` - expected one of: swap, raw, adaptive"),
        }
    }
}

struct CurrentDoc {
    doc_hash: String,
    text: String,
    /// The context found on disk at load time, if any - not necessarily
    /// *used* yet. `Adaptive` mode decides lazily, on the first real
    /// question, whether to actually warm-start from it.
    disk_context: Option<Vec<i64>>,
    /// The context this conversation is actually using, once established
    /// (by whichever path: warm-started from `disk_context`, or a fresh
    /// cold prime) - `None` until the first question is answered.
    active_context: Option<Vec<i64>>,
}

pub struct Session<G: Generator> {
    generator: G,
    model: String,
    context_length: u32,
    work_dir: PathBuf,
    mode: Mode,
    disk_cache: Cache,
    docling_cache: DoclingCache,
    speed_profile: Option<SpeedProfile>,
    disk_bandwidth_bytes_per_sec: f64,
    current: Option<CurrentDoc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct LoadReport {
    pub chars: usize,
    pub fits_in_one_chunk: bool,
    /// True if a previously-computed context for this exact
    /// `(model, context_size, document)` was found on disk - not
    /// necessarily what gets used (see [`Mode::Adaptive`]).
    pub warm_from_disk: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnswerReport {
    pub answer: String,
    pub used_map_reduce: bool,
    pub chunks_mapped: usize,
    /// Set only in `Adaptive` mode, and only when a real decision was
    /// made (a disk-cached context existed to choose between) - `None`
    /// otherwise (`Swap`/`Raw` have nothing to decide; a first-ever
    /// document has no disk context to weigh against).
    pub adaptive_decision: Option<AdaptiveDecision>,
    pub timings: Vec<CallTiming>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct AdaptiveDecision {
    pub predicted_swap_ms: f64,
    pub predicted_raw_ms: f64,
    pub chose_swap: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
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
        mode: Mode,
        disk_cache: Cache,
        docling_cache: DoclingCache,
    ) -> Result<Self> {
        let model = model.into();
        let speed_profile = SpeedProfile::load(&disk_cache, &model, context_length)?;
        // One real, one-time measurement - disk bandwidth doesn't
        // meaningfully change moment to moment the way VRAM/RAM
        // contention does, so unlike those (queried live per decision,
        // see `predict_swap_wins`) this is measured once up front.
        let disk_bandwidth_bytes_per_sec = hardware::measure_disk_read_bandwidth_bytes_per_sec(DISK_BENCH_BYTES)?;
        Ok(Self {
            generator,
            model,
            context_length,
            work_dir,
            mode,
            disk_cache,
            docling_cache,
            speed_profile,
            disk_bandwidth_bytes_per_sec,
            current: None,
        })
    }

    fn max_doc_chars(&self) -> usize {
        let usable_tokens = (self.context_length as f32 * (1.0 - OVERHEAD_FRACTION)) as usize;
        usable_tokens * CHARS_PER_TOKEN
    }

    fn cache_key(&self, doc_hash: &str) -> String {
        format!("{}|{}|{doc_hash}", self.model, self.context_length)
    }

    fn save_speed_profile(&self) -> Result<()> {
        if let Some(profile) = &self.speed_profile {
            profile.save(&self.disk_cache, &self.model, self.context_length)?;
        }
        Ok(())
    }

    /// Blends a real ingest call's own timing into the persisted profile,
    /// seeding one from scratch on this `(model, context_length)`'s first
    /// ever observation. Takes the raw response (not just `CallTiming`)
    /// since it needs `eval_count` for the eval-tok/s side, which
    /// `CallTiming` doesn't carry (no caller outside this module needs it).
    fn record_ingest_from_response(&mut self, resp: &GenerateResponse, timing: &CallTiming) {
        let prefill_tps = resp.prompt_eval_count as f64 / (timing.prompt_eval_duration_ms / 1000.0).max(1e-9);
        let eval_tps = resp.eval_count as f64 / (timing.eval_duration_ms / 1000.0).max(1e-9);
        match &mut self.speed_profile {
            Some(profile) => profile.update_from_ingest(prefill_tps, eval_tps),
            None => self.speed_profile = Some(SpeedProfile::seed_from_ingest(prefill_tps, eval_tps)),
        }
        let _ = self.save_speed_profile();
    }

    /// Every "answer" call passes some context (whichever way it was
    /// established) - a real swap-shaped observation regardless of mode,
    /// so worth recording even in `Raw` (free, real data for whenever
    /// `Adaptive` is used with this model later).
    fn record_swap(&mut self, resp: &GenerateResponse, timing: &CallTiming) {
        let overhead_ms = (timing.wall_ms - timing.prompt_eval_duration_ms - timing.eval_duration_ms).max(0.0);
        if let Some(profile) = &mut self.speed_profile {
            profile.update_from_swap(resp.eval_count as f64, overhead_ms);
            let _ = self.save_speed_profile();
        }
    }

    /// Predicts whether reusing `disk_context` will be faster than a
    /// fresh cold prime of `text`, using this model's persisted
    /// [`SpeedProfile`] and a live VRAM/RAM snapshot. `false` (fall back
    /// to a cold prime) when there isn't yet enough real data to predict
    /// from - the honest bootstrap case for a never-before-seen model.
    fn predict_swap(&self, text: &str, question: &str, disk_context: &[i64]) -> Option<AdaptiveDecision> {
        let profile = self.speed_profile.as_ref()?;
        if !profile.has_swap_data() {
            return None;
        }
        let document_tokens = (text.chars().count() / CHARS_PER_TOKEN).max(1) as u64;
        let follow_up_tokens = (question.chars().count() / CHARS_PER_TOKEN).max(1) as f64;
        let disk_bytes_per_token = serde_json::to_vec(disk_context)
            .map(|v| v.len() as f64 / (disk_context.len().max(1) as f64))
            .unwrap_or(8.0);
        let speed = ModelSpeed {
            prefill_tokens_per_sec: profile.prefill_tokens_per_sec,
            eval_tokens_per_sec: profile.eval_tokens_per_sec,
            expected_answer_tokens: profile.expected_answer_tokens,
            disk_bytes_per_token,
            fixed_overhead_ms: profile.fixed_overhead_ms,
        };
        let snapshot = SystemSnapshot {
            vram_free_fraction: hardware::vram_free_fraction(),
            ram_free_fraction: hardware::ram_free_fraction(),
            disk_bandwidth_bytes_per_sec: self.disk_bandwidth_bytes_per_sec,
        };
        let prediction = predictor::predict(document_tokens, &speed, follow_up_tokens, &snapshot, true);
        let predicted_swap_ms = prediction.predicted_swap_ms?;
        Some(AdaptiveDecision {
            predicted_swap_ms,
            predicted_raw_ms: prediction.predicted_ingest_ms,
            chose_swap: predicted_swap_ms < prediction.predicted_ingest_ms,
        })
    }

    /// Parses `sources` (each either a document Docling can open, or a
    /// `.zip` of them, extracted one level deep) into their combined text
    /// and a combined content hash. Per-file text is cached in
    /// [`DoclingCache`], keyed by that file's own content hash - a file
    /// reused across different document sets is parsed once. The combined
    /// hash is order-independent (hash of the *sorted* per-file hashes),
    /// so the same set of files always keys identically regardless of
    /// the order they were passed in.
    fn parse_documents(&self, sources: &[PathBuf]) -> Result<(String, String)> {
        let mut files: Vec<PathBuf> = Vec::new();
        for source in sources {
            let is_zip = source.extension().map(|e| e.eq_ignore_ascii_case("zip")).unwrap_or(false);
            if is_zip {
                let zip_hash = hash::hash_file(source)?;
                let extract_dir = self.work_dir.join(format!("extracted-{zip_hash}"));
                files.extend(
                    archive::extract_one_level(source, &extract_dir)?
                        .into_iter()
                        .filter(|p| !p.extension().map(|e| e.eq_ignore_ascii_case("zip")).unwrap_or(false)),
                );
            } else {
                files.push(source.clone());
            }
        }

        let file_hashes: Vec<(PathBuf, String)> =
            files.iter().map(|f| Ok::<_, anyhow::Error>((f.clone(), hash::hash_file(f)?))).collect::<Result<_>>()?;

        let mut sorted_hashes: Vec<&str> = file_hashes.iter().map(|(_, h)| h.as_str()).collect();
        sorted_hashes.sort_unstable();
        let combined_hash = hash::hash_bytes(sorted_hashes.join(",").as_bytes());

        let mut combined = String::new();
        for (file, file_hash) in &file_hashes {
            let text = match self.docling_cache.get(file_hash)? {
                Some(cached) => cached,
                None => {
                    let out_dir = self.work_dir.join(format!("docling-{file_hash}"));
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
                    let page_chunks = chunk::chunk_docling_json(&json_path, DEFAULT_MAX_CHARS)?;
                    let mut file_text = String::new();
                    for c in page_chunks {
                        file_text.push_str(&c.text);
                        file_text.push('\n');
                    }
                    self.docling_cache.put(file_hash, &file_text)?;
                    file_text
                }
            };
            if !combined.is_empty() {
                combined.push_str("\n\n");
            }
            combined.push_str(&format!("=== {} ===\n", file.file_name().unwrap_or_default().to_string_lossy()));
            combined.push_str(&text);
        }

        Ok((combined, combined_hash))
    }

    /// If `new_doc_hash` differs from whatever was previously loaded,
    /// evicts the previous document set's LLM-context disk cache entry
    /// immediately rather than leaving it to LRU aging - see module doc
    /// comment. A no-op when there's no previous document, or when it's
    /// the same one being reloaded.
    fn evict_previous_if_different(&mut self, new_doc_hash: &str) -> Result<()> {
        if let Some(previous) = &self.current {
            if previous.doc_hash != new_doc_hash {
                self.disk_cache.remove(&self.cache_key(&previous.doc_hash))?;
            }
        }
        Ok(())
    }

    /// Loads one or more documents as a single combined corpus. If this
    /// is a genuinely *different* document set than whatever was
    /// previously loaded, the previous set's LLM-context disk cache entry
    /// is evicted immediately (not left to LRU aging) - see module doc
    /// comment.
    pub fn load_documents(&mut self, sources: &[PathBuf]) -> Result<LoadReport> {
        let (text, doc_hash) = self.parse_documents(sources)?;
        let chars = text.chars().count();
        let fits_in_one_chunk = chars <= self.max_doc_chars();

        self.evict_previous_if_different(&doc_hash)?;
        self.current = None; // drop the previous document's in-memory context now, not later

        let disk_context: Option<Vec<i64>> = self
            .disk_cache
            .get(&self.cache_key(&doc_hash))?
            .map(|bytes| serde_json::from_slice(&bytes))
            .transpose()?;
        let warm_from_disk = disk_context.is_some();

        self.current = Some(CurrentDoc { doc_hash, text, disk_context, active_context: None });
        Ok(LoadReport { chars, fits_in_one_chunk, warm_from_disk })
    }

    /// Convenience wrapper for the common single-file case.
    pub fn load_document(&mut self, source: &Path) -> Result<LoadReport> {
        self.load_documents(&[source.to_path_buf()])
    }

    /// Answers `question` about whichever document(s) were last loaded.
    pub fn ask(&mut self, question: &str) -> Result<AnswerReport> {
        let max_chars = self.max_doc_chars();
        let current = self.current.as_ref().context("no document loaded - call load_documents first")?;
        let doc_hash = current.doc_hash.clone();
        let text = current.text.clone();
        let mut timings = Vec::new();

        if text.chars().count() <= max_chars {
            // Raw mode never reuses context, in-memory or disk, across
            // separate questions - every question pays a full cold
            // reprocess (see module doc comment).
            let existing_active = if self.mode == Mode::Raw { None } else { current.active_context.clone() };

            let mut adaptive_decision = None;
            let context = match existing_active {
                Some(c) => c,
                None => {
                    let disk_context = if self.mode == Mode::Raw { None } else { current.disk_context.clone() };
                    let use_swap = match (self.mode, &disk_context) {
                        (Mode::Raw, _) => false,
                        (Mode::Swap, Some(_)) => true,
                        (Mode::Swap, None) => false,
                        (Mode::Adaptive, Some(ctx)) => {
                            let decision = self.predict_swap(&text, question, ctx);
                            let chose_swap = decision.as_ref().map(|d| d.chose_swap).unwrap_or(false);
                            adaptive_decision = decision;
                            chose_swap
                        }
                        (Mode::Adaptive, None) => false,
                    };

                    if use_swap {
                        disk_context.expect("use_swap only true when disk_context is Some")
                    } else {
                        let prime_prompt = format!(
                            "You will be asked questions about the following document. Read it, then wait for the question.\n\n{text}"
                        );
                        let (resp, timing) = timed(&self.generator, &prime_prompt, None, "prime")?;
                        self.record_ingest_from_response(&resp, &timing);
                        timings.push(timing);
                        if self.mode != Mode::Raw {
                            self.disk_cache.put(&self.cache_key(&doc_hash), &serde_json::to_vec(&resp.context)?)?;
                        }
                        if let Some(c) = self.current.as_mut() {
                            c.active_context = Some(resp.context.clone());
                        }
                        resp.context
                    }
                }
            };

            let (resp, timing) = timed(&self.generator, &format!("Question: {question}\nAnswer:"), Some(&context), "answer")?;
            self.record_swap(&resp, &timing);
            timings.push(timing);
            if self.mode != Mode::Raw {
                self.disk_cache.put(&self.cache_key(&doc_hash), &serde_json::to_vec(&resp.context)?)?;
                if let Some(c) = self.current.as_mut() {
                    c.active_context = Some(resp.context.clone());
                }
            }

            Ok(AnswerReport {
                answer: resp.response.trim().to_string(),
                used_map_reduce: false,
                chunks_mapped: 0,
                adaptive_decision,
                timings,
            })
        } else {
            // Too big for one chunk: question-aware map-reduce. Every
            // call here is already context-free (no reuse between
            // chunks), so mode has no further effect on this path - it's
            // already the "raw" shape by construction.
            let pieces = chunk::split_to_max(&text, max_chars / 2);
            let mut extracted = Vec::new();
            for (i, piece) in pieces.iter().enumerate() {
                let prompt = format!(
                    "From the excerpt below, extract only information relevant to answering this question: {question}\nIf nothing in this excerpt is relevant, reply with exactly: NONE\n\nExcerpt:\n{piece}"
                );
                let (resp, timing) = timed(&self.generator, &prompt, None, &format!("map[{i}]"))?;
                self.record_ingest_from_response(&resp, &timing);
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
            self.record_ingest_from_response(&resp, &timing);
            timings.push(timing);

            Ok(AnswerReport {
                answer: resp.response.trim().to_string(),
                used_map_reduce: true,
                chunks_mapped: pieces.len(),
                adaptive_decision: None,
                timings,
            })
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

    fn temp_cache(label: &str) -> (Cache, std::path::PathBuf) {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("docuzent-core-session-test-{label}-{}-{n}.redb", std::process::id()));
        (Cache::open(&path, 10_000_000).unwrap(), path)
    }

    fn temp_docling_cache(label: &str) -> (DoclingCache, std::path::PathBuf) {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("docuzent-core-session-test-docling-{label}-{}-{n}.redb", std::process::id()));
        (DoclingCache::open(&path, 10_000_000).unwrap(), path)
    }

    /// Builds a session without paying `open()`'s real disk-bandwidth
    /// measurement - tests construct the struct directly since they don't
    /// exercise adaptive-mode's disk-bandwidth-dependent path anyway
    /// (that needs a real `predict_swap` call, covered by the dedicated
    /// adaptive-mode tests below with an explicit fast/slow speed profile).
    fn session_with_current(generator: FakeGenerator, text: &str, context_length: u32, mode: Mode) -> (Session<FakeGenerator>, std::path::PathBuf, std::path::PathBuf) {
        let (cache, cache_path) = temp_cache("current");
        let (docling_cache, docling_path) = temp_docling_cache("current");
        let mut session = Session {
            generator,
            model: "test-model".to_string(),
            context_length,
            work_dir: std::env::temp_dir(),
            mode,
            disk_cache: cache,
            docling_cache,
            speed_profile: None,
            disk_bandwidth_bytes_per_sec: 3_000_000_000.0,
            current: None,
        };
        session.current = Some(CurrentDoc { doc_hash: "testhash".to_string(), text: text.to_string(), disk_context: None, active_context: None });
        (session, cache_path, docling_path)
    }

    #[test]
    fn ask_without_loading_a_document_fails() {
        let (mut session, cache_path, docling_path) = session_with_current(FakeGenerator::new(), "x", 4096, Mode::Swap);
        session.current = None;
        assert!(session.ask("anything").is_err());
        std::fs::remove_file(&cache_path).ok();
        std::fs::remove_file(&docling_path).ok();
    }

    #[test]
    fn small_document_uses_single_chunk_path_and_primes_once() {
        let (mut session, cache_path, docling_path) = session_with_current(FakeGenerator::new(), "a short document", 100_000, Mode::Swap);
        let report = session.ask("what is this about?").unwrap();
        assert!(!report.used_map_reduce);
        assert_eq!(report.answer, "a real answer");
        // prime + answer = 2 calls, both recorded
        assert_eq!(report.timings.len(), 2);
        assert_eq!(report.timings[0].label, "prime");
        assert_eq!(report.timings[1].label, "answer");
        std::fs::remove_file(&cache_path).ok();
        std::fs::remove_file(&docling_path).ok();
    }

    #[test]
    fn second_question_reuses_the_primed_context_without_repriming_in_swap_mode() {
        let (mut session, cache_path, docling_path) = session_with_current(FakeGenerator::new(), "a short document", 100_000, Mode::Swap);
        session.ask("first question").unwrap();
        let report = session.ask("second question").unwrap();
        // Only "answer", no second "prime" - the in-memory context was reused.
        assert_eq!(report.timings.len(), 1);
        assert_eq!(report.timings[0].label, "answer");
        std::fs::remove_file(&cache_path).ok();
        std::fs::remove_file(&docling_path).ok();
    }

    #[test]
    fn raw_mode_never_reuses_context_even_within_the_same_session() {
        let (mut session, cache_path, docling_path) = session_with_current(FakeGenerator::new(), "a short document", 100_000, Mode::Raw);
        session.ask("first question").unwrap();
        let report = session.ask("second question").unwrap();
        // Raw mode re-primes every single question - no in-memory reuse.
        assert_eq!(report.timings.len(), 2);
        assert_eq!(report.timings[0].label, "prime");
        assert_eq!(report.timings[1].label, "answer");
        std::fs::remove_file(&cache_path).ok();
        std::fs::remove_file(&docling_path).ok();
    }

    #[test]
    fn raw_mode_never_writes_to_the_disk_cache() {
        let (mut session, cache_path, docling_path) = session_with_current(FakeGenerator::new(), "a short document", 100_000, Mode::Raw);
        session.ask("a question").unwrap();
        assert_eq!(session.disk_cache.get(&session.cache_key("testhash")).unwrap(), None);
        std::fs::remove_file(&cache_path).ok();
        std::fs::remove_file(&docling_path).ok();
    }

    #[test]
    fn loading_a_new_document_clears_the_in_memory_context() {
        let (mut session, cache_path, docling_path) = session_with_current(FakeGenerator::new(), "a short document", 100_000, Mode::Swap);
        session.ask("first question").unwrap();
        assert!(session.current.as_ref().unwrap().active_context.is_some());

        // Simulate loading a different document by clearing directly -
        // load_documents() itself needs a real Docling run, tested
        // separately in the integration test.
        session.current = None;
        assert!(session.ask("anything").is_err(), "no document loaded after clearing");
        std::fs::remove_file(&cache_path).ok();
        std::fs::remove_file(&docling_path).ok();
    }

    #[test]
    fn oversized_document_uses_map_reduce() {
        let big_text = "word ".repeat(100_000); // far larger than a tiny context window allows
        let (mut session, cache_path, docling_path) = session_with_current(FakeGenerator::new(), &big_text, 128, Mode::Swap);
        let report = session.ask("what is this about?").unwrap();
        assert!(report.used_map_reduce);
        assert!(report.chunks_mapped > 1);
        // one map call per chunk, plus one reduce call
        assert_eq!(report.timings.len(), report.chunks_mapped + 1);
        assert_eq!(report.timings.last().unwrap().label, "reduce");
        std::fs::remove_file(&cache_path).ok();
        std::fs::remove_file(&docling_path).ok();
    }

    #[test]
    fn loading_a_different_document_evicts_the_previous_ones_disk_entry_immediately() {
        let (mut session, cache_path, docling_path) = session_with_current(FakeGenerator::new(), "doc a", 100_000, Mode::Swap);
        // Prime doc A for real, so it has a genuine disk-cache entry to evict.
        session.ask("about doc a").unwrap();
        let key_a = session.cache_key("testhash");
        assert!(session.disk_cache.get(&key_a).unwrap().is_some(), "doc A should have a real disk entry before switching");

        session.evict_previous_if_different("a-different-hash").unwrap();

        assert_eq!(session.disk_cache.get(&key_a).unwrap(), None, "doc A's entry should be evicted immediately on switching to a different document");
        std::fs::remove_file(&cache_path).ok();
        std::fs::remove_file(&docling_path).ok();
    }

    #[test]
    fn reloading_the_same_document_does_not_evict_its_own_entry() {
        let (mut session, cache_path, docling_path) = session_with_current(FakeGenerator::new(), "doc a", 100_000, Mode::Swap);
        session.ask("about doc a").unwrap();
        let key_a = session.cache_key("testhash");
        assert!(session.disk_cache.get(&key_a).unwrap().is_some());

        session.evict_previous_if_different("testhash").unwrap(); // same hash as current

        assert!(session.disk_cache.get(&key_a).unwrap().is_some(), "reloading the same document must not evict its own cache entry");
        std::fs::remove_file(&cache_path).ok();
        std::fs::remove_file(&docling_path).ok();
    }

    #[test]
    fn adaptive_mode_with_no_prior_speed_data_falls_back_to_a_cold_prime() {
        let (mut session, cache_path, docling_path) = session_with_current(FakeGenerator::new(), "a short document", 100_000, Mode::Adaptive);
        // Seed a disk_context as if a previous run had cached one, but
        // with no speed_profile at all - nothing to predict from yet.
        session.current.as_mut().unwrap().disk_context = Some(vec![1, 2, 3]);
        let report = session.ask("what is this about?").unwrap();
        assert!(report.adaptive_decision.is_none(), "no profile yet - should not have made a real decision");
        assert_eq!(report.timings[0].label, "prime", "should have cold-primed with no data to predict from");
        std::fs::remove_file(&cache_path).ok();
        std::fs::remove_file(&docling_path).ok();
    }

    #[test]
    fn adaptive_mode_swaps_when_a_fast_model_makes_swap_clearly_cheaper() {
        // A real document, not a token or two - large enough that
        // reingesting the *whole* thing costs meaningfully more than
        // reprocessing just the (short) follow-up question, which is what
        // makes swap the right call here. A tiny document would cost less
        // to reingest than the follow-up question itself, which isn't a
        // realistic case this decision needs to get right.
        let big_text = "word ".repeat(400);
        let (mut session, cache_path, docling_path) = session_with_current(FakeGenerator::new(), &big_text, 100_000, Mode::Adaptive);
        session.current.as_mut().unwrap().disk_context = Some(vec![1, 2, 3]);
        // A profile with real swap data and a slow prefill speed makes a
        // fresh cold prime clearly the expensive option.
        session.speed_profile = Some(SpeedProfile::seed_from_ingest(1.0, 1000.0));
        session.speed_profile.as_mut().unwrap().update_from_swap(5.0, 1.0);
        let report = session.ask("what is this about?").unwrap();
        let decision = report.adaptive_decision.expect("should have made a real decision");
        assert!(decision.chose_swap, "swap should clearly win against a very slow prefill speed");
        assert_eq!(report.timings.len(), 1, "swap path skips the prime call entirely");
        assert_eq!(report.timings[0].label, "answer");
        std::fs::remove_file(&cache_path).ok();
        std::fs::remove_file(&docling_path).ok();
    }

    // Real-Ollama tests below - `#[ignore]`d by default (run with
    // `cargo test -- --ignored`), since they need a live `ollama serve`
    // with `qwen2.5:3b` pulled. They bypass Docling entirely by hand-
    // populating `current` with real text rather than going through
    // `load_documents` - the Docling-parsing path itself is legacy code
    // this session didn't change (beyond the new per-file cache wrapper,
    // already covered by `docling_cache`'s own tests with fake text);
    // what's genuinely new here is the mode logic, eviction, and adaptive
    // prediction, all of which operate purely on already-parsed text and
    // real Ollama responses - exactly what these tests exercise for real.

    use crate::generate::OllamaClient;

    const REAL_HOST: &str = "http://localhost:11434";
    const REAL_MODEL: &str = "qwen2.5:3b";

    fn real_session(mode: Mode, cache_path: &Path, docling_path: &Path) -> Session<OllamaClient> {
        let context_length = infer_context_length(REAL_HOST, REAL_MODEL).expect("ollama must be running with qwen2.5:3b pulled");
        let cache = Cache::open(cache_path, 1_000_000_000).unwrap();
        let docling_cache = DoclingCache::open(docling_path, 1_000_000_000).unwrap();
        Session::open(OllamaClient::new(REAL_HOST, REAL_MODEL, context_length), REAL_MODEL, context_length, std::env::temp_dir(), mode, cache, docling_cache).unwrap()
    }

    fn set_current(session: &mut Session<OllamaClient>, doc_hash: &str, text: &str, disk_context: Option<Vec<i64>>) {
        session.current = Some(CurrentDoc { doc_hash: doc_hash.to_string(), text: text.to_string(), disk_context, active_context: None });
    }

    #[test]
    #[ignore]
    fn real_ollama_swap_mode_persists_context_and_reuses_it_on_a_fresh_session() {
        let cache_path = std::env::temp_dir().join(format!("docuzent-real-swap-{}.redb", std::process::id()));
        let docling_path = std::env::temp_dir().join(format!("docuzent-real-swap-docling-{}.redb", std::process::id()));
        let _ = std::fs::remove_file(&cache_path);
        let text = "The quarterly report shows revenue growth across all regions, driven primarily by strong performance in the enterprise software division. ".repeat(20);

        let mut session1 = real_session(Mode::Swap, &cache_path, &docling_path);
        set_current(&mut session1, "real-swap-doc", &text, None);
        let r1 = session1.ask("What drove the growth?").unwrap();
        assert_eq!(r1.timings[0].label, "prime", "first-ever call for this doc should cold-prime");
        assert!(!r1.answer.is_empty());
        drop(session1);

        let mut session2 = real_session(Mode::Swap, &cache_path, &docling_path);
        let disk_context: Option<Vec<i64>> = session2.disk_cache.get(&session2.cache_key("real-swap-doc")).unwrap().map(|b| serde_json::from_slice(&b).unwrap());
        set_current(&mut session2, "real-swap-doc", &text, disk_context);
        let r2 = session2.ask("Which division was strongest?").unwrap();
        assert_eq!(r2.timings.len(), 1, "a fresh session should reuse the disk-persisted context, not re-prime");
        assert_eq!(r2.timings[0].label, "answer");
        assert!(!r2.answer.is_empty());
        println!("real swap-mode answers: {:?} / {:?}", r1.answer, r2.answer);

        let _ = std::fs::remove_file(&cache_path);
        let _ = std::fs::remove_file(&docling_path);
    }

    #[test]
    #[ignore]
    fn real_ollama_raw_mode_never_persists_and_never_reuses() {
        let cache_path = std::env::temp_dir().join(format!("docuzent-real-raw-{}.redb", std::process::id()));
        let docling_path = std::env::temp_dir().join(format!("docuzent-real-raw-docling-{}.redb", std::process::id()));
        let _ = std::fs::remove_file(&cache_path);
        let text = "The quarterly report shows revenue growth across all regions. ".repeat(10);

        let mut session = real_session(Mode::Raw, &cache_path, &docling_path);
        set_current(&mut session, "real-raw-doc", &text, None);
        let r1 = session.ask("Summarize this.").unwrap();
        let r2 = session.ask("Summarize it again.").unwrap();
        assert_eq!(r1.timings[0].label, "prime");
        assert_eq!(r2.timings[0].label, "prime", "raw mode re-primes every single question, even within the same session");
        assert_eq!(session.disk_cache.get(&session.cache_key("real-raw-doc")).unwrap(), None, "raw mode must never write to the disk cache");
        println!("real raw-mode answers: {:?} / {:?}", r1.answer, r2.answer);

        let _ = std::fs::remove_file(&cache_path);
        let _ = std::fs::remove_file(&docling_path);
    }

    #[test]
    #[ignore]
    fn real_ollama_adaptive_mode_builds_a_real_profile_then_makes_a_real_decision() {
        let cache_path = std::env::temp_dir().join(format!("docuzent-real-adaptive-{}.redb", std::process::id()));
        let docling_path = std::env::temp_dir().join(format!("docuzent-real-adaptive-docling-{}.redb", std::process::id()));
        let _ = std::fs::remove_file(&cache_path);
        let text = "The quarterly report shows revenue growth across all regions, driven by strong enterprise software performance. ".repeat(20);

        // First ask: no profile yet, must cold-prime (nothing to predict from).
        let mut session1 = real_session(Mode::Adaptive, &cache_path, &docling_path);
        set_current(&mut session1, "real-adaptive-doc", &text, None);
        let r1 = session1.ask("What drove the growth?").unwrap();
        assert!(r1.adaptive_decision.is_none(), "no prior data - should not have made a real decision yet");
        assert_eq!(r1.timings[0].label, "prime");
        drop(session1);

        // Second ask, fresh session: a real profile now exists (from
        // session1's real calls) and a real disk-cached context exists -
        // adaptive mode should make a genuine, real prediction this time.
        let mut session2 = real_session(Mode::Adaptive, &cache_path, &docling_path);
        let disk_context: Option<Vec<i64>> = session2.disk_cache.get(&session2.cache_key("real-adaptive-doc")).unwrap().map(|b| serde_json::from_slice(&b).unwrap());
        assert!(disk_context.is_some(), "session1 should have persisted a context to disk");
        set_current(&mut session2, "real-adaptive-doc", &text, disk_context);
        let r2 = session2.ask("Which division was strongest?").unwrap();
        let decision = r2.adaptive_decision.expect("a real profile and a real disk context should produce a real decision");
        println!(
            "real adaptive decision: predicted swap {:.0}ms vs raw {:.0}ms, chose {}",
            decision.predicted_swap_ms,
            decision.predicted_raw_ms,
            if decision.chose_swap { "swap" } else { "raw" }
        );
        assert!(!r2.answer.is_empty());

        let _ = std::fs::remove_file(&cache_path);
        let _ = std::fs::remove_file(&docling_path);
    }
}
