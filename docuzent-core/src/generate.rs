//! A minimal client for Ollama's `/api/generate`, including its resumable
//! `context` mechanism - the closest thing its public API exposes to a "KV
//! cache." To be precise about what this actually is: Ollama does not
//! expose a model's real attention key/value tensors over its API at all.
//! `context` is a token-id array; replaying it lets the server continue a
//! conversation without the client resending everything said so far.
//! Measured empirically against a local Ollama (qwen2.5:3b): a follow-up
//! call reusing `context` reports a `prompt_eval_duration` roughly 10x
//! lower than a cold call re-sending the same document text. Whether that
//! is the server reusing real KV state internally or something else, it is
//! a genuine, measurable win - which is what [`crate::session`] persists
//! to disk and benchmarks.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub trait Generator {
    fn generate(&self, prompt: &str, context: Option<&[i64]>) -> Result<GenerateResponse>;
}

/// Without an explicit cap, Ollama generates until a natural stop or the
/// context fills - fine for a short factual answer, but a broad/
/// open-ended question (e.g. "summarize every topic in this document")
/// can legitimately run for minutes per call with no cap at all, times
/// however many map-reduce chunks a big document needs (sequential, not
/// parallel). This bounds a single call's worst case without silently
/// truncating a real answer - generous enough for a genuine answer, not
/// so large that one verbose chunk can dominate a whole map-reduce pass.
const DEFAULT_NUM_PREDICT: i32 = 768;

pub struct OllamaClient {
    host: String,
    model: String,
    /// Passed as `options.num_ctx` on every request - without this, Ollama
    /// silently runs at its own 4096-token default regardless of what
    /// context length the caller inferred and sized chunking decisions
    /// around, truncating anything longer without any error (the real bug
    /// this field exists to fix).
    num_ctx: u32,
}

impl OllamaClient {
    pub fn new(host: impl Into<String>, model: impl Into<String>, num_ctx: u32) -> Self {
        Self { host: host.into(), model: model.into(), num_ctx }
    }
}

#[derive(Serialize)]
struct GenerateOptions {
    num_ctx: u32,
    num_predict: i32,
}

#[derive(Serialize)]
struct GenerateRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<&'a [i64]>,
    options: GenerateOptions,
}

#[derive(Deserialize, Debug, Clone, Default)]
pub struct GenerateResponse {
    #[serde(default)]
    pub response: String,
    #[serde(default)]
    pub context: Vec<i64>,
    #[serde(default)]
    pub prompt_eval_count: u64,
    /// Nanoseconds, as Ollama reports it.
    #[serde(default)]
    pub prompt_eval_duration: u64,
    #[serde(default)]
    pub eval_count: u64,
    /// Nanoseconds, as Ollama reports it.
    #[serde(default)]
    pub eval_duration: u64,
    #[serde(default)]
    pub total_duration: u64,
}

impl Generator for OllamaClient {
    fn generate(&self, prompt: &str, context: Option<&[i64]>) -> Result<GenerateResponse> {
        let url = format!("{}/api/generate", self.host.trim_end_matches('/'));
        let req = GenerateRequest { model: &self.model, prompt, stream: false, context, options: GenerateOptions { num_ctx: self.num_ctx, num_predict: DEFAULT_NUM_PREDICT } };
        let resp: GenerateResponse = ureq::post(&url)
            .send_json(req)
            .with_context(|| format!("ollama generate request to {url} failed - is `ollama serve` running?"))?
            .into_json()
            .context("failed to parse ollama generate response")?;
        Ok(resp)
    }
}
