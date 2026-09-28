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
///
/// Sized with headroom for *reasoning* models (Ollama's `thinking`
/// capability, e.g. the qwen3 family): confirmed via a real failure that
/// a smaller cap can be exhausted entirely inside the model's internal
/// `<think>` trace, leaving `response` empty before the model ever
/// reaches a visible answer - see [`GenerateResponse::text`] for the
/// other half of this fix (falling back to `thinking` when that happens
/// anyway, since no fixed cap can rule it out for every model/question).
const DEFAULT_NUM_PREDICT: i32 = 1536;

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
    /// Present only for reasoning-capable models (Ollama's `thinking`
    /// field) - the model's internal reasoning trace, kept separate from
    /// `response` (the visible answer). Real, confirmed failure mode:
    /// under a real compound question, the model can exhaust its whole
    /// `num_predict` budget inside this trace and never reach a visible
    /// answer at all, leaving `response` empty while this field holds a
    /// real (if truncated) trace - see [`Self::text`].
    #[serde(default)]
    pub thinking: Option<String>,
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

impl GenerateResponse {
    /// The real answer text to show a caller - `response` when it's
    /// non-empty (the normal case for every model this project has
    /// tested that doesn't expose a separate reasoning trace), falling
    /// back to `thinking` when `response` came back empty but the model
    /// did produce something (a reasoning model that ran out of budget
    /// before emitting a visible answer - see the doc comment on
    /// `thinking`). Never silently returns a blank answer when the model
    /// genuinely said something, even if it's an incomplete trace rather
    /// than a clean final answer.
    pub fn text(&self) -> String {
        let trimmed = self.response.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
        match self.thinking.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
            // Labeled, not silently substituted - a raw reasoning trace
            // isn't a clean answer, and the caller should be able to tell
            // this happened rather than mistake it for a normal response.
            Some(thinking) => format!("[the model ran out of budget before completing an answer - showing its unfinished reasoning instead]\n\n{thinking}"),
            None => trimmed.to_string(),
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_prefers_response_when_non_empty() {
        let resp = GenerateResponse { response: "the real answer".to_string(), thinking: Some("some reasoning".to_string()), ..Default::default() };
        assert_eq!(resp.text(), "the real answer");
    }

    #[test]
    fn text_falls_back_to_thinking_when_response_is_empty() {
        // The real failure mode this exists for: a reasoning model
        // exhausts its num_predict budget entirely inside <think>,
        // leaving response empty - confirmed via a real qwen3:0.6b call.
        let resp = GenerateResponse { response: "".to_string(), thinking: Some("the model's real reasoning trace".to_string()), ..Default::default() };
        let text = resp.text();
        assert!(text.contains("the model's real reasoning trace"));
        assert!(text.contains("ran out of budget"), "the fallback must be labeled, not silently substituted");
    }

    #[test]
    fn text_falls_back_to_thinking_when_response_is_only_whitespace() {
        let resp = GenerateResponse { response: "   \n  ".to_string(), thinking: Some("real reasoning".to_string()), ..Default::default() };
        assert!(resp.text().contains("real reasoning"));
    }

    #[test]
    fn text_is_empty_when_both_response_and_thinking_are_empty() {
        let resp = GenerateResponse { response: "".to_string(), thinking: None, ..Default::default() };
        assert_eq!(resp.text(), "");
    }

    #[test]
    fn text_ignores_an_empty_thinking_field_and_stays_empty() {
        let resp = GenerateResponse { response: "".to_string(), thinking: Some("   ".to_string()), ..Default::default() };
        assert_eq!(resp.text(), "");
    }
}
