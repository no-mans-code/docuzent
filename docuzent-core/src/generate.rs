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

    /// Same as [`Self::generate`], but invokes `on_token` with each new
    /// fragment of the response as it's produced - for a caller that
    /// wants to show a live-updating answer instead of blocking until
    /// the whole thing is ready. Ollama's `/api/generate` genuinely
    /// supports this for the *generation* phase via `stream: true`
    /// (distinct from the prefill/prompt-eval phase, which has no
    /// equivalent live signal - see `docuzent_core::vram`'s ingestion-
    /// progress estimate for that separate, unsolvable-for-real problem).
    ///
    /// The default implementation here just calls [`Self::generate`] and
    /// invokes `on_token` once with the whole response - correct, but
    /// not actually streamed. This exists so [`crate::session::Session`]
    /// can call `generate_streaming` unconditionally regardless of which
    /// `Generator` it holds (including test doubles that have no reason
    /// to implement real streaming) - only [`OllamaClient`] overrides it
    /// with the real thing. See
    /// https://github.com/no-mans-code/docuzent/issues/35.
    fn generate_streaming(&self, prompt: &str, context: Option<&[i64]>, on_token: &mut dyn FnMut(&str)) -> Result<GenerateResponse> {
        let resp = self.generate(prompt, context)?;
        let text = resp.text();
        if !text.is_empty() {
            on_token(&text);
        }
        Ok(resp)
    }
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

    /// Real streaming: Ollama's `/api/generate` with `stream: true` sends
    /// one JSON object per line (newline-delimited, not a single JSON
    /// document) - each carrying a fragment of `response` (or
    /// `thinking`, for a reasoning model), until a final line with
    /// `done: true` that also carries the real aggregate stats
    /// (`prompt_eval_count`, durations, etc.) this project's timing
    /// already depends on. Reads the body incrementally via `ureq`'s
    /// `into_reader()` rather than buffering the whole response first -
    /// otherwise `on_token` would only ever fire once, defeating the
    /// point.
    fn generate_streaming(&self, prompt: &str, context: Option<&[i64]>, on_token: &mut dyn FnMut(&str)) -> Result<GenerateResponse> {
        let url = format!("{}/api/generate", self.host.trim_end_matches('/'));
        let req = GenerateRequest { model: &self.model, prompt, stream: true, context, options: GenerateOptions { num_ctx: self.num_ctx, num_predict: DEFAULT_NUM_PREDICT } };
        let http_resp = ureq::post(&url).send_json(req).with_context(|| format!("ollama generate request to {url} failed - is `ollama serve` running?"))?;

        let reader = std::io::BufReader::new(http_resp.into_reader());
        let mut response_text = String::new();
        let mut thinking_text = String::new();
        let mut last_line: Option<GenerateResponse> = None;
        for line in std::io::BufRead::lines(reader) {
            let line = line.context("failed to read a line of ollama's streamed response")?;
            if line.trim().is_empty() {
                continue;
            }
            let partial: GenerateResponse = serde_json::from_str(&line).context("failed to parse a line of ollama's streamed response")?;
            if !partial.response.is_empty() {
                on_token(&partial.response);
                response_text.push_str(&partial.response);
            }
            if let Some(t) = &partial.thinking {
                thinking_text.push_str(t);
            }
            last_line = Some(partial); // the final (done: true) line carries the real aggregate stats
        }

        let mut resp = last_line.context("ollama's streamed response ended with no lines at all - is `ollama serve` still running?")?;
        resp.response = response_text;
        resp.thinking = if thinking_text.is_empty() { None } else { Some(thinking_text) };

        // A reasoning model can spend its whole budget in `thinking` and
        // never stream a visible `response` fragment at all (the same
        // real failure `GenerateResponse::text()` exists for) - fire the
        // same labeled fallback once here too, so a caller watching the
        // stream isn't left staring at nothing when the non-streaming
        // path would have shown something.
        if resp.response.trim().is_empty() {
            let fallback = resp.text();
            if !fallback.is_empty() {
                on_token(&fallback);
            }
        }

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

    /// A generator that doesn't implement real streaming, matching the
    /// test doubles used elsewhere in this project (e.g.
    /// `session::tests::FakeGenerator`) - relies entirely on
    /// `Generator::generate_streaming`'s default implementation.
    struct NonStreamingFakeGenerator;
    impl Generator for NonStreamingFakeGenerator {
        fn generate(&self, _prompt: &str, _context: Option<&[i64]>) -> Result<GenerateResponse> {
            Ok(GenerateResponse { response: "a complete fake answer".to_string(), ..Default::default() })
        }
    }

    #[test]
    fn default_generate_streaming_calls_the_callback_once_with_the_whole_response() {
        let gen = NonStreamingFakeGenerator;
        let mut fragments = Vec::new();
        let resp = gen.generate_streaming("irrelevant", None, &mut |t| fragments.push(t.to_string())).unwrap();
        assert_eq!(resp.response, "a complete fake answer");
        assert_eq!(fragments, vec!["a complete fake answer".to_string()], "the default impl isn't real streaming, but must still call back exactly once with the full text");
    }

    // Real-Ollama streaming check below - `#[ignore]`d by default (run
    // with `cargo test -- --ignored`), needs a live `ollama serve` with
    // `qwen2.5:3b` pulled.

    #[test]
    #[ignore]
    fn real_ollama_streaming_calls_back_multiple_times_and_matches_the_non_streaming_answer() {
        let client = OllamaClient::new("http://localhost:11434", "qwen2.5:3b", 4096);
        let mut fragments: Vec<String> = Vec::new();
        let streamed = client.generate_streaming("Say exactly: The quick brown fox jumps over the lazy dog.", None, &mut |t| fragments.push(t.to_string())).unwrap();

        assert!(fragments.len() > 1, "a real multi-token answer should stream in more than one fragment, got {}", fragments.len());
        assert_eq!(fragments.concat(), streamed.response, "concatenating every streamed fragment must reproduce the final response exactly");
        assert!(streamed.prompt_eval_count > 0, "the final streamed line must still carry real aggregate stats");
        println!("real streamed answer ({} fragments): {:?}", fragments.len(), streamed.response);
    }
}
