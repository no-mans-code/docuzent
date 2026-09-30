//! The language-model engine, and nothing else.
//!
//! Two small traits describe everything a reader needs from a
//! model:
//!
//! * [`Llm`] - generate text from a prompt (streaming), count tokens.
//! * [`KvSlot`] - save the model's *real attention state* (its KV cache) to
//!   a file, restore it, or wipe it.
//!
//! The second is the point. A book is read once, its parts' KV states are
//! written to disk, and from then on a part is "read" by loading its saved
//! state in about a tenth of a second instead of processing its four
//! thousand tokens again (measured: 0.07-0.10 s versus 1.75-2.06 s on an
//! RTX 5080 with Qwen3-14B). Ollama cannot do this - its `context` is only
//! a list of token ids, and replaying it re-processes every token - so the
//! real engine is llama.cpp's server ([`LlamaServer`]), whose slot API
//! saves and restores the actual tensors.
//!
//! [`SimEngine`] implements both traits without a GPU (it models a prefix
//! cache and counts the tokens each call has to process), so everything
//! built on this crate is testable on its own.

pub mod chatml;
pub mod embed;
pub mod json;
pub mod ollama;
pub mod server;
pub mod sim;

pub use embed::{cosine_similarity, Embedder, OllamaEmbedder, OpenAiEmbedder, SimEmbedder};
pub use ollama::{OllamaModel, OllamaServer};
pub use server::LlamaServer;
pub use sim::SimEngine;

use anyhow::Result;

/// How a completion is sampled.
#[derive(Debug, Clone, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_p: f32,
    /// Most tokens to generate.
    pub n_predict: i32,
    pub stop: Vec<String>,
}

impl Sampling {
    /// Deterministic - for extraction, planning and classification, where
    /// the same input should give the same output.
    pub fn precise(n_predict: i32) -> Self {
        Self { temperature: 0.0, top_p: 1.0, n_predict, stop: Vec::new() }
    }

    /// Qwen3's recommended thinking settings - for a reply that is worked out before it is written.
    pub fn thinking(n_predict: i32) -> Self {
        Self { temperature: 0.6, top_p: 0.95, n_predict, stop: Vec::new() }
    }

    /// Natural - for the reply a person reads. Qwen3's recommended
    /// non-thinking settings.
    pub fn natural(n_predict: i32) -> Self {
        Self { temperature: 0.7, top_p: 0.8, n_predict, stop: Vec::new() }
    }
}

/// What a completion did, so callers (and tests) can see *where the time
/// went* - not only what came back.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Completion {
    pub text: String,
    /// Tokens the model had to process afresh for this prompt. Small when
    /// the prompt's beginning was already in the KV cache.
    pub prompt_tokens: usize,
    /// Tokens of the prompt that were already in the KV cache.
    pub cached_tokens: usize,
    pub prompt_ms: f64,
    pub gen_tokens: usize,
    pub gen_ms: f64,
    /// Stopped because it hit `n_predict`, not because it finished.
    pub truncated: bool,
}

/// Generating text.
pub trait Llm: Send + Sync {
    /// Completes `prompt`, calling `on_token` with each fragment as it is
    /// produced. Whatever prefix of `prompt` is already in the KV cache is
    /// not processed again.
    fn complete(&self, prompt: &str, sampling: &Sampling, on_token: &mut dyn FnMut(&str)) -> Result<Completion>;

    /// Exact token count of `text` under the loaded model's tokenizer.
    fn count_tokens(&self, text: &str) -> Result<usize>;

    /// Whether the model can reason before it answers (a `<think>` block). A question that needs counting is
    /// only sent to reason when it can; otherwise it would be asked twice.
    fn can_think(&self) -> bool {
        true
    }
}

/// Time and size of one slot operation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SlotIo {
    pub tokens: usize,
    pub bytes: u64,
    pub ms: f64,
}

/// The model's single working KV slot, saved to and restored from files
/// in the engine's slot directory.
pub trait KvSlot: Send + Sync {
    /// Writes the slot's current KV state to `file`.
    fn save(&self, file: &str) -> Result<SlotIo>;

    /// Loads `file` into the slot, replacing whatever was there. `None`
    /// when the file is missing or unusable (e.g. evicted from the LRU
    /// store) - the caller re-reads the text instead.
    fn restore(&self, file: &str) -> Result<Option<SlotIo>>;

    /// Empties the slot.
    fn erase(&self) -> Result<()>;

    /// How many tokens the slot can hold.
    fn context_size(&self) -> usize;

    /// Whether saved states really are saved. False for an engine that cannot (Ollama): nothing is written, and
    /// every part is read with the prompt that needs it.
    fn persists(&self) -> bool {
        true
    }
}

/// Something that can both generate and save/restore its KV state - what a
/// running model server is. Blanket-implemented, so both [`LlamaServer`] and
/// [`SimEngine`] qualify.
pub trait Engine: Llm + KvSlot {}
impl<T: Llm + KvSlot> Engine for T {}
