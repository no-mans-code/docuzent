//! Reading a document for a question - four ways, one interface ([`modes::read`]).
//!
//! * [`modes`] - the four reading modes: RAG, expanded RAG, RAG pointing at saved KV parts, saved KV parts only.
//! * [`reader`] - reading saved KV parts: score every part, read the relevant ones closely (Modes 3 and 4).
//! * [`index`] - the RAG index: chunks, expansions, hybrid word + vector search, stored per document.
//! * [`expand`] - what the model adds to an index: guided chunk boundaries, and Mode 2's expansions.
//! * [`answer`] - a plain answer from what was read.
//! * [`corpus`] - what is read: a document in parts, each with a saved KV state.
//! * [`document`] - a document made ready to read, for tools without a library of their own (the CLI, the evaluator).
//!
//! How the modes compare on real books - accuracy, speed, learning time, disk - is measured by `docuzent-eval` and
//! written up in docs/READING_MODES.md.

pub mod answer;
pub mod corpus;
pub mod document;
pub mod expand;
pub mod index;
pub mod modes;
pub mod reader;
pub mod text;

pub use answer::Think;
pub use corpus::{Corpus, MemCorpus};
pub use document::Document;
pub use index::{Chunking, Index, Search};
pub use modes::{read, search_each, Found, Mode, Shelf, Sources};
pub use reader::{Passage, ReadOptions, Reading};

/// The least a model's window can be to read a document: a part (up to 4,000 tokens), the question about it and the
/// answer must fit together. A smaller window would not fail - the server would quietly cut the part short.
pub const MIN_CONTEXT: usize = 8192;

/// Fails, saying why, if `model`'s window of `context` tokens cannot read a document.
pub fn check_context(model: &str, context: usize) -> anyhow::Result<()> {
    if context < MIN_CONTEXT {
        anyhow::bail!("`{model}` has a {context}-token window, but reading a document needs at least {MIN_CONTEXT}: a part (up to 4,000 tokens), the question about it and the answer must fit together");
    }
    Ok(())
}

#[cfg(test)]
mod context_tests {
    #[test]
    fn a_window_too_small_for_a_part_is_refused() {
        assert!(super::check_context("yi:6b", 4096).unwrap_err().to_string().contains("at least 8192"));
        assert!(super::check_context("qwen3:0.6b", 12288).is_ok());
    }
}
