//! How much of a model's window each step of reading may use - derived from the window, so a model with a small one
//! (1,024 tokens) and a model with a large one (40,960) both read a whole document: the small one in more, smaller
//! pieces, the large one with more of the document in view at once.
//!
//! At the window the engine was built and measured with ([`REFERENCE_CONTEXT`], 12,288 tokens) every number is what
//! it always was. Quantities that must *fit* (a part, an answer's room) shrink with a smaller window and do not grow
//! past their reference size; quantities that are *how much is handed over* (chunks searched for, characters of
//! passages one answer reads) grow with a larger window too.

/// The window the reading was designed and first measured at.
pub const REFERENCE_CONTEXT: usize = 12_288;
/// The least window a document can be read with: a part of a few hundred tokens, its question, and a short answer.
pub const FLOOR_CONTEXT: usize = 1_024;
/// The most of a window docuzent will ever plan for, whatever the model offers: memory for the KV cache grows with
/// the window, and past this the gain is small. It can be lowered ([`max_context`]), never raised.
pub const HARD_MAX_CONTEXT: usize = 32_768;

/// The window docuzent plans for: [`HARD_MAX_CONTEXT`], or less if `DOCUZENT_MAX_CONTEXT` says so (a number of
/// tokens; anything above the hard limit, or unreadable, is ignored).
pub fn max_context() -> usize {
    std::env::var("DOCUZENT_MAX_CONTEXT").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(HARD_MAX_CONTEXT, |n| n.clamp(FLOOR_CONTEXT, HARD_MAX_CONTEXT))
}

/// The sizes of every step, for one window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Budget {
    pub context: usize,
    /// Most tokens in a part (the unit whose KV state is saved, and that is scored and read whole).
    pub part_tokens: usize,
    /// Most characters in an index chunk.
    pub chunk_chars: usize,
    /// Characters of each neighbouring chunk handed over with a found chunk.
    pub neighbour_chars: usize,
    /// Chunks a RAG search hands over (Modes 1 and 2).
    pub rag_chunks: usize,
    /// Chunks Mode 3 searches for, and the most parts it then reads whole.
    pub rag_kv_chunks: usize,
    pub rag_kv_parts: usize,
    /// Characters of passages one answering call reads; more are swept (answer::answer_within).
    pub answer_chars: usize,
    pub answer_tokens: i32,
    /// Tokens an answer may reason for first (0: no room to reason).
    pub think_tokens: i32,
    /// Tokens a close reading of a part may write.
    pub leaf_tokens: i32,
    /// Passages described in one expansion call, and the tokens it may write.
    pub expand_batch: usize,
    pub expand_tokens: i32,
}

impl Budget {
    /// The budget for an engine whose window is `engine_context` tokens - capped at [`max_context`].
    pub fn for_engine(engine_context: usize) -> Self {
        Self::for_context(engine_context.min(max_context()))
    }

    pub fn for_context(context: usize) -> Self {
        let s = context as f64 / REFERENCE_CONTEXT as f64;
        let fit = s.min(1.0);
        let at_least = |x: f64, min: f64| x.round().max(min);
        let answer_tokens = at_least(600.0 * fit, 150.0) as i32;
        let think = 1800.0 * fit;
        // reasoning in a few dozen tokens is no reasoning: below ~300, answer straight
        let think_tokens = if think < 300.0 { 0 } else { think.round() as i32 };
        let part_tokens = at_least(4000.0 * fit, 250.0) as usize;
        // what one answering call can hold beside the instructions and question (~450 tokens), the reasoning and
        // the answer, at ~3.2 characters a token
        let room = context.saturating_sub(answer_tokens as usize + think_tokens as usize + 450);
        let answer_chars = ((26_000.0 * s) as usize).min((room as f64 * 3.2) as usize).max(400);
        // a chunk fits one answering call, and can be described beside its part (instructions ~300 tokens, ~250 to
        // write the description)
        let chunk_chars = 1500.min(answer_chars).min(context.saturating_sub(part_tokens + 550) * 3).max(300);
        let chunk_tokens = chunk_chars / 3;
        let leaf_tokens = 700.min(context.saturating_sub(part_tokens + 300)).max(100) as i32;
        let expand_batch = if fit >= 1.0 { 3 } else { 1 };
        let expand_tokens = 1600.min(context.saturating_sub(part_tokens + 300 + chunk_tokens * expand_batch)).max(120) as i32;
        Self {
            context,
            part_tokens,
            chunk_chars,
            neighbour_chars: (600.0 * fit).round() as usize,
            rag_chunks: at_least(8.0 * s, 2.0) as usize,
            rag_kv_chunks: at_least(12.0 * s, 2.0) as usize,
            rag_kv_parts: at_least(4.0 * s, 1.0) as usize,
            answer_chars,
            answer_tokens,
            think_tokens,
            leaf_tokens,
            expand_batch,
            expand_tokens,
        }
    }
}

impl Default for Budget {
    fn default() -> Self {
        Self::for_context(REFERENCE_CONTEXT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_the_reference_window_every_size_is_what_it_always_was() {
        let b = Budget::default();
        assert_eq!(
            (b.part_tokens, b.chunk_chars, b.neighbour_chars, b.rag_chunks, b.rag_kv_chunks, b.rag_kv_parts, b.answer_chars, b.answer_tokens, b.think_tokens, b.leaf_tokens, b.expand_batch, b.expand_tokens),
            (4000, 1500, 600, 8, 12, 4, 26_000, 600, 1800, 700, 3, 1600)
        );
    }

    #[test]
    fn a_small_window_reads_in_small_pieces_that_fit_and_a_large_one_hands_over_more() {
        let small = Budget::for_context(1024);
        // a part and its close reading fit the window; an answer's passages, instructions and answer fit too
        assert!(small.part_tokens + 300 + small.leaf_tokens as usize <= 1024, "{small:?}");
        assert!(small.answer_chars / 3 + 450 + small.answer_tokens as usize + small.think_tokens as usize <= 1100, "{small:?}");
        assert!(small.chunk_chars <= small.answer_chars && small.expand_batch == 1 && small.think_tokens == 0);
        // describing a chunk beside its part fits too
        assert!(small.part_tokens + 300 + small.chunk_chars / 3 + small.expand_tokens as usize <= 1024 && small.expand_tokens >= 200, "{small:?}");
        let large = Budget::for_context(40_960);
        assert_eq!((large.part_tokens, large.chunk_chars), (4000, 1500), "a part does not grow past its saved-state size");
        assert!(large.rag_chunks > 20 && large.answer_chars > 80_000 && large.rag_kv_parts > 10, "{large:?}");
    }
}
