//! Qwen3's ChatML prompt format, built by hand.
//!
//! docuzent builds every prompt itself instead of using a server-side chat
//! template, because the *exact text* is what makes KV reuse work: a saved
//! KV state covers a prompt prefix, and it is only reused if the next
//! prompt begins with precisely the same text. So prompts are laid out
//! **static part first** (the book excerpt, the persona, the core ideas -
//! all of which have saved KV states), and everything that changes per
//! question comes after.

const SYSTEM: &str = "<|im_start|>system\n";
const USER: &str = "<|im_start|>user\n";
const ASSISTANT: &str = "<|im_start|>assistant\n";
const END: &str = "<|im_end|>\n";
/// Qwen3 reasons in a `<think>` block before answering. An empty one asks
/// it to answer directly: replies are spoken to a person, and reasoning
/// tokens would only be latency.
const NO_THINKING: &str = "<think>\n\n</think>\n\n";

/// A system turn. As the first thing in a prompt, this is the part that
/// gets primed and saved.
pub fn system(text: &str) -> String {
    format!("{SYSTEM}{text}{END}")
}

/// A user turn.
pub fn user(text: &str) -> String {
    format!("{USER}{text}{END}")
}

/// The opening of the model's own turn, thinking switched off. A prompt
/// ends here and the model writes what follows.
pub fn assistant_open() -> String {
    format!("{ASSISTANT}{NO_THINKING}")
}

/// The opening of the model's own turn with thinking left on: it reasons in a `<think>` block first (counting,
/// comparing, ordering), then answers. Only for questions that need working out.
pub fn assistant_open_thinking() -> String {
    ASSISTANT.to_string()
}

/// A completed assistant turn (for conversation history).
pub fn assistant(text: &str) -> String {
    format!("{ASSISTANT}{text}{END}")
}

/// `prefix` (already a run of complete turns) followed by one user
/// question and the opening of the answer.
pub fn ask(prefix: &str, question: &str) -> String {
    format!("{prefix}{}{}", user(question), assistant_open())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turns_are_laid_out_the_way_qwen3_expects() {
        assert_eq!(system("be kind"), "<|im_start|>system\nbe kind<|im_end|>\n");
        assert_eq!(ask(&system("s"), "hi"), "<|im_start|>system\ns<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }

    /// Regression: KV reuse depends on a question's prompt *starting with*
    /// the primed prefix, byte for byte. Anything that changed a prefix
    /// (trimming, re-wrapping) would silently turn every "restore" into a
    /// full re-read.
    #[test]
    fn a_question_prompt_always_begins_with_its_primed_prefix() {
        let prefix = system("Part 3 of \"The Book\":\nSome text.");
        for q in ["a", "what is duty?", "multi\nline question"] {
            assert!(ask(&prefix, q).starts_with(&prefix));
        }
    }
}
