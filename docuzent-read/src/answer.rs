//! A plain answer from what was read - no persona, no voice. What the evaluation uses to compare the reading modes
//! (so the modes, not a character's manner, are what differs), and what a tool that just wants the answer uses.
//!
//! The rules are the ones that raised thebook's accuracy (docs/READING_MODES.md): answer exactly what was asked,
//! with the names, numbers and order the passages give; work out any counting before stating it; say in a sentence
//! when the passages do not say something, and stop - never guess or bring in outside knowledge.

use anyhow::Result;
use docuzent_llm::{chatml, Llm, Sampling};

use crate::reader::{passages_block, Passage};
use crate::text::needs_thinking;

/// Most characters of passages an answer is given (~6,500 tokens: room in a 12,288-token window for the question,
/// the reasoning and the answer).
pub const ANSWER_PASSAGES_CHARS: usize = 26_000;
const ANSWER_TOKENS: i32 = 600;
const THINK_TOKENS: i32 = 1800;

/// Off the leash: the document first, then reasoning and knowledge from beyond it - always said to be so.
pub const OFFLEASH_RULE: &str = "The person has taken you OFF THE LEASH for this answer: the rule to use only the document is lifted. Start from what the passages say and build on it: reason step by step, work numbers through, and use what you know from beyond the document where it is not enough. Say plainly which parts come from the document and which are your own reasoning or knowledge (\"the document says...\", \"beyond the document...\"). If the document is a story and what you know goes further into it than the passages do, say so before you say it.";

pub fn answer_prompt(question: &str, passages: &[Passage]) -> String {
    answer_prompt_leash(question, passages, false)
}

/// The answer prompt, on the leash (only the passages) or off it ([`OFFLEASH_RULE`]).
pub fn answer_prompt_leash(question: &str, passages: &[Passage], offleash: bool) -> String {
    if offleash {
        let evidence = if passages.is_empty() { "(No passage of the document speaks to this.)".to_string() } else { passages_block(passages, ANSWER_PASSAGES_CHARS) };
        return format!("Passages from the document:\n{evidence}\n\nQuestion: {question}\n\n{OFFLEASH_RULE} Answer in a few sentences, showing the working.");
    }
    let evidence = if passages.is_empty() { "(No passage of the document speaks to this.)".to_string() } else { passages_block(passages, ANSWER_PASSAGES_CHARS) };
    format!(
        "Passages from the document:\n{evidence}\n\nQuestion: {question}\n\nAnswer exactly what was asked, plainly and specifically: the names, numbers, order and events the passages give. Work out any counting, adding or comparing step by step before you state it. Use only these passages. If they do not say something, say in a sentence that the document does not say it, and stop: never guess, fill in, or bring in what you know from anywhere else. Answer in a few sentences."
    )
}

/// The answer, reasoned through first when the question counts or compares and the model can reason.
pub fn answer(llm: &dyn Llm, question: &str, passages: &[Passage]) -> Result<String> {
    answer_leash(llm, question, passages, false)
}

/// [`answer`], on the leash or off it (off the leash, every answer is reasoned through first).
pub fn answer_leash(llm: &dyn Llm, question: &str, passages: &[Passage], offleash: bool) -> Result<String> {
    let prompt = answer_prompt_leash(question, passages, offleash);
    if (needs_thinking(question) || offleash) && llm.can_think() {
        let c = llm.complete(&format!("{}{}", chatml::user(&prompt), chatml::assistant_open_thinking()), &Sampling::thinking(ANSWER_TOKENS + THINK_TOKENS), &mut |_| {})?;
        if let Some(i) = c.text.find("</think>") {
            return Ok(c.text[i + "</think>".len()..].trim().to_string());
        }
    }
    Ok(llm.complete(&chatml::ask("", &prompt), &Sampling::precise(ANSWER_TOKENS), &mut |_| {})?.text.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use docuzent_llm::SimEngine;

    #[test]
    fn the_answer_is_asked_for_plainly_from_the_passages_alone() {
        let p = answer_prompt("How old is Flamel?", &[Passage { part: 3, book: String::new(), text: "He is six hundred and sixty-five.".into() }]);
        assert!(p.contains("[part 3]") && p.contains("six hundred and sixty-five") && p.contains("does not say it") && p.contains("never guess"));
        assert!(answer_prompt("x", &[]).contains("No passage"));
    }

    #[test]
    fn a_counting_question_is_reasoned_through_and_only_the_answer_is_kept() {
        let e = SimEngine::new(12288, |p| if p.ends_with("<|im_start|>assistant\n") { "<think>472 + 10 = 482</think>\n\nGryffindor: 482.".into() } else { "straight".into() });
        assert_eq!(answer(&e, "How many points did Gryffindor have?", &[]).unwrap(), "Gryffindor: 482.");
        assert_eq!(answer(&e, "Who is Hagrid?", &[]).unwrap(), "straight");
    }
}
