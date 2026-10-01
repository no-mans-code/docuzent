//! A plain answer from what was read - no persona, no voice. What the evaluation uses to compare the reading modes
//! (so the modes, not a character's manner, are what differs), and what a tool that just wants the answer uses.
//!
//! The rules are the ones that raised thebook's accuracy (docs/READING_MODES.md): answer exactly what was asked,
//! with the names, numbers and order the passages give; work out any counting before stating it; say in a sentence
//! when the passages do not say something, and stop - never guess or bring in outside knowledge.

use anyhow::{bail, Result};
use docuzent_llm::{chatml, Llm, Sampling};
use serde::{Deserialize, Serialize};

use crate::budget::Budget;
use crate::reader::{passages_block, Passage};
use crate::text::needs_thinking;

/// Most characters of passages one answer reads at the reference window ([`Budget::answer_chars`]); more are swept
/// by [`fit_passages`], never dropped.
pub const ANSWER_PASSAGES_CHARS: usize = 26_000;

/// Off the leash: the document first, then reasoning and knowledge from beyond it - always said to be so.
pub const OFFLEASH_RULE: &str = "The person has taken you OFF THE LEASH for this answer: the rule to use only the document is lifted. Start from what the passages say and build on it: reason step by step, work numbers through, and use what you know from beyond the document where it is not enough. Say plainly which parts come from the document and which are your own reasoning or knowledge (\"the document says...\", \"beyond the document...\"). If the document is a story and what you know goes further into it than the passages do, say so before you say it.";

/// When an answer is reasoned through before it is written. Reasoning only happens on a model that can
/// ([`Llm::can_think`]: Ollama's `thinking` capability, or a `<think>` chat template); on any other, every setting
/// answers straight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Think {
    /// Counting, comparing and ordering questions ([`needs_thinking`]), and every answer off the leash.
    #[default]
    Auto,
    /// Every answer: slower, for hard questions the detector does not see (a numerical worded as a story).
    Always,
    /// None, not even off the leash: the fastest, for a model whose reasoning wanders.
    Never,
}

impl Think {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.trim().to_lowercase().as_str() {
            "auto" | "" => Think::Auto,
            "always" | "on" | "think" => Think::Always,
            "never" | "off" | "nothink" => Think::Never,
            other => bail!("unknown thinking setting `{other}` - auto, always or never"),
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Think::Auto => "auto",
            Think::Always => "always",
            Think::Never => "never",
        }
    }

    /// Whether an answer to `question` (off the leash or not) is reasoned through, on a model that `can_think`.
    pub fn wants(self, question: &str, offleash: bool, can_think: bool) -> bool {
        can_think
            && match self {
                Think::Auto => needs_thinking(question) || offleash,
                Think::Always => true,
                Think::Never => false,
            }
    }
}

pub fn answer_prompt(question: &str, passages: &[Passage]) -> String {
    answer_prompt_leash(question, passages, false)
}

/// The answer prompt, on the leash (only the passages) or off it ([`OFFLEASH_RULE`]).
pub fn answer_prompt_leash(question: &str, passages: &[Passage], offleash: bool) -> String {
    if offleash {
        let evidence = if passages.is_empty() { "(No passage of the document speaks to this.)".to_string() } else { passages_block(passages, usize::MAX) };
        return format!("Passages from the document:\n{evidence}\n\nQuestion: {question}\n\n{OFFLEASH_RULE} Answer in a few sentences, showing the working.");
    }
    let evidence = if passages.is_empty() { "(No passage of the document speaks to this.)".to_string() } else { passages_block(passages, usize::MAX) };
    format!(
        "Passages from the document:\n{evidence}\n\nQuestion: {question}\n\nAnswer exactly what was asked, plainly and specifically: the names, numbers, order and events the passages give. Work out any counting, adding or comparing step by step before you state it. Use only these passages - and what the question itself gives: figures or facts it states may be worked through with what the passages say (a rule, a law, a formula, a method), showing the working. If they do not say something, say in a sentence that the document does not say it, and stop: never guess, fill in, or bring in what you know from anywhere else. Answer in a few sentences."
    )
}

/// The answer, reasoned through first when the question counts or compares and the model can reason.
pub fn answer(llm: &dyn Llm, question: &str, passages: &[Passage]) -> Result<String> {
    answer_leash(llm, question, passages, false)
}

/// [`answer`], on the leash or off it (off the leash, every answer is reasoned through first).
pub fn answer_leash(llm: &dyn Llm, question: &str, passages: &[Passage], offleash: bool) -> Result<String> {
    answer_with(llm, question, passages, offleash, Think::Auto)
}

/// [`answer_leash`], reasoning through first as `think` says - sized for the reference window
/// ([`Budget::default`]). Use [`answer_within`] with the model's own budget.
pub fn answer_with(llm: &dyn Llm, question: &str, passages: &[Passage], offleash: bool, think: Think) -> Result<String> {
    answer_within(llm, question, passages, offleash, think, &Budget::default())
}

/// The answer, within `budget`: every passage is read - when they are more than one call holds, they are swept
/// first ([`fit_passages`]), never cut short.
pub fn answer_within(llm: &dyn Llm, question: &str, passages: &[Passage], offleash: bool, think: Think, budget: &Budget) -> Result<String> {
    let passages = fit_passages(llm, question, passages, budget)?;
    let prompt = answer_prompt_leash(question, &passages, offleash);
    if budget.think_tokens > 0 && think.wants(question, offleash, llm.can_think()) {
        let c = llm.complete(&format!("{}{}", chatml::user(&prompt), chatml::assistant_open_thinking()), &Sampling::thinking(budget.answer_tokens + budget.think_tokens), &mut |_| {})?;
        if let Some(i) = c.text.find("</think>") {
            return Ok(c.text[i + "</think>".len()..].trim().to_string());
        }
    }
    Ok(llm.complete(&chatml::ask("", &prompt), &Sampling::precise(budget.answer_tokens), &mut |_| {})?.text.trim().to_string())
}

/// Characters `passages` take in a prompt.
fn block_chars(passages: &[Passage]) -> usize {
    passages_block(passages, usize::MAX).chars().count()
}

/// `passages`, made to fit one answering call of `budget` - **without any of them going unread**. Passages that fit
/// are returned as they are. Otherwise they are swept: read in groups that each fit a call, every sentence that
/// helps answer `question` copied out word for word under its part label, and those copies are the passages that go
/// on - swept again if they still do not fit. (Cutting the list short instead would hide the passages past the cut:
/// the answer would then say "the document does not say" about what the document says.)
pub fn fit_passages(llm: &dyn Llm, question: &str, passages: &[Passage], budget: &Budget) -> Result<Vec<Passage>> {
    let mut current: Vec<Passage> = passages.to_vec();
    while block_chars(&current) > budget.answer_chars {
        current = sweep(llm, question, &current, budget)?;
    }
    Ok(current)
}

pub fn sweep_prompt(question: &str, block: &str) -> String {
    format!("Passages from the document:\n{block}\n\nQuestion: {question}\n\nCopy out, word for word, every sentence of these passages that helps answer the question - names, numbers, steps, formulas, what happened and why - each under its [part N] label. Copy nothing else and add nothing of your own. If no sentence helps, reply exactly: NONE")
}

/// The sentences of `text`: split after `.`, `!`, `?` or `…` followed by a space, and at line breaks.
fn sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.split('\n') {
        let chars: Vec<(usize, char)> = line.char_indices().collect();
        let mut start = 0;
        for (n, &(i, c)) in chars.iter().enumerate() {
            let ends = matches!(c, '.' | '!' | '?' | '…') && chars.get(n + 1).is_none_or(|&(_, next)| next.is_whitespace());
            if ends || n + 1 == chars.len() {
                let end = i + c.len_utf8();
                let s = line[start..end].trim();
                if !s.is_empty() {
                    out.push(s.to_string());
                }
                start = end;
            }
        }
    }
    out
}

/// A group's passages as numbered sentences ("[n] (part P) ..."), and for each number the passage it came from.
fn number_sentences(group: &[Passage]) -> (String, Vec<(usize, String)>) {
    let mut listing = String::new();
    let mut index = Vec::new();
    for (p, passage) in group.iter().enumerate() {
        for s in sentences(&passage.text) {
            index.push((p, s.clone()));
            listing.push_str(&format!("[{}] (part {}) {s}\n", index.len(), passage.part));
        }
    }
    (listing, index)
}

pub fn pick_prompt(question: &str, numbered: &str) -> String {
    format!("Numbered sentences from the document:\n{numbered}\nQuestion: {question}\n\nWhich of these sentences help answer the question - names, numbers, steps, formulas, what happened and why? Reply with their numbers only, separated by commas (for example: 3, 7, 12), and nothing else. If none helps, reply exactly: NONE")
}

/// Room to write the numbers of every sentence, and no more.
fn pick_tokens(sentences: usize) -> i32 {
    (sentences * 3 + 16).min(600) as i32
}

/// The sentence numbers a reply names (each between 1 and `count`), in order, once each.
fn picked_numbers(reply: &str, count: usize) -> Vec<usize> {
    let mut out: Vec<usize> = reply.split(|c: char| !c.is_ascii_digit()).filter_map(|t| t.parse().ok()).filter(|n| (1..=count).contains(n)).collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// The picked sentences, copied exactly, gathered back under the passages they came from (in their order).
fn picked_passages(group: &[Passage], index: &[(usize, String)], picked: &[usize]) -> Vec<Passage> {
    let mut by_passage: Vec<Vec<&str>> = vec![Vec::new(); group.len()];
    for &n in picked {
        let (p, s) = &index[n - 1];
        by_passage[*p].push(s);
    }
    by_passage.into_iter().enumerate().filter(|(_, s)| !s.is_empty()).map(|(p, s)| Passage { text: s.join(" "), ..group[p].clone() }).collect()
}

/// One round of [`fit_passages`]: every passage read, in groups that fit a call (a passage longer than a call is
/// split, not cut); what each group keeps is capped so that the round as a whole must shrink.
fn sweep(llm: &dyn Llm, question: &str, passages: &[Passage], budget: &Budget) -> Result<Vec<Passage>> {
    let room = budget.answer_chars.saturating_sub(200).max(200);
    let mut pieces: Vec<Passage> = Vec::new();
    for p in passages {
        if p.text.chars().count() <= room {
            pieces.push(p.clone());
        } else {
            pieces.extend(docuzent_doc::chunk::split_to_max(&p.text, room).into_iter().map(|t| Passage { text: t, ..p.clone() }));
        }
    }
    let mut groups: Vec<Vec<Passage>> = vec![Vec::new()];
    for p in pieces {
        let last = groups.last_mut().unwrap();
        if !last.is_empty() && block_chars(last) + p.text.chars().count() + 20 > room {
            groups.push(vec![p]);
        } else {
            last.push(p);
        }
    }
    // together, what the groups keep is at most about one call's worth of passages (~3 characters a token)
    let keep_tokens = ((budget.answer_chars / 3) / groups.len()).clamp(60, budget.answer_tokens.max(60) as usize * 2) as i32;
    let mut kept = Vec::new();
    for g in &groups {
        // The model points at the sentences that help, by number, and they are copied here exactly - a few tokens
        // to write instead of the sentences themselves (on Qwen3-14B ~2 s a group instead of ~30), and no quotation
        // can come out reworded. A reply with no numbers in it falls back to the model copying them out.
        let numbered = number_sentences(g);
        let c = llm.complete(&chatml::ask("", &pick_prompt(question, &numbered.0)), &Sampling::precise(pick_tokens(numbered.1.len())), &mut |_| {})?;
        let reply = c.text.trim();
        if crate::reader::is_none(reply) {
            continue;
        }
        let picked = picked_numbers(reply, numbered.1.len());
        if !picked.is_empty() {
            kept.extend(picked_passages(g, &numbered.1, &picked));
            continue;
        }
        let c = llm.complete(&chatml::ask("", &sweep_prompt(question, &passages_block(g, usize::MAX))), &Sampling::precise(keep_tokens), &mut |_| {})?;
        let text = c.text.trim();
        if !crate::reader::is_none(text) && !text.is_empty() {
            kept.push(Passage { part: g[0].part, book: g[0].book.clone(), text: text.to_string() });
        }
    }
    // the round must shrink: what is kept was capped, so it can only be longer than its input if the model padded
    // it (or picked every sentence) - then keep each group's copy to its share
    if block_chars(&kept) >= block_chars(passages) {
        let share = (budget.answer_chars / kept.len().max(1)).max(100);
        for k in &mut kept {
            k.text = k.text.chars().take(share).collect();
        }
    }
    Ok(kept)
}

#[cfg(test)]
mod tests {
    use super::*;
    use docuzent_llm::SimEngine;

    #[test]
    fn the_answer_is_asked_for_plainly_from_the_passages_alone() {
        let p = answer_prompt("How old is Flamel?", &[Passage { part: 3, book: String::new(), text: "He is six hundred and sixty-five.".into() }]);
        assert!(p.contains("[part 3]") && p.contains("six hundred and sixty-five") && p.contains("does not say it") && p.contains("never guess"));
        assert!(p.contains("what the question itself gives"), "a numerical's own figures may be worked through with the passages' formula");
        assert!(answer_prompt("x", &[]).contains("No passage"));
    }

    #[test]
    fn a_counting_question_is_reasoned_through_and_only_the_answer_is_kept() {
        let e = SimEngine::new(12288, |p| if p.ends_with("<|im_start|>assistant\n") { "<think>472 + 10 = 482</think>\n\nGryffindor: 482.".into() } else { "straight".into() });
        assert_eq!(answer(&e, "How many points did Gryffindor have?", &[]).unwrap(), "Gryffindor: 482.");
        assert_eq!(answer(&e, "Who is Hagrid?", &[]).unwrap(), "straight");
    }

    #[test]
    fn thinking_can_be_asked_for_always_or_never_and_only_happens_on_a_model_that_can() {
        let e = SimEngine::new(12288, |p| if p.ends_with("<|im_start|>assistant\n") { "<think>v = gt = 40</think>\n\n40 m/s.".into() } else { "straight".into() });
        let q = "A stone falls for 4 s. What speed does it reach?";
        assert_eq!(answer_with(&e, q, &[], false, Think::Auto).unwrap(), "straight", "auto: not a counting question");
        assert_eq!(answer_with(&e, q, &[], false, Think::Always).unwrap(), "40 m/s.");
        assert_eq!(answer_with(&e, "How many points?", &[], true, Think::Never).unwrap(), "straight", "never, even off the leash");
        assert!(Think::Auto.wants("Who is Hagrid?", true, true) && !Think::Always.wants("x", false, false), "off the leash reasons on auto; a model that cannot, never does");
        assert_eq!((Think::parse("always").unwrap(), Think::parse("off").unwrap(), Think::parse("").unwrap()), (Think::Always, Think::Never, Think::Auto));
        assert!(Think::parse("maybe").is_err());
        assert_eq!(serde_json::to_string(&Think::Always).unwrap(), "\"always\"");
    }

    /// More passages than a small window holds: every one is read (in groups that fit), the sentences that bear
    /// on the question kept, and the answer reads those - the passage that answers, at the end of the list, is
    /// not cut off.
    #[test]
    fn passages_too_many_for_the_window_are_swept_and_none_goes_unread() {
        use std::sync::{Arc, Mutex};
        let seen: Arc<Mutex<Vec<String>>> = Arc::default();
        let log = seen.clone();
        let e = SimEngine::new(1024, move |p| {
            log.lock().unwrap().push(p.to_string());
            if p.contains("Numbered sentences from the document") {
                // point at the sentence about the dragon, if this group has it
                p.lines().find(|l| l.contains("dragon egg") && l.starts_with('[')).and_then(|l| l[1..].split(']').next()).map(str::to_string).unwrap_or_else(|| "NONE".into())
            } else if p.contains("Hagrid won the dragon egg") {
                "In a card game.".into()
            } else {
                "The document does not say.".into()
            }
        });
        let b = Budget::for_context(1024);
        let mut passages: Vec<Passage> = (1..=8).map(|i| Passage { part: i, book: String::new(), text: format!("Part {i} is about something else entirely. {}", "Filler words go here. ".repeat(40)) }).collect();
        passages.push(Passage { part: 9, book: String::new(), text: "Hagrid won the dragon egg in a card game at the pub.".into() });
        assert!(block_chars(&passages) > b.answer_chars, "more than one call holds");
        let fitted = fit_passages(&e, "Where did Hagrid get the dragon egg?", &passages, &b).unwrap();
        assert!(block_chars(&fitted) <= b.answer_chars);
        let prompts = seen.lock().unwrap().clone();
        for p in &passages {
            assert!(prompts.iter().any(|x| x.contains(&p.text[..30])), "part {} was read", p.part);
        }
        assert_eq!(answer_within(&e, "Where did Hagrid get the dragon egg?", &passages, false, Think::Auto, &b).unwrap(), "In a card game.");
        assert!(fitted.iter().any(|f| f.part == 9 && f.text == "Hagrid won the dragon egg in a card game at the pub."), "the sentence is the book's own, copied exactly: {fitted:?}");
        assert!(!prompts.iter().any(|x| x.contains("Copy out, word for word")), "the model only pointed; it copied nothing");
    }

    #[test]
    fn sentences_are_numbered_picked_by_number_and_copied_exactly() {
        assert_eq!(sentences("One. Two? Three!
Four… five"), vec!["One.", "Two?", "Three!", "Four…", "five"]);
        assert_eq!(sentences("Mr. Smith arrived."), vec!["Mr.", "Smith arrived."], "an abbreviation splits a sentence - picked by number, both halves are still the book's words");
        assert_eq!(picked_numbers("Sentences 3, 7 and 12. Also 7, and 99.", 12), vec![3, 7, 12], "in order, once, only real numbers");
        let g = vec![Passage { part: 2, book: String::new(), text: "A. B. C.".into() }, Passage { part: 5, book: String::new(), text: "D. E.".into() }];
        let (listing, index) = number_sentences(&g);
        assert!(listing.contains("[2] (part 2) B.") && listing.contains("[5] (part 5) E."));
        let kept = picked_passages(&g, &index, &[1, 3, 5]);
        assert_eq!(kept.iter().map(|p| (p.part, p.text.as_str())).collect::<Vec<_>>(), vec![(2, "A. C."), (5, "E.")]);
    }

    /// A model that does not answer with numbers: the group is copied out the old way instead - never dropped.
    #[test]
    fn a_reply_without_numbers_falls_back_to_copying() {
        let e = SimEngine::new(1024, |p| if p.contains("Numbered sentences") { "The second one, I think.".into() } else if p.contains("Copy out, word for word") { "[part 9] The egg came from a card game.".into() } else { "ok".into() });
        let b = Budget::for_context(1024);
        let passages: Vec<Passage> = (1..=9).map(|i| Passage { part: i, book: String::new(), text: format!("Part {i} sentence. {}", "More words here. ".repeat(40)) }).collect();
        let fitted = fit_passages(&e, "Where did the egg come from?", &passages, &b).unwrap();
        assert!(fitted.iter().any(|f| f.text.contains("card game")), "{fitted:?}");
    }
}
