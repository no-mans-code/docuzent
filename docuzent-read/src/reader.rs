//! Reading a document for a question over its saved KV parts - Mode 4, and the close reading Mode 3 uses.
//!
//! Each part's KV state is swapped in (a restore from disk, about a tenth of a second - not a re-read of four
//! thousand tokens), and:
//!
//! 1. **Score** ([`scan`] only) - every part answers with one digit: how much of it bears on the question. One token
//!    of output, so a part costs a restore and a fraction of a second. No index decides in advance which parts may
//!    matter, so nothing is missed for want of a matching word. A word-match backstop ([`lexical_parts`]) adds the
//!    parts that share the question's rarer words, for questions worded oddly enough to fool the digit.
//! 2. **Read closely** ([`read_parts`]) - each relevant part is asked to answer from itself and quote its proof,
//!    then pointed at the first words of its own second half and asked again: one pass tends to report what comes
//!    first and stop (measured: a line 63% into a part was missed until this pass was added).
//!
//! What comes back are the part's own words, in reading order.

use anyhow::Result;
use docuzent_doc::kvpool::{KvPool, Swap};
use docuzent_llm::{chatml, Sampling};
use serde::{Deserialize, Serialize};

use crate::corpus::Corpus;
use crate::text::{clean_rephrasing, stem, STOP};

/// Tokens an extraction may run to per part: a safety bound, not a cap on what is gathered (a part that runs past
/// it is counted in [`Reading::truncated`]).
pub const LEAF_TOKENS: i32 = 700;
/// A score is one digit; a couple of tokens leaves room for a stray space.
const SCORE_TOKENS: i32 = 3;
/// Parts scoring at least this (of 9) are read closely - all of them, however many.
pub const MIN_SCORE: u8 = 3;
/// A part scoring at least this whose extraction came back empty is asked again, for its closest lines.
pub const RETRY_SCORE: u8 = 7;
/// How many parts the word match alone may bring in.
const LEXICAL_PARTS: usize = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Passage {
    /// 1-based part number.
    pub part: usize,
    /// In a merged corpus: which document it is from. Empty otherwise.
    #[serde(default)]
    pub book: String,
    pub text: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reading {
    pub passages: Vec<Passage>,
    /// Parts swapped in from their saved state (or already resident).
    pub recalled: usize,
    /// Parts that had to be read again because their saved state was gone (or the engine keeps none).
    pub read_afresh: usize,
    /// Prompt tokens the model processed - small when the KV swaps did their job.
    pub processed_tokens: usize,
    /// How much of each part (0-9) bears on the question, in part order (empty when nothing was scored).
    pub scores: Vec<u8>,
    /// Parts read closely.
    pub read_closely: usize,
    /// Extractions that ran into [`LEAF_TOKENS`] and were cut off.
    pub truncated: usize,
    /// The question in the document's own words, when nothing scored for the original.
    pub reworded: Option<String>,
    /// Chunks the index returned (the RAG modes).
    pub retrieved: usize,
    pub ms: f64,
}

/// Options for one reading.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReadOptions<'a> {
    /// The person's whole message, when the question was drawn from a longer one: background for the extractions.
    pub context: Option<&'a str>,
    /// When set, only these parts (0-based) are looked at.
    pub only: Option<&'a [usize]>,
}

/// The person's whole message, as background for one question drawn from it. Not used for scoring: on the real
/// model it inflates a part's score (one went from 3 to 7 with the message attached).
pub fn context_note(query: &str, context: Option<&str>) -> String {
    match context.map(str::trim) {
        Some(c) if !c.is_empty() && c != query.trim() => format!("(This is one part of the person's longer message: \"{c}\" Keep the whole message in mind, but answer only the question above.)\n"),
        _ => String::new(),
    }
}

/// What a part is asked: to *answer from itself* and quote its proof, with no cap on how much it says. ("Copy out
/// the sentences that bear on it, at most three" took the first lines that mentioned the subject and dropped the
/// one that answered, 800 tokens later.)
pub fn extract_prompt(query: &str, context: Option<&str>) -> String {
    format!(
        "A reader asks: \"{query}\"\n{}Using only this part of the book, tell me everything in it that bears on that question. Answer what it answers, and quote - exactly as written - the lines that show it. \
Include every relevant fact, event and spoken line, and who says or does it, in the order they appear in the part - including anything later in the part that changes a figure or an outcome (points added, a decision reversed); leave out what does not bear on the question. \
If this part does not bear on it at all, reply with exactly: NONE",
        context_note(query, context)
    )
}

/// The words the second half of a part starts with: a paragraph or sentence boundary just past its middle, and
/// the next few words, exactly as written.
pub fn mid_anchor(text: &str) -> Option<String> {
    let mid = text.len() / 2;
    let mut start = mid;
    while !text.is_char_boundary(start) {
        start += 1;
    }
    let rest = &text[start..];
    let cut = [rest.find("\n\n").map(|i| i + 2), rest.find(". ").map(|i| i + 2), rest.find("? ").map(|i| i + 2), rest.find("! ").map(|i| i + 2)].into_iter().flatten().min().or_else(|| rest.find(char::is_whitespace))?;
    let words: Vec<&str> = rest[cut..].split_whitespace().take(10).collect();
    (words.len() >= 4).then(|| words.join(" "))
}

/// Asked after the first extraction, in the same conversation. "Anything else?" was measured not to help - the
/// model says NONE. Pointing it at the exact words the second half begins with does.
pub fn second_half_prompt(query: &str, anchor: &str) -> String {
    format!("Now read only the second half of this part, from the words \"{anchor}\" to its very end, closely, line by line. Quote exactly as written every line there that bears on the question (\"{query}\") - including anything that changes a figure or an outcome - with who says or does it, in the order it appears. Leave out what you already gave. If nothing in the second half bears on it, reply with exactly: NONE")
}

/// An answer followed by NOT FOUND ("... does not address it directly. NOT FOUND") keeps the answer.
pub fn strip_trailing_not_found(answer: &str) -> String {
    let a = answer.trim();
    let lower = a.to_ascii_lowercase();
    let stripped = lower.trim_end_matches(|c: char| !c.is_alphanumeric()).strip_suffix("not found");
    match stripped {
        Some(rest) if !rest.trim().is_empty() => a[..rest.len()].trim_end().to_string(),
        _ => a.to_string(),
    }
}

/// True for a reply that means "nothing here".
pub fn is_none(reply: &str) -> bool {
    let t = reply.trim().trim_matches(|c: char| !c.is_alphanumeric());
    t.is_empty() || t.eq_ignore_ascii_case("none") || t.to_ascii_lowercase().starts_with("none ")
}

pub fn score_prompt(query: &str) -> String {
    format!("A reader asks: \"{query}\"\nHow much of this part of the book bears on that question? Reply with a single digit from 0 (nothing at all) to 9 (a great deal), and nothing else.")
}

/// The digit a model answered with; `None` if it answered with something else.
pub fn parse_score(reply: &str) -> Option<u8> {
    reply.chars().find(|c| c.is_ascii_digit()).map(|c| c as u8 - b'0')
}

/// The parts to read closely: **all** whose score says they bear on the question, in document order, with no cap.
/// If none reaches [`MIN_SCORE`] but something scores at all, the nearest two.
pub fn relevant_parts(scores: &[u8]) -> Vec<usize> {
    let mut chosen: Vec<usize> = (0..scores.len()).filter(|i| scores[*i] >= MIN_SCORE).collect();
    if chosen.is_empty() {
        let mut ranked: Vec<usize> = (0..scores.len()).collect();
        ranked.sort_by(|a, b| scores[*b].cmp(&scores[*a]).then(a.cmp(b)));
        chosen = ranked.into_iter().filter(|i| scores[*i] >= 2).take(2).collect();
        chosen.sort_unstable();
    }
    chosen
}

pub fn reword_prompt(query: &str, context: Option<&str>) -> String {
    format!(
        "A reader asks: \"{query}\"
{}The book uses older, different words from a modern reader (the way an old translation says \"wrath\" or \"passion\" for anger, or \"sorrow\" for pain). Rewrite the question using the words and ideas this book itself uses. Keep it one question. Reply with the rewritten question only.",
        context_note(query, context)
    )
}

/// For a part scored high that could not be quoted from: its closest lines and how they bear on the question.
pub fn gist_prompt(query: &str, context: Option<&str>) -> String {
    format!("A reader asks: \"{query}\"
{}This part does bear on it, though not in the reader's words. Copy out, exactly as written, the one or two lines of this part that come closest, then a dash, then one plain sentence saying how they bear on the question. Say nothing the part does not say. If nothing does, reply with exactly: NONE", context_note(query, context))
}

/// A backstop for the one-digit scoring: the parts whose own words match the question's (rarer words weigh more -
/// "feast" and "cup" say more than "house"), needing two distinct matching words when the question has that many.
/// The close reading decides whether they really answer it, so a wrong pick costs time, never a wrong answer.
pub fn lexical_parts(query: &str, texts: &[&str], k: usize) -> Vec<usize> {
    let mut terms: Vec<String> = Vec::new();
    for w in query.split(|c: char| !c.is_alphabetic()).filter(|w| w.chars().count() >= 4) {
        let lw = w.to_lowercase();
        if STOP.contains(&lw.as_str()) {
            continue;
        }
        let s = stem(&lw);
        if !terms.contains(&s) {
            terms.push(s);
        }
    }
    if terms.is_empty() || texts.is_empty() {
        return Vec::new();
    }
    let counts: Vec<Vec<usize>> = texts
        .iter()
        .map(|t| {
            let mut c = vec![0usize; terms.len()];
            for w in t.split(|ch: char| !ch.is_alphabetic()).filter(|w| !w.is_empty()) {
                if let Some(i) = terms.iter().position(|x| *x == stem(w)) {
                    c[i] += 1;
                }
            }
            c
        })
        .collect();
    let n = texts.len() as f64;
    let idf: Vec<f64> = (0..terms.len()).map(|t| ((n + 1.0) / counts.iter().filter(|c| c[t] > 0).count().max(1) as f64).ln()).collect();
    let need = terms.len().min(2);
    let mut ranked: Vec<(usize, f64)> = counts
        .iter()
        .enumerate()
        .filter(|(_, c)| c.iter().filter(|x| **x > 0).count() >= need)
        .map(|(i, c)| (i, (0..terms.len()).filter(|t| c[*t] > 0).map(|t| idf[t] * (1.0 + (c[t] as f64).ln())).sum::<f64>()))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    ranked.into_iter().take(k).map(|(i, _)| i).collect()
}

fn count_swap(out: &mut Reading, swap: &Swap) -> bool {
    let again = matches!(swap, Swap::Reprimed { .. } | Swap::NotSaved);
    if again {
        out.read_afresh += 1;
    } else {
        out.recalled += 1;
    }
    again
}

fn score_parts(pool: &KvPool, corpus: &dyn Corpus, query: &str, on_stage: &dyn Fn(&str), out: &mut Reading, only: Option<&[usize]>) -> Result<Vec<u8>> {
    let total = corpus.part_count();
    let mut scores = Vec::with_capacity(total);
    for i in 0..total {
        if only.is_some_and(|o| !o.contains(&i)) {
            scores.push(0);
            continue;
        }
        on_stage(&format!("Reading part {} of {}", i + 1, total));
        let prefix = corpus.part_prefix(i);
        let (c, swap) = pool.with_resident(corpus.part_file(i), corpus.part_owner(i), &prefix, |llm| llm.complete(&chatml::ask(&prefix, &score_prompt(query)), &Sampling::precise(SCORE_TOKENS), &mut |_| {}))?;
        out.processed_tokens += c.prompt_tokens;
        let again = count_swap(out, &swap);
        // A reply that is not a digit is no evidence the part is irrelevant: read it closely.
        scores.push(parse_score(&c.text).unwrap_or(5));
        on_stage(&format!("Part {} of {}: {}", i + 1, total, if again { "read afresh" } else { "recalled from memory" }));
    }
    Ok(scores)
}

/// Reads parts `chosen` (0-based, in the order given) closely: each answers from itself with its proof, then its
/// second half is read again. Parts in `strong` (scored high) that yield nothing are asked for their closest lines.
pub fn read_parts(pool: &KvPool, corpus: &dyn Corpus, query: &str, chosen: &[usize], strong: &[usize], opts: ReadOptions, on_stage: &dyn Fn(&str), out: &mut Reading) -> Result<()> {
    let context = opts.context;
    for (n, i) in chosen.iter().enumerate() {
        on_stage(&format!("Looking closely at part {} ({} of {})", i + 1, n + 1, chosen.len()));
        let file = corpus.part_file(*i);
        let owner = corpus.part_owner(*i);
        let prefix = corpus.part_prefix(*i);
        let (c, swap) = pool.with_resident(file, owner, &prefix, |llm| llm.complete(&chatml::ask(&prefix, &extract_prompt(query, context)), &Sampling::precise(LEAF_TOKENS), &mut |_| {}))?;
        if out.scores.is_empty() {
            // no scoring pass (Mode 3): this is where the part was recalled or read
            count_swap(out, &swap);
        }
        out.processed_tokens += c.prompt_tokens;
        if c.truncated {
            out.truncated += 1;
        }
        let mut text = c.text.trim().to_string();
        if is_none(&text) && strong.contains(i) {
            let (g, _) = pool.with_resident(file, owner, &prefix, |llm| llm.complete(&chatml::ask(&prefix, &gist_prompt(query, context)), &Sampling::precise(LEAF_TOKENS / 2), &mut |_| {}))?;
            out.processed_tokens += g.prompt_tokens;
            text = g.text.trim().to_string();
        }
        if let Some(anchor) = mid_anchor(corpus.part_text(*i)) {
            let first = if text.is_empty() { "NONE".to_string() } else { text.clone() };
            let again = format!("{prefix}{}{}{}{}", chatml::user(&extract_prompt(query, context)), chatml::assistant(&first), chatml::user(&second_half_prompt(query, &anchor)), chatml::assistant_open());
            let (m, _) = pool.with_resident(file, owner, &prefix, |llm| llm.complete(&again, &Sampling::precise(LEAF_TOKENS), &mut |_| {}))?;
            out.processed_tokens += m.prompt_tokens;
            let more = strip_trailing_not_found(m.text.trim());
            if !is_none(&more) {
                text = if is_none(&text) { more } else { format!("{text}\n\n{more}") };
            }
        }
        if !is_none(&text) {
            out.passages.push(Passage { part: *i + 1, book: corpus.part_source(*i), text });
        }
    }
    out.read_closely += chosen.len();
    Ok(())
}

/// Mode 4: scores every part, then reads closely every part that scores (plus the word-match backstop). If nothing
/// scores - a modern word the document never uses ("hatred" where an old translation says "wrath") - the question
/// is put in the document's own words and scored once more.
pub fn scan(pool: &KvPool, corpus: &dyn Corpus, query: &str, opts: ReadOptions, on_stage: &dyn Fn(&str)) -> Result<Reading> {
    let started = std::time::Instant::now();
    let mut out = Reading::default();
    let mut query = query.to_string();

    let mut scores = score_parts(pool, corpus, &query, on_stage, &mut out, opts.only)?;
    if relevant_parts(&scores).is_empty() {
        on_stage("Putting the question in the book's own words");
        let gist = corpus.gist();
        let system = chatml::system(&format!("You are reading the book \"{}\".{}", corpus.title(), if gist.is_empty() { String::new() } else { format!(" Its central ideas:\n{gist}") }));
        let raw = pool.with_scratch(|llm| Ok(llm.complete(&chatml::ask(&system, &reword_prompt(&query, opts.context)), &Sampling::precise(80), &mut |_| {})?.text))?;
        let reworded = clean_rephrasing(&raw, &query);
        if reworded != query {
            query = reworded.clone();
            out.reworded = Some(reworded);
            scores = score_parts(pool, corpus, &query, on_stage, &mut out, opts.only)?;
        }
    }
    let texts: Vec<&str> = (0..corpus.part_count()).map(|i| corpus.part_text(i)).collect();
    for i in lexical_parts(&query, &texts, LEXICAL_PARTS) {
        if opts.only.is_none_or(|o| o.contains(&i)) && scores[i] < RETRY_SCORE {
            scores[i] = RETRY_SCORE;
        }
    }
    out.scores = scores;
    let chosen = relevant_parts(&out.scores);
    let strong: Vec<usize> = chosen.iter().copied().filter(|i| out.scores[*i] >= RETRY_SCORE).collect();
    read_parts(pool, corpus, &query, &chosen, &strong, opts, on_stage, &mut out)?;
    out.ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(out)
}

/// The passages as they appear in a prompt, bounded.
pub fn passages_block(passages: &[Passage], max_chars: usize) -> String {
    let mut out = String::new();
    for p in passages {
        let label = if p.book.is_empty() { format!("part {}", p.part) } else { format!("{}, part {}", p.book, p.part) };
        let piece = format!("[{label}]\n{}\n\n", p.text);
        if out.chars().count() + piece.chars().count() > max_chars {
            break;
        }
        out.push_str(&piece);
    }
    out.trim_end().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_second_half_is_pointed_at_by_its_first_words() {
        let text = "One two three four five six. Seven eight nine ten eleven twelve. Thirteen fourteen fifteen sixteen. Seventeen eighteen nineteen twenty twentyone twentytwo.";
        let a = mid_anchor(text).unwrap();
        assert!(text.contains(&a) && text.find(&a).unwrap() >= text.len() / 2, "{a}");
        assert!(a.starts_with("Seventeen"), "a sentence start just past the middle: {a}");
        assert!(mid_anchor("too short").is_none());
        let p = second_half_prompt("Why?", &a);
        assert!(p.contains(&a) && p.contains("second half") && p.contains("NONE"));
    }

    #[test]
    fn the_word_match_finds_the_part_that_uses_the_questions_rare_words() {
        let texts = ["Harry flew his broom and won points in a match.", "At the feast the House Cup went to Gryffindor: points were counted, the cup awarded.", "Hagrid cooked a rock cake. More points were lost.", "Snape brewed a potion."];
        let got = lexical_parts("How did the house cup feast points end up?", &texts, 2);
        assert_eq!(got.first(), Some(&1), "{got:?}");
        assert!(lexical_parts("What is this?", &texts, 2).is_empty(), "nothing to search with");
        assert!(lexical_parts("Tell me about potions", &texts, 3).contains(&3), "plural and singular match");
    }

    #[test]
    fn none_replies_are_recognised_despite_punctuation_and_trailing_words() {
        for s in ["NONE", "none.", "\"None\"", "  None \n", "None - nothing here", ""] {
            assert!(is_none(s), "{s:?}");
        }
        for s in ["Nonetheless the verse says duty", "The soul is never born.", "Krishna speaks"] {
            assert!(!is_none(s), "{s:?}");
        }
    }

    #[test]
    fn passages_are_labelled_in_reading_order_and_bounded() {
        let ps: Vec<Passage> = (1..=50).map(|n| Passage { part: n, book: String::new(), text: "verse ".repeat(60) }).collect();
        let b = passages_block(&ps, 3000);
        assert!(b.starts_with("[part 1]") && b.chars().count() <= 3000);
        assert!(!b.contains("[part 50]"), "the bound drops the tail, not the head");
    }

    #[test]
    fn an_answer_followed_by_not_found_keeps_the_answer() {
        assert_eq!(strip_trailing_not_found("It does not address it directly. NOT FOUND"), "It does not address it directly.");
        assert_eq!(strip_trailing_not_found("NOT FOUND"), "NOT FOUND");
        assert_eq!(strip_trailing_not_found("Duty is the way."), "Duty is the way.");
    }

    #[test]
    fn scores_are_read_from_whatever_the_model_wrote_around_the_digit() {
        assert_eq!((parse_score("7"), parse_score(" 3."), parse_score("Score: 9"), parse_score("nothing")), (Some(7), Some(3), Some(9), None));
    }

    #[test]
    fn every_part_that_scores_is_read_in_order_with_no_cap() {
        assert_eq!(relevant_parts(&[0, 8, 2, 9, 1, 3, 7, 6]), vec![1, 3, 5, 6, 7]);
        let many: Vec<u8> = (0..40).map(|i| if i % 2 == 0 { 9 } else { 0 }).collect();
        assert_eq!(relevant_parts(&many).len(), 20);
        assert_eq!(relevant_parts(&[0, 0, 0, 0]), Vec::<usize>::new());
        assert_eq!(relevant_parts(&[0, 2, 1, 2, 0]), vec![1, 3], "a faint match still reads the nearest two");
    }

    #[test]
    fn a_question_drawn_from_a_longer_message_carries_it_except_when_scoring() {
        let whole = "I was not accepted last year and feel I failed. Is god testing my resilience?";
        let step = "Is god testing my resilience?";
        for prompt in [extract_prompt(step, Some(whole)), reword_prompt(step, Some(whole)), gist_prompt(step, Some(whole))] {
            assert!(prompt.contains(step) && prompt.contains("I was not accepted last year"), "{prompt}");
        }
        assert!(!extract_prompt(step, Some(step)).contains("longer message"));
        assert!(!score_prompt(step).contains("longer message"));
    }

    #[test]
    fn the_extraction_prompt_asks_for_the_parts_own_words_with_no_cap_and_a_none_escape() {
        let p = extract_prompt("what is duty?", None);
        assert!(p.contains("exactly as written") && p.contains("NONE") && p.contains("tell me everything in it that bears on that question") && !p.contains("at most three"));
    }
}
