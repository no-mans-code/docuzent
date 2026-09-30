//! Splitting a book into *parts* - the units whose KV state is saved.
//!
//! A part must fit the model's slot with room left for a question and its
//! answer, so the limit is in **tokens**, measured with the model's own
//! tokenizer - not guessed from characters. Parts break between paragraphs
//! (never mid-sentence), so what a part contains reads as text and a
//! passage is not cut in two at a boundary where it could be missed.
//!
//! The split is deterministic: the same text always gives the same parts,
//! and each part carries a hash of its text, so a saved KV file can be
//! recognised as still valid (or not) from its name alone.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use docuzent_llm::chatml;

use crate::hash;

/// Tokens of book text per part. About 4,000 tokens is ~15,000 characters:
/// large enough that a 35,000-token book is nine parts, small enough that
/// one part's saved state is a few hundred megabytes and restores in a
/// tenth of a second.
pub const PART_TOKEN_LIMIT: usize = 4000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Part {
    pub index: usize,
    pub text: String,
    /// Exact tokens of `text` under the model's tokenizer.
    pub tokens: usize,
    /// SHA-256 of `text`.
    pub hash: String,
}

impl Part {
    /// The KV file "kind": position and content, so a re-split or an edited
    /// book can never be served an old part's saved state.
    pub fn kv_kind(&self) -> String {
        format!("part-{:03}-{}", self.index + 1, &self.hash[..8])
    }

    /// The text whose KV state is saved: everything about this part that
    /// is the same for every question, as one complete system turn.
    pub fn prefix(&self, title: &str, author: Option<&str>, total: usize) -> String {
        let by = author.map(|a| format!(" by {a}")).unwrap_or_default();
        chatml::system(&format!("You are reading part {} of {total} of the book \"{title}\"{by}. Read it closely: you will be asked about it, and must answer only from what it says.\n\n{}", self.index + 1, self.text))
    }
}

/// Splits `text` into parts of at most `limit` tokens, breaking between
/// paragraphs. `count` is the exact tokenizer.
pub fn split_into_parts(text: &str, limit: usize, count: &dyn Fn(&str) -> Result<usize>) -> Result<Vec<Part>> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(Vec::new());
    }
    // Chars per token, measured on this very book, to size greedy fills
    // without tokenizing every paragraph. Every finished part is then
    // verified exactly, so the estimate can only cost a re-split, never a
    // part that is too big.
    let sample: String = text.chars().take(20_000).collect();
    let sample_tokens = count(&sample)?.max(1);
    let chars_per_token = (sample.chars().count() as f64 / sample_tokens as f64).max(1.0);
    let target_chars = (limit as f64 * chars_per_token * 0.97) as usize;

    let mut units: std::collections::VecDeque<String> = paragraphs(text, target_chars).into();
    let mut parts = Vec::new();
    while !units.is_empty() {
        let mut current = String::new();
        while let Some(next) = units.front() {
            let joined_len = current.chars().count() + next.chars().count() + if current.is_empty() { 0 } else { 2 };
            if !current.is_empty() && joined_len > target_chars {
                break;
            }
            let next = units.pop_front().unwrap();
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(&next);
        }
        // Exact check: give paragraphs back until it fits.
        let mut tokens = count(&current)?;
        while tokens > limit {
            match current.rfind("\n\n") {
                Some(cut) => {
                    let (keep, give_back) = current.split_at(cut);
                    units.push_front(give_back.trim_start().to_string());
                    current = keep.to_string();
                    tokens = count(&current)?;
                }
                None => {
                    // One paragraph alone is too big for the estimate: halve it on a word boundary.
                    let (a, b) = halve(&current);
                    units.push_front(b);
                    current = a;
                    tokens = count(&current)?;
                }
            }
        }
        parts.push(Part { index: parts.len(), hash: hash::hash_bytes(current.as_bytes()), text: current, tokens });
    }
    Ok(parts)
}

/// Paragraphs, with any paragraph larger than `max_chars` broken on line
/// then word boundaries so no unit is bigger than a part.
fn paragraphs(text: &str, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    for p in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        if p.chars().count() <= max_chars {
            out.push(p.to_string());
            continue;
        }
        let mut rest = p.to_string();
        while rest.chars().count() > max_chars {
            let (a, b) = halve_at(&rest, max_chars);
            out.push(a);
            rest = b;
        }
        if !rest.trim().is_empty() {
            out.push(rest.trim().to_string());
        }
    }
    out
}

fn halve(s: &str) -> (String, String) {
    halve_at(s, s.chars().count() / 2)
}

/// Splits `s` at about `at` characters, on a line break if there is one
/// nearby, else on whitespace, else exactly.
fn halve_at(s: &str, at: usize) -> (String, String) {
    let chars: Vec<char> = s.chars().collect();
    let at = at.clamp(1, chars.len().saturating_sub(1).max(1));
    let window_start = at.saturating_sub(at / 4);
    let cut = chars[window_start..at].iter().rposition(|c| *c == '\n').or_else(|| chars[window_start..at].iter().rposition(|c| c.is_whitespace())).map(|i| window_start + i + 1).unwrap_or(at);
    (chars[..cut].iter().collect::<String>().trim_end().to_string(), chars[cut..].iter().collect::<String>().trim_start().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The simulated tokenizer: four characters per token.
    fn count(s: &str) -> Result<usize> {
        Ok(s.chars().count().div_ceil(4))
    }

    fn book(paragraphs: usize) -> String {
        (0..paragraphs).map(|i| format!("Paragraph {i}. {}", "The soul is unborn and eternal, and it cannot be cut. ".repeat(6)).trim_end().to_string()).collect::<Vec<_>>().join("\n\n")
    }

    #[test]
    fn every_part_fits_the_limit_and_nothing_is_lost_or_reordered() {
        let text = book(60);
        let parts = split_into_parts(&text, 500, &count).unwrap();
        assert!(parts.len() > 3);
        assert!(parts.iter().all(|p| p.tokens <= 500 && p.tokens == count(&p.text).unwrap()));
        let rejoined = parts.iter().map(|p| p.text.as_str()).collect::<Vec<_>>().join("\n\n");
        assert_eq!(rejoined, text, "concatenating the parts gives the book back exactly");
        assert!(parts.iter().enumerate().all(|(i, p)| p.index == i));
    }

    #[test]
    fn parts_are_filled_not_fragmented() {
        let parts = split_into_parts(&book(60), 500, &count).unwrap();
        let (last, rest) = parts.split_last().unwrap();
        assert!(rest.iter().all(|p| p.tokens > 350), "each part but the last is well filled: {:?}", rest.iter().map(|p| p.tokens).collect::<Vec<_>>());
        assert!(last.tokens > 0);
    }

    #[test]
    fn a_paragraph_bigger_than_a_part_is_split_on_word_boundaries() {
        let huge = "word ".repeat(3000);
        let parts = split_into_parts(&huge, 300, &count).unwrap();
        assert!(parts.len() > 3 && parts.iter().all(|p| p.tokens <= 300));
        assert!(parts.iter().all(|p| p.text.split_whitespace().all(|w| w == "word")), "no word was cut in half");
    }

    #[test]
    fn the_split_is_deterministic_and_hashes_track_content() {
        let a = split_into_parts(&book(30), 400, &count).unwrap();
        let b = split_into_parts(&book(30), 400, &count).unwrap();
        assert_eq!(a, b);
        let edited = split_into_parts(&book(30).replacen("Paragraph 0", "Paragraph zero", 1), 400, &count).unwrap();
        assert_ne!(a[0].kv_kind(), edited[0].kv_kind(), "an edited part can never be served the old saved state");
    }

    #[test]
    fn empty_text_has_no_parts_and_a_short_book_is_one() {
        assert!(split_into_parts("  \n\n ", 400, &count).unwrap().is_empty());
        assert_eq!(split_into_parts("A short book.", 400, &count).unwrap().len(), 1);
    }

    #[test]
    fn the_prefix_is_a_complete_system_turn_naming_the_part() {
        let p = &split_into_parts("Some text.", 400, &count).unwrap()[0];
        let prefix = p.prefix("The Song", Some("Arnold"), 9);
        assert!(prefix.starts_with("<|im_start|>system\n") && prefix.ends_with("<|im_end|>\n"));
        assert!(prefix.contains("part 1 of 9 of the book \"The Song\" by Arnold") && prefix.contains("Some text."));
    }
}
