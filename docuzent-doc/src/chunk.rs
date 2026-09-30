//! Turns a Docling JSON export into big, page-shaped chunks - "big chunks
//! that fit in memory," not many small ones. Docling emits a flat `texts`
//! array where each item carries `prov[0].page_no`; consecutive items on
//! the same page are coalesced into one chunk, only split further if a
//! single page's text still exceeds `max_chars`.

use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

/// Generous default: "big chunks," not the small token windows a code-RAG
/// would use. Large enough to hold a real page of prose, safely inside any
/// local embedding model's input limit.
pub const DEFAULT_MAX_CHARS: usize = 4000;

#[derive(Deserialize)]
struct DoclingDoc {
    #[serde(default)]
    texts: Vec<DoclingText>,
}

#[derive(Deserialize)]
struct DoclingText {
    text: String,
    #[serde(default)]
    prov: Vec<Prov>,
}

#[derive(Deserialize)]
struct Prov {
    page_no: u32,
}

/// One chunk ready to be embedded and stored.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkInput {
    pub chunk_index: usize,
    pub page: Option<u32>,
    pub text: String,
}

/// Parses a Docling JSON export and groups its text items into big,
/// page-bounded chunks.
pub fn chunk_docling_json(json_path: &Path, max_chars: usize) -> Result<Vec<ChunkInput>> {
    let raw = std::fs::read_to_string(json_path)
        .with_context(|| format!("failed to read {}", json_path.display()))?;
    let doc: DoclingDoc = serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse docling JSON at {}", json_path.display()))?;
    Ok(chunk_texts(&doc.texts, max_chars))
}

fn chunk_texts(texts: &[DoclingText], max_chars: usize) -> Vec<ChunkInput> {
    let mut pages: Vec<(Option<u32>, String)> = Vec::new();
    for item in texts {
        if item.text.trim().is_empty() {
            continue;
        }
        let page = item.prov.first().map(|p| p.page_no);
        match pages.last_mut() {
            Some((last_page, buf)) if *last_page == page => {
                buf.push('\n');
                buf.push_str(&item.text);
            }
            _ => pages.push((page, item.text.clone())),
        }
    }

    let mut chunks = Vec::new();
    let mut idx = 0;
    for (page, text) in pages {
        for piece in split_to_max(&text, max_chars) {
            chunks.push(ChunkInput { chunk_index: idx, page, text: piece });
            idx += 1;
        }
    }
    chunks
}

/// Splits text into pieces no larger than `max_chars`, breaking on
/// whitespace near the boundary rather than mid-word. Most pages never hit
/// this at all - it exists only for the rare page whose text overruns the
/// cap on its own. `pub(crate)` since [`crate::session`] reuses it for
/// plain-text map-reduce splitting, not just page-sized chunking.
pub fn split_to_max(text: &str, max_chars: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return vec![text.trim().to_string()];
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let mut end = (start + max_chars).min(chars.len());
        if end < chars.len() {
            if let Some(back) = chars[start..end].iter().rposition(|c| c.is_whitespace()) {
                if back > 0 {
                    end = start + back;
                }
            }
        }
        let piece: String = chars[start..end].iter().collect();
        let piece = piece.trim().to_string();
        if !piece.is_empty() {
            out.push(piece);
        }
        start = end.max(start + 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str, page: u32) -> DoclingText {
        DoclingText { text: s.to_string(), prov: vec![Prov { page_no: page }] }
    }

    #[test]
    fn coalesces_same_page_items_into_one_chunk() {
        let items = vec![text("first paragraph", 1), text("second paragraph", 1), text("page two", 2)];
        let chunks = chunk_texts(&items, DEFAULT_MAX_CHARS);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].page, Some(1));
        assert_eq!(chunks[0].text, "first paragraph\nsecond paragraph");
        assert_eq!(chunks[1].page, Some(2));
        assert_eq!(chunks[1].text, "page two");
    }

    #[test]
    fn splits_an_oversized_page() {
        let big = "word ".repeat(1000); // ~5000 chars, over a 100-char cap
        let items = vec![text(&big, 1)];
        let chunks = chunk_texts(&items, 100);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(c.text.chars().count() <= 100);
            assert_eq!(c.page, Some(1));
        }
    }

    #[test]
    fn skips_empty_text_items() {
        let items = vec![text("   ", 1), text("real content", 1)];
        let chunks = chunk_texts(&items, DEFAULT_MAX_CHARS);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "real content");
    }
}
