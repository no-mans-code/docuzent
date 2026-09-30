//! A document's real title and author, clean.
//!
//! File names are a poor source ("Harry Potter and The Philosopher's Stone (SDoc).pdf", "vdoc.pub_50-successful-ivy-
//! league-...pdf", "0-The art of not giving a damn.pdf", "FRANKL_Viktor_Man's_Search_For_Meaning-1963_text.pdf").
//! The title page is a good one. So:
//!
//! 1. [`clean_file_title`] tidies the file name without a model: extension, site and scanner tags, "PDF", leading
//!    numbering, underscores and URL escapes gone. This is the fallback, and a hint to step 2.
//! 2. [`detect`] shows the model the document's first pages (about ten - [`HEAD_CHARS`]) and the tidied file name,
//!    and asks for the title and author as the document itself prints them.
//! 3. Whatever comes back must pass [`hygienic`]: one line, a sane length, no file junk, not a sentence about the
//!    book. Otherwise the tidied file name stands.
//!
//! A title the person gives themselves always wins over both - that is the caller's to keep.

use anyhow::Result;
use docuzent_llm::{chatml, json, Llm, Sampling};
use serde::{Deserialize, Serialize};

/// About ten printed pages of prose.
pub const HEAD_CHARS: usize = 20_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DocTitle {
    pub title: String,
    pub author: Option<String>,
    /// Where it came from: `document` (read from its first pages) or `file` (the file name, tidied).
    pub source: String,
}

/// Tags that scanners, download sites and converters add to file names.
const JUNK: &[&str] = &["sdoc", "pdf", "epub", "mobi", "ebook", "e-book", "text", "txt", "scan", "scanned", "ocr", "vdoc.pub", "vdoc", "libgen", "z-lib", "zlib", "dokumen.pub", "pdfdrive", "www", "com", "org", "final", "copy", "retail", "converted"];

fn is_junk(word: &str) -> bool {
    let w = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '.').to_ascii_lowercase();
    JUNK.contains(&w.as_str())
}

/// The file name, tidied: no extension, no site or scanner tags, no leading numbering, words separated by spaces.
pub fn clean_file_title(file_name: &str) -> String {
    let base = file_name.rsplit(['/', '\\']).next().unwrap_or(file_name);
    let stem = match base.rfind('.') {
        Some(i) if base.len() - i <= 6 => &base[..i],
        _ => base,
    };
    let decoded = percent_decode(stem);
    // bracketed tags: "(SDoc)", "[PDF]", "{ebook}"
    let mut s = String::new();
    let mut depth = 0i32;
    let mut bracket = String::new();
    for c in decoded.chars() {
        match c {
            '(' | '[' | '{' => {
                depth += 1;
                bracket.clear();
            }
            ')' | ']' | '}' if depth > 0 => {
                depth -= 1;
                // keep a bracketed part that is real words ("(Bhagavad-Gita)"), drop tags ("(SDoc)", "[PDF]")
                if !bracket.split_whitespace().all(is_junk) {
                    s.push_str(&format!("({})", bracket.trim()));
                }
            }
            _ if depth > 0 => bracket.push(c),
            _ => s.push(c),
        }
    }
    let spaced: String = s.chars().map(|c| if c == '_' || c == '+' { ' ' } else { c }).collect();
    // "vdoc.pub" and friends before splitting on dashes and dots
    let mut words: Vec<String> = spaced.split_whitespace().flat_map(|w| if w.contains("vdoc.pub") || w.contains("dokumen.pub") { vec![w.replace("vdoc.pub", " ").replace("dokumen.pub", " ")] } else { vec![w.to_string()] }).flat_map(|w| w.split_whitespace().map(str::to_string).collect::<Vec<_>>()).collect();
    // a word joined by hyphens is kept ("Bhagavad-Gita") unless the whole name was hyphen-separated
    let hyphenated = words.len() <= 2 && words.iter().any(|w| w.matches('-').count() >= 2);
    if hyphenated {
        words = words.iter().flat_map(|w| w.split('-').map(str::to_string).collect::<Vec<_>>()).collect();
    }
    let kept: Vec<String> = words.into_iter().map(|w| w.trim_matches(|c: char| c == '-' || c == '.').to_string()).filter(|w| !w.is_empty() && !is_junk(w)).collect();
    // leading numbering - "0-The art", "01 Title" - but not a number that is part of the title ("50 Essays", "1984")
    let numbering = |w: &str| !w.is_empty() && w.chars().all(|c| c.is_ascii_digit()) && w.starts_with('0');
    let mut joined = kept.join(" ");
    loop {
        let first: String = joined.chars().take_while(|c| c.is_ascii_digit()).collect();
        let rest = &joined[first.len()..];
        if numbering(&first) && (rest.is_empty() || rest.starts_with([' ', '-', '.'])) {
            joined = rest.trim_start_matches([' ', '-', '.']).to_string();
        } else {
            break;
        }
    }
    let joined = joined.trim().to_string();
    if joined.is_empty() {
        return "Untitled".into();
    }
    // all lower case or all upper case: title case it
    if joined == joined.to_lowercase() || joined == joined.to_uppercase() {
        title_case(&joined)
    } else {
        joined
    }
}

fn title_case(s: &str) -> String {
    const SMALL: &[&str] = &["a", "an", "and", "as", "at", "but", "by", "for", "in", "nor", "of", "on", "or", "the", "to", "with"];
    s.split_whitespace()
        .enumerate()
        .map(|(i, w)| {
            let l = w.to_lowercase();
            if i > 0 && SMALL.contains(&l.as_str()) {
                l
            } else {
                let mut c = l.chars();
                c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(b) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

/// A title fit to show: one line, 1-150 characters, with no file junk, and not a sentence *about* the document.
pub fn hygienic(title: &str) -> Option<String> {
    let t = title.trim().trim_matches(|c: char| c == '"' || c == '\'' || c == '*' || c == '`' || c == '“' || c == '”').trim();
    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    let t = t.trim_end_matches(|c: char| c == '.' || c == ',' || c == ';' || c == ':').trim().to_string();
    let lower = t.to_lowercase();
    if t.is_empty() || t.chars().count() > 150 || t.contains('\n') {
        return None;
    }
    if ["this book", "the title is", "this document", "unknown", "untitled", "n/a", "not stated", "i cannot", "i can't"].iter().any(|p| lower.starts_with(p)) {
        return None;
    }
    if lower.ends_with(".pdf") || lower.ends_with(".epub") || lower.contains("vdoc.pub") || lower.contains("(sdoc)") || lower.contains('_') {
        return None;
    }
    Some(t)
}

pub fn title_prompt(file_title: &str, head: &str) -> String {
    format!(
        "Here are the first pages of a document, and the name of the file it came in (\"{file_title}\").\n\n---\n{head}\n---\n\nWhat is this document's real title, and who wrote it, as the document itself prints them (its title page, cover or opening)? Use the proper capitalisation. Leave out series names, publishers, site names, edition notes, file tags and \"PDF\"; keep a subtitle only if it is part of how the book is known. If the pages do not show a title, use the file's name, tidied. Reply with JSON only: {{\"title\": \"...\", \"author\": \"...\" or null}}"
    )
}

/// The document's first pages - what a person would look at to see what it is.
pub fn head(text: &str) -> String {
    let mut end = text.len().min(HEAD_CHARS);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

/// Reads the title and author from the document's first pages; falls back to the tidied file name.
pub fn detect(llm: &dyn Llm, file_name: &str, text: &str) -> Result<DocTitle> {
    let file_title = clean_file_title(file_name);
    let fallback = DocTitle { title: file_title.clone(), author: None, source: "file".into() };
    if text.trim().is_empty() {
        return Ok(fallback);
    }
    let reply = llm.complete(&chatml::ask("", &title_prompt(&file_title, &head(text))), &Sampling::precise(120), &mut |_| {})?.text;
    Ok(parse_reply(&reply).unwrap_or(fallback))
}

/// The model's `{"title", "author"}`, if it is a clean one.
pub fn parse_reply(reply: &str) -> Option<DocTitle> {
    let v: serde_json::Value = serde_json::from_str(json::first_object(reply)?).ok()?;
    let title = hygienic(v["title"].as_str()?)?;
    let author = v["author"].as_str().and_then(hygienic).filter(|a| a.chars().count() <= 100 && !a.to_lowercase().contains("unknown"));
    Some(DocTitle { title, author, source: "document".into() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use docuzent_llm::SimEngine;

    #[test]
    fn file_names_seen_in_the_wild_are_tidied() {
        for (file, want) in [
            ("Harry Potter and The Philosopher's Stone (SDoc).pdf", "Harry Potter and The Philosopher's Stone"),
            ("vdoc.pub_50-successful-ivy-league-application-essays.pdf", "50 Successful Ivy League Application Essays"),
            ("0-The art of not giving a damn.pdf", "The art of not giving a damn"),
            ("FRANKL_Viktor_Man's_Search_For_Meaning-1963_text.pdf", "FRANKL Viktor Man's Search For Meaning-1963"),
            ("Talking with Psychopaths and Savages PDF.pdf", "Talking with Psychopaths and Savages"),
            ("gita.txt", "Gita"),
            ("01 - Nineteen Eighty-Four.epub", "Nineteen Eighty-Four"),
            ("1984.epub", "1984"),
            ("The%20Song%20Celestial%20(Bhagavad-Gita).epub", "The Song Celestial (Bhagavad-Gita)"),
            ("THE_REPUBLIC.txt", "The Republic"),
            (".pdf", "Untitled"),
        ] {
            assert_eq!(clean_file_title(file), want, "{file}");
        }
    }

    #[test]
    fn a_title_must_be_clean_to_be_kept() {
        assert_eq!(hygienic("  \"Man's Search for Meaning.\" "), Some("Man's Search for Meaning".into()));
        for bad in ["", "This book is about meaning", "Unknown", "harry_potter.pdf", "x".repeat(200).as_str(), "Title (SDoc)"] {
            assert_eq!(hygienic(bad), None, "{bad}");
        }
    }

    #[test]
    fn the_model_reads_the_title_page_and_a_bad_answer_falls_back_to_the_file() {
        let good = SimEngine::new(12288, |p| if p.contains("real title") { r#"{"title": "Man's Search for Meaning", "author": "Viktor E. Frankl"}"#.into() } else { String::new() });
        let t = detect(&good, "FRANKL_Viktor_Man's_Search_For_Meaning-1963_text.pdf", "MAN'S SEARCH FOR MEANING\nby Viktor E. Frankl\n...").unwrap();
        assert_eq!((t.title.as_str(), t.author.as_deref(), t.source.as_str()), ("Man's Search for Meaning", Some("Viktor E. Frankl"), "document"));

        let bad = SimEngine::new(12288, |_| r#"{"title": "This book is about a boy wizard", "author": null}"#.into());
        let t = detect(&bad, "Harry Potter and The Philosopher's Stone (SDoc).pdf", "Mr and Mrs Dursley ...").unwrap();
        assert_eq!((t.title.as_str(), t.source.as_str()), ("Harry Potter and The Philosopher's Stone", "file"));
    }

    #[test]
    fn about_ten_pages_are_shown_and_never_split_inside_a_character() {
        let text = "é".repeat(HEAD_CHARS);
        let h = head(&text);
        assert!(h.len() <= HEAD_CHARS && h.chars().all(|c| c == 'é'));
    }
}
