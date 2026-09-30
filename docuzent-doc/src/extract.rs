//! Turns an uploaded book - in whatever format - into one clean plain-text
//! string plus whatever title/author the file itself declares.
//!
//! Native (no external tools): plain text / Markdown (with Project
//! Gutenberg licence boilerplate stripped), EPUB, and HTML. Everything
//! else (PDF, DOCX, scans, ...) goes through Docling exactly as it does in
//! docuzent - see [`crate::ingest`].

use std::io::Read;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::chunk;
use crate::ingest::{self, IngestOptions};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExtractedBook {
    pub title: Option<String>,
    pub author: Option<String>,
    pub text: String,
}

/// Extracts `path` into plain text. `docling_bin` is only consulted for
/// formats that need Docling.
pub fn extract_book(path: &Path, docling_bin: Option<&str>) -> Result<ExtractedBook> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    let mut book = match ext.as_str() {
        "txt" | "text" | "md" | "markdown" | "" => extract_plain(&read_text_lossy(path)?),
        "html" | "htm" | "xhtml" => ExtractedBook { title: html_title(&read_text_lossy(path)?), author: None, text: html_to_text(&read_text_lossy(path)?) },
        "epub" => extract_epub(path)?,
        "pdf" => extract_pdf(path, docling_bin)?,
        _ => extract_with_docling(path, docling_bin)?,
    };
    book.text = normalize_whitespace(&book.text);
    if book.text.trim().is_empty() {
        bail!("no readable text found in `{}`", path.display());
    }
    if book.title.as_deref().map(str::trim).unwrap_or("").is_empty() {
        book.title = Some(title_from_filename(path));
    }
    Ok(book)
}

fn read_text_lossy(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(&bytes);
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

/// The file name, tidied (see [`crate::title::clean_file_title`]): the title until the document's own first pages
/// are read (`crate::title::detect`).
fn title_from_filename(path: &Path) -> String {
    crate::title::clean_file_title(&path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default())
}

// ---- plain text -----------------------------------------------------------

/// Plain text, with a Project Gutenberg wrapper (licence header/footer)
/// removed and its `Title:`/`Author:` lines read - those pages are a big
/// share of freely available books, and their licence text would otherwise
/// end up in the "book" the persona is derived from.
pub fn extract_plain(raw: &str) -> ExtractedBook {
    let raw = raw.replace("\r\n", "\n");
    let mut title = None;
    let mut author = None;
    for line in raw.lines().take(60) {
        let l = line.trim();
        if title.is_none() {
            if let Some(v) = l.strip_prefix("Title:") {
                title = Some(v.trim().to_string());
            }
        }
        if author.is_none() {
            if let Some(v) = l.strip_prefix("Author:") {
                author = Some(v.trim().to_string());
            }
        }
    }
    if author.is_none() {
        // Many classics have no single author line, only a translator.
        author = raw.lines().take(60).find_map(|l| l.trim().strip_prefix("Translator:").map(|v| format!("{} (translator)", v.trim()))).filter(|a| !a.starts_with(" ("));
    }
    if title.is_none() {
        // "The Project Gutenberg eBook of X" is the other place it appears.
        if let Some(line) = raw.lines().take(5).find(|l| l.contains("Project Gutenberg eBook of")) {
            if let Some(idx) = line.find("eBook of") {
                title = Some(line[idx + "eBook of".len()..].trim().to_string());
            }
        }
    }

    let start_marker = raw.find("*** START OF");
    let end_marker = raw.find("*** END OF");
    let body = match (start_marker, end_marker) {
        (Some(s), Some(e)) if e > s => {
            let after_start = raw[s..].find('\n').map(|n| s + n + 1).unwrap_or(s);
            &raw[after_start..e]
        }
        (Some(s), None) => {
            let after_start = raw[s..].find('\n').map(|n| s + n + 1).unwrap_or(s);
            &raw[after_start..]
        }
        _ => raw.as_str(),
    };
    ExtractedBook { title, author, text: body.to_string() }
}

/// Collapses runs of blank lines and trailing spaces without touching
/// paragraph structure - the text is later chunked on whitespace, and long
/// runs of blanks just waste context.
pub fn normalize_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank_run = 0;
    for line in text.replace("\r\n", "\n").lines() {
        let line = line.trim_end();
        if line.trim().is_empty() {
            blank_run += 1;
            if blank_run <= 1 {
                out.push('\n');
            }
        } else {
            blank_run = 0;
            out.push_str(line);
            out.push('\n');
        }
    }
    out.trim().to_string()
}

// ---- HTML -----------------------------------------------------------------

fn html_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let s = lower.find("<title")?;
    let s = s + lower[s..].find('>')? + 1;
    let e = s + lower[s..].find("</title")?;
    let t = decode_entities(html[s..e].trim());
    (!t.is_empty()).then_some(t)
}

/// Strips markup, keeping paragraph/heading/line structure as newlines.
pub fn html_to_text(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let bytes = html.as_bytes();
    let lower = html.to_ascii_lowercase();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Skip whole <script>/<style>/<head> bodies.
            let mut skipped = false;
            for tag in ["script", "style", "head"] {
                if lower[i..].starts_with(&format!("<{tag}")) {
                    let close = format!("</{tag}");
                    if let Some(end) = lower[i..].find(&close) {
                        let after = i + end;
                        i = lower[after..].find('>').map(|n| after + n + 1).unwrap_or(bytes.len());
                        skipped = true;
                    }
                    break;
                }
            }
            if skipped {
                continue;
            }
            let Some(rel_end) = html[i..].find('>') else { break };
            let tag = lower[i + 1..i + rel_end].trim_start_matches('/').split(|c: char| c.is_whitespace() || c == '/').next().unwrap_or("").to_string();
            if matches!(tag.as_str(), "p" | "div" | "br" | "li" | "tr" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "blockquote" | "section" | "article" | "pre") {
                out.push('\n');
                if matches!(tag.as_str(), "p" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" | "blockquote") {
                    out.push('\n');
                }
            }
            i += rel_end + 1;
        } else {
            let next = html[i..].find('<').map(|n| i + n).unwrap_or(bytes.len());
            out.push_str(&html[i..next]);
            i = next;
        }
    }
    let decoded = decode_entities(&out);
    // Source markup wraps lines arbitrarily; only our own inserted newlines
    // are meaningful, so squeeze other whitespace runs on each line.
    decoded.lines().map(|l| l.split_whitespace().collect::<Vec<_>>().join(" ")).collect::<Vec<_>>().join("\n")
}

fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let tail = &rest[amp..];
        match tail.find(';').filter(|&n| n <= 10) {
            Some(semi) => {
                let ent = &tail[1..semi];
                let replaced = match ent {
                    "amp" => Some('&'),
                    "lt" => Some('<'),
                    "gt" => Some('>'),
                    "quot" => Some('"'),
                    "apos" => Some('\''),
                    "nbsp" => Some(' '),
                    "mdash" => Some('—'),
                    "ndash" => Some('–'),
                    "hellip" => Some('…'),
                    "lsquo" => Some('‘'),
                    "rsquo" => Some('’'),
                    "ldquo" => Some('“'),
                    "rdquo" => Some('”'),
                    _ => ent
                        .strip_prefix("#x")
                        .or_else(|| ent.strip_prefix("#X"))
                        .and_then(|h| u32::from_str_radix(h, 16).ok())
                        .or_else(|| ent.strip_prefix('#').and_then(|d| d.parse().ok()))
                        .and_then(char::from_u32),
                };
                match replaced {
                    Some(c) => {
                        out.push(c);
                        rest = &tail[semi + 1..];
                    }
                    None => {
                        out.push('&');
                        rest = &tail[1..];
                    }
                }
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

// ---- EPUB -----------------------------------------------------------------

fn zip_read(archive: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Result<String> {
    let mut f = archive.by_name(name).with_context(|| format!("epub is missing `{name}`"))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Value of `attr="..."` (either quote style) inside one tag's text.
fn attr(tag: &str, name: &str) -> Option<String> {
    let lower = tag.to_ascii_lowercase();
    let mut from = 0;
    while let Some(p) = lower[from..].find(name) {
        let idx = from + p;
        let before_ok = idx == 0 || !lower.as_bytes()[idx - 1].is_ascii_alphanumeric() && lower.as_bytes()[idx - 1] != b'-';
        let after = &tag[idx + name.len()..];
        let after_trim = after.trim_start();
        if before_ok && after_trim.starts_with('=') {
            let val = after_trim[1..].trim_start();
            let quote = val.chars().next()?;
            if quote == '"' || quote == '\'' {
                let end = val[1..].find(quote)?;
                return Some(val[1..1 + end].to_string());
            }
        }
        from = idx + name.len();
    }
    None
}

/// All `<name ...>` opening tags' text (between `<` and `>`), any namespace prefix.
fn tags<'a>(xml: &'a str, name: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let lower = xml.to_ascii_lowercase();
    let mut from = 0;
    while let Some(p) = lower[from..].find('<') {
        let start = from + p;
        let Some(end_rel) = lower[start..].find('>') else { break };
        let inner = &xml[start + 1..start + end_rel];
        let tag_name = inner.split(|c: char| c.is_whitespace() || c == '/').next().unwrap_or("");
        let bare = tag_name.rsplit(':').next().unwrap_or(tag_name);
        if bare.eq_ignore_ascii_case(name) {
            out.push(inner);
        }
        from = start + end_rel + 1;
    }
    out
}

/// Text content of the first `<name>...</name>` (any namespace prefix).
fn element_text(xml: &str, name: &str) -> Option<String> {
    let lower = xml.to_ascii_lowercase();
    let mut from = 0;
    while let Some(p) = lower[from..].find('<') {
        let start = from + p;
        let end_rel = lower[start..].find('>')?;
        let inner = &lower[start + 1..start + end_rel];
        let tag_name = inner.split(|c: char| c.is_whitespace() || c == '/').next().unwrap_or("");
        let bare = tag_name.rsplit(':').next().unwrap_or(tag_name);
        if bare == name && !inner.starts_with('/') {
            let content_start = start + end_rel + 1;
            let close = lower[content_start..].find(&format!("</{tag_name}"))?;
            let t = decode_entities(xml[content_start..content_start + close].trim());
            return (!t.is_empty()).then_some(t);
        }
        from = start + end_rel + 1;
    }
    None
}

fn extract_epub(path: &Path) -> Result<ExtractedBook> {
    let file = std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file).with_context(|| format!("`{}` is not a valid epub (zip) file", path.display()))?;

    let container = zip_read(&mut archive, "META-INF/container.xml")?;
    let opf_path = tags(&container, "rootfile").into_iter().find_map(|t| attr(t, "full-path")).context("epub container.xml names no package file")?;
    let opf = zip_read(&mut archive, &opf_path)?;
    let opf_dir = opf_path.rsplit_once('/').map(|(d, _)| format!("{d}/")).unwrap_or_default();

    // manifest id -> href, then reading order from the spine.
    let manifest: Vec<(String, String)> = tags(&opf, "item").into_iter().filter_map(|t| Some((attr(t, "id")?, attr(t, "href")?))).collect();
    let spine: Vec<String> = tags(&opf, "itemref").into_iter().filter_map(|t| attr(t, "idref")).collect();

    let mut text = String::new();
    for idref in &spine {
        let Some((_, href)) = manifest.iter().find(|(id, _)| id == idref) else { continue };
        let href = href.split('#').next().unwrap_or(href);
        let lower = href.to_ascii_lowercase();
        if !(lower.ends_with(".xhtml") || lower.ends_with(".html") || lower.ends_with(".htm") || lower.ends_with(".xml")) {
            continue;
        }
        let full = percent_decode(&format!("{opf_dir}{href}"));
        let Ok(html) = zip_read(&mut archive, &full) else { continue };
        let part = html_to_text(&html);
        if !part.trim().is_empty() {
            text.push_str(part.trim());
            text.push_str("\n\n");
        }
    }

    Ok(ExtractedBook { title: element_text(&opf, "title"), author: element_text(&opf, "creator"), text })
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

// ---- PDF ------------------------------------------------------------------

/// Fewest letters and digits a PDF's text layer must hold to count as text
/// (a scan has none, or only stray marks).
const MIN_PDF_CHARS: usize = 400;

/// PDFs with a text layer are read with poppler's `pdftotext` - small,
/// quick and installed with the app. Only a scan (no text layer) needs
/// Docling's OCR, which is heavy and optional.
fn extract_pdf(path: &Path, docling_bin: Option<&str>) -> Result<ExtractedBook> {
    let bin = std::env::var("PDFTOTEXT_BIN").unwrap_or_else(|_| "pdftotext".to_string());
    match std::process::Command::new(&bin).args(["-enc", "UTF-8", "-nopgbrk"]).arg(path).arg("-").output() {
        Ok(out) if out.status.success() => {
            let text = unwrap_pdf_lines(&String::from_utf8_lossy(&out.stdout));
            if text.chars().filter(|c| c.is_alphanumeric()).count() >= MIN_PDF_CHARS {
                return Ok(ExtractedBook { title: None, author: None, text });
            }
            // A scan: only OCR can read it.
            extract_with_docling(path, docling_bin).with_context(|| format!("`{}` has no text layer (it looks like a scanned PDF); reading it needs OCR, which needs Docling. Use a PDF with selectable text, or convert it to .txt/.epub", path.display()))
        }
        Ok(out) => {
            let why = String::from_utf8_lossy(&out.stderr).trim().to_string();
            bail!("`{}` could not be read as a PDF: {}", path.display(), if why.is_empty() { "pdftotext failed" } else { &why });
        }
        Err(_) => extract_with_docling(path, docling_bin).with_context(|| format!("`{}` is a PDF, and neither pdftotext (poppler) nor Docling is installed to read it", path.display())),
    }
}

/// Text from a PDF has a line break at the end of every printed line. Joins
/// each paragraph back into one line (a blank line still ends a paragraph),
/// mends words hyphenated across lines, and drops bare page numbers.
pub fn unwrap_pdf_lines(raw: &str) -> String {
    let mut paras: Vec<String> = Vec::new();
    let mut cur = String::new();
    for line in raw.replace('\r', "").replace('\u{c}', "\n\n").lines() {
        let l = line.trim();
        if l.is_empty() {
            if !cur.is_empty() {
                paras.push(std::mem::take(&mut cur));
            }
            continue;
        }
        if l.len() <= 4 && l.chars().all(|c| c.is_ascii_digit()) {
            continue; // a page number
        }
        let broken_word = cur.ends_with('-') && cur.chars().rev().nth(1).is_some_and(|c| c.is_alphabetic()) && l.chars().next().is_some_and(|c| c.is_lowercase());
        if broken_word {
            cur.pop();
        } else if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(l);
    }
    if !cur.is_empty() {
        paras.push(cur);
    }
    paras.join("\n\n")
}

// ---- Docling --------------------------------------------------------------

fn extract_with_docling(path: &Path, docling_bin: Option<&str>) -> Result<ExtractedBook> {
    let out_dir = std::env::temp_dir().join(format!("docuzent-docling-{}-{}", std::process::id(), crate::hash::hash_bytes(path.to_string_lossy().as_bytes()).chars().take(10).collect::<String>()));
    let report = ingest::run(&IngestOptions { source: path.to_path_buf(), output: out_dir.clone(), to: "json".to_string(), device: "auto".to_string(), docling_bin: docling_bin.map(str::to_string) })
        .with_context(|| format!("`{}` needs Docling to be read (PDF/DOCX/scans) - install it, or convert the book to .txt/.epub", path.display()))?;
    let stem = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
    let json_name = report.produced_files.iter().find(|f| f.starts_with(&stem)).context("docling produced no output for this file")?;
    let chunks = chunk::chunk_docling_json(&out_dir.join(json_name), chunk::DEFAULT_MAX_CHARS)?;
    let mut text = String::new();
    for c in chunks {
        text.push_str(&c.text);
        text.push_str("\n\n");
    }
    let _ = std::fs::remove_dir_all(&out_dir);
    Ok(ExtractedBook { title: None, author: None, text })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdf_lines_are_joined_into_paragraphs_and_hyphenated_words_mended() {
        let raw = "Chapter 1\n\nThe soul is never\nborn and never dies; it is unbe-\ngotten and eternal.\n\n12\n\nNext paragraph\nstarts here.\n";
        assert_eq!(unwrap_pdf_lines(raw), "Chapter 1\n\nThe soul is never born and never dies; it is unbegotten and eternal.\n\nNext paragraph starts here.");
        assert_eq!(unwrap_pdf_lines("a well-\nKnown thing"), "a well- Known thing", "a capital after a hyphen is a real hyphen, not a break");
    }

    #[test]
    fn a_file_that_is_not_a_pdf_says_so() {
        let p = std::env::temp_dir().join(format!("not-a-pdf-{}.pdf", std::process::id()));
        std::fs::write(&p, "this is not a pdf").unwrap();
        let e = format!("{:#}", extract_book(&p, None).unwrap_err());
        let _ = std::fs::remove_file(&p);
        assert!(e.to_lowercase().contains("pdf"), "{e}");
    }

    #[test]
    fn gutenberg_wrapper_is_stripped_and_metadata_read() {
        let raw = "The Project Gutenberg eBook of Test\n\nTitle: The Test Book\nAuthor: Jane Roe\n\n*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nReal text here.\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\nLicence junk";
        let b = extract_plain(raw);
        assert_eq!(b.title.as_deref(), Some("The Test Book"));
        assert_eq!(b.author.as_deref(), Some("Jane Roe"));
        assert_eq!(b.text.trim(), "Real text here.");
    }

    #[test]
    fn translator_stands_in_when_there_is_no_author_line() {
        let b = extract_plain("Title: The Song

Translator: Sir Edwin Arnold

*** START OF X ***
body text
*** END OF X ***");
        assert_eq!(b.author.as_deref(), Some("Sir Edwin Arnold (translator)"));
    }

    #[test]
    fn plain_text_without_markers_is_kept_whole() {
        assert_eq!(extract_plain("just words").text, "just words");
    }

    #[test]
    fn html_becomes_text_with_paragraphs_and_entities() {
        let t = html_to_text("<html><head><title>x</title><style>p{}</style></head><body><h1>Chapter&nbsp;1</h1><p>It was a <b>dark</b> &amp; stormy night.</p><script>var a=1<2;</script><p>End &#8212; &#x2014;.</p></body></html>");
        assert!(t.contains("Chapter 1"), "{t}");
        assert!(t.contains("It was a dark & stormy night."), "{t}");
        assert!(t.contains("End — —."), "{t}");
        assert!(!t.contains("var a"), "{t}");
        assert!(!t.contains("p{}"), "{t}");
    }

    #[test]
    fn whitespace_normalization_keeps_single_paragraph_breaks() {
        assert_eq!(normalize_whitespace("a  \n\n\n\nb\n"), "a\n\nb");
    }

    #[test]
    fn filename_becomes_a_readable_title() {
        assert_eq!(title_from_filename(Path::new("/x/the_song-celestial.txt")), "The Song Celestial");
    }

    #[test]
    fn attr_reads_either_quote_style_and_ignores_prefixed_names() {
        assert_eq!(attr(r#"item id="a" href='ch1.xhtml'"#, "href").as_deref(), Some("ch1.xhtml"));
        assert_eq!(attr(r#"rootfile full-path="OEBPS/c.opf""#, "full-path").as_deref(), Some("OEBPS/c.opf"));
        assert_eq!(attr(r#"a data-id="x""#, "id"), None);
    }

    #[test]
    fn epub_round_trip_reads_spine_order_title_and_author() {
        use std::io::Write;
        let path = std::env::temp_dir().join(format!("docuzent-test-{}.epub", std::process::id()));
        {
            let f = std::fs::File::create(&path).unwrap();
            let mut z = zip::ZipWriter::new(f);
            let opts = zip::write::SimpleFileOptions::default();
            z.start_file("META-INF/container.xml", opts).unwrap();
            z.write_all(br#"<container><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="x"/></rootfiles></container>"#).unwrap();
            z.start_file("OEBPS/content.opf", opts).unwrap();
            z.write_all(br#"<package><metadata><dc:title>My Epub</dc:title><dc:creator>A. Writer</dc:creator></metadata><manifest><item id="b" href="two.xhtml"/><item id="a" href="one.xhtml"/></manifest><spine><itemref idref="a"/><itemref idref="b"/></spine></package>"#).unwrap();
            z.start_file("OEBPS/one.xhtml", opts).unwrap();
            z.write_all(b"<html><body><p>First chapter.</p></body></html>").unwrap();
            z.start_file("OEBPS/two.xhtml", opts).unwrap();
            z.write_all(b"<html><body><p>Second chapter.</p></body></html>").unwrap();
            z.finish().unwrap();
        }
        let b = extract_book(&path, None).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(b.title.as_deref(), Some("My Epub"));
        assert_eq!(b.author.as_deref(), Some("A. Writer"));
        assert!(b.text.find("First chapter").unwrap() < b.text.find("Second chapter").unwrap(), "{}", b.text);
    }

    #[test]
    fn empty_books_are_rejected() {
        let path = std::env::temp_dir().join(format!("docuzent-empty-{}.txt", std::process::id()));
        std::fs::write(&path, "   \n\n").unwrap();
        let r = extract_book(&path, None);
        let _ = std::fs::remove_file(&path);
        assert!(r.is_err());
    }
}
