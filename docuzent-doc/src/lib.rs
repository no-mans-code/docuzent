//! Documents, before any question is asked of them.
//!
//! * [`extract`] - text out of `.txt .md .epub .html .pdf .docx` (plain formats natively, PDFs with poppler, scans
//!   and office files through Docling - see [`ingest`]).
//! * [`title`] - a clean title and author, from the document's own first pages and its file name.
//! * [`parts`] - the text split into parts that each fit one context window, between paragraphs.
//! * [`kvpool`] - the model's one working KV slot, and the saved KV state of every part swapped in and out of it.
//! * [`chunk`], [`hash`] - Docling JSON into page-shaped chunks; content hashes for cache keys.

pub mod chunk;
pub mod extract;
pub mod hash;
pub mod ingest;
pub mod kvpool;
pub mod parts;
pub mod title;
