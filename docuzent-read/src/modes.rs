//! The four ways of reading a document for a question.
//!
//! | mode | learning | a question |
//! |---|---|---|
//! | [`Mode::Rag`] | cut into chunks, indexed (words + vectors) | the best chunks, as they are |
//! | [`Mode::RagExpanded`] | ... and every chunk described by the model (context, people, setting, facts, questions) | the best chunks, found through any of those |
//! | [`Mode::RagKv`] | the expanded index **and** every part's KV state saved | the expanded index's passages first, as they are; then the parts they point at, restored and read whole, as enrichment |
//! | [`Mode::Kv`] | every part's KV state saved | every part restored and scored; the relevant ones read closely |
//!
//! Which is best depends on the document and the question; the evaluation (`docuzent-eval`) measures each on real
//! books, and the results are in docs/READING_MODES.md.

use anyhow::Result;
use docuzent_doc::kvpool::KvPool;
use docuzent_llm::Embedder;
use serde::{Deserialize, Serialize};

use crate::corpus::Corpus;
use crate::index::{Index, Search};
use crate::reader::{read_parts, scan, Passage, ReadOptions, Reading};

/// Chunks the RAG modes hand to the answer.
pub const RAG_CHUNKS: usize = 8;
/// Characters of each neighbouring chunk handed over with a chunk a RAG search found (Modes 1 and 2): the end of the
/// one before and the start of the one after, from the same part. A chunk is cut at a size, not where the book's
/// thought ends, so an answer can begin in the chunk before the one that matched or end in the one after; and the
/// line that names who "he" is is often just before. ("Small-to-big" retrieval: search small, read bigger.)
pub const NEIGHBOUR_CHARS: usize = 600;
/// Chunks Mode 3 looks at to choose parts...
pub const RAG_KV_CHUNKS: usize = 12;
/// ... and the most parts it then reads closely.
pub const RAG_KV_PARTS: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mode {
    #[serde(rename = "rag")]
    Rag,
    #[serde(rename = "rag-expanded")]
    RagExpanded,
    #[serde(rename = "rag-kv")]
    RagKv,
    #[serde(rename = "kv")]
    Kv,
}

impl Mode {
    pub const ALL: [Mode; 4] = [Mode::Rag, Mode::RagExpanded, Mode::RagKv, Mode::Kv];

    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().replace('_', "-").as_str() {
            "rag" | "1" | "mode1" => Ok(Mode::Rag),
            "rag-expanded" | "expanded" | "2" | "mode2" => Ok(Mode::RagExpanded),
            "rag-kv" | "hybrid" | "3" | "mode3" => Ok(Mode::RagKv),
            "kv" | "lru" | "kv-only" | "4" | "mode4" => Ok(Mode::Kv),
            other => anyhow::bail!("unknown reading mode `{other}` - use rag, rag-expanded, rag-kv or kv"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Rag => "rag",
            Mode::RagExpanded => "rag-expanded",
            Mode::RagKv => "rag-kv",
            Mode::Kv => "kv",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Mode::Rag => "RAG",
            Mode::RagExpanded => "RAG, expanded",
            Mode::RagKv => "expanded RAG, enriched from saved parts",
            Mode::Kv => "saved parts (read every part)",
        }
    }

    pub fn needs_index(self) -> bool {
        self != Mode::Kv
    }

    /// Mode 2 searches the expanded index; so does Mode 3, whose passages come first and are enriched from the
    /// saved parts they point at.
    pub fn needs_expansions(self) -> bool {
        matches!(self, Mode::RagExpanded | Mode::RagKv)
    }

    /// Whether every part's KV state must be saved when the document is learned.
    pub fn needs_saved_parts(self) -> bool {
        matches!(self, Mode::RagKv | Mode::Kv)
    }
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One document's index, and where its parts start among the corpus's parts (0 for a single document; for a corpus
/// merged from several documents, each document's index keeps its own part numbers and says where they begin).
#[derive(Clone, Copy)]
pub struct Shelf<'a> {
    pub index: &'a Index,
    pub first_part: usize,
}

/// What a reading has to work with.
pub struct Sources<'a> {
    pub pool: &'a KvPool,
    pub corpus: &'a dyn Corpus,
    /// One index per document. A merged corpus (a crossover of two books) has one per book: each is searched on
    /// its own and the results interleaved by rank, so every book is represented fairly. (A single index across
    /// the documents is a later step.) Empty: no index.
    pub indexes: Vec<Shelf<'a>>,
    pub embedder: Option<&'a dyn Embedder>,
}

/// A chunk found by a search, placed in the whole corpus.
#[derive(Debug, Clone, PartialEq)]
pub struct Found {
    /// 0-based part in the corpus.
    pub part: usize,
    pub text: String,
    /// Which shelf (document index) it was found in, and its chunk id there.
    pub shelf: usize,
    pub chunk: usize,
}

/// Searches every document's index on its own for its best `k` chunks, then takes them in turn - the best of each,
/// then the second best of each, and so on - up to `k` in all.
pub fn search_each(shelves: &[Shelf], query: &str, k: usize, mode: Mode, embedder: Option<&dyn Embedder>, only: Option<&[usize]>) -> Result<Vec<Found>> {
    let mut per: Vec<Vec<Found>> = Vec::with_capacity(shelves.len());
    for (s, shelf) in shelves.iter().enumerate() {
        let how = if mode == Mode::Rag || !shelf.index.has_expansions() { Search::TextOnly } else { Search::Expanded };
        let local: Option<Vec<usize>> = only.map(|o| o.iter().filter(|p| **p >= shelf.first_part).map(|p| p - shelf.first_part).collect());
        if local.as_ref().is_some_and(|l| l.is_empty()) {
            per.push(Vec::new());
            continue;
        }
        let hits = shelf.index.search(query, k, how, embedder, local.as_deref())?;
        per.push(hits.into_iter().map(|(c, _)| &shelf.index.chunks[c]).map(|c| Found { part: c.part + shelf.first_part, text: c.text.clone(), shelf: s, chunk: c.id }).collect());
    }
    let mut out = Vec::new();
    for rank in 0.. {
        let mut any = false;
        for list in &per {
            if let Some(f) = list.get(rank) {
                any = true;
                if out.len() < k {
                    out.push(f.clone());
                }
            }
        }
        if !any || out.len() >= k {
            break;
        }
    }
    Ok(out)
}

/// `found` with the end of the chunk before and the start of the chunk after each one (from the same part, and not
/// when that neighbour was found itself - it is handed over whole).
pub fn with_neighbours(shelves: &[Shelf], found: &[Found], chars: usize) -> Vec<Found> {
    let is_found = |s: usize, c: usize| found.iter().any(|f| f.shelf == s && f.chunk == c);
    found
        .iter()
        .map(|f| {
            let chunks = &shelves[f.shelf].index.chunks;
            let me = &chunks[f.chunk];
            let near = |c: Option<usize>| c.and_then(|c| chunks.get(c)).filter(|n| n.part == me.part && !is_found(f.shelf, n.id));
            let mut text = String::new();
            if let Some(prev) = near(f.chunk.checked_sub(1)) {
                let tail: String = prev.text.chars().rev().take(chars).collect::<Vec<_>>().into_iter().rev().collect();
                text.push_str(&format!("…{}\n\n", tail.trim_start()));
            }
            text.push_str(&f.text);
            if let Some(next) = near(Some(f.chunk + 1)) {
                let head: String = next.text.chars().take(chars).collect();
                text.push_str(&format!("\n\n{}…", head.trim_end()));
            }
            Found { text, ..f.clone() }
        })
        .collect()
}

fn chunk_passages(corpus: &dyn Corpus, found: &[Found]) -> Vec<Passage> {
    let mut sorted: Vec<&Found> = found.iter().collect();
    sorted.sort_by_key(|f| f.part); // document order reads as the document does (stable: chunks of a part keep their order)
    sorted.into_iter().map(|f| Passage { part: f.part + 1, book: corpus.part_source(f.part), text: f.text.clone() }).collect()
}

/// Reads `corpus` for `query` in `mode`. A RAG mode without an index falls back to reading every part (Mode 4) and
/// says so.
pub fn read(mode: Mode, src: Sources, query: &str, opts: ReadOptions, on_stage: &dyn Fn(&str)) -> Result<Reading> {
    let started = std::time::Instant::now();
    let Sources { pool, corpus, indexes, embedder } = src;
    if !mode.needs_index() || indexes.is_empty() {
        if mode.needs_index() {
            on_stage("No index for this book yet: reading every part instead");
        }
        return scan(pool, corpus, query, opts, on_stage);
    }
    // how much to hand over, for this model's window (at 12,288 tokens: 8 chunks; 12 and 4 parts for Mode 3)
    let b = crate::Budget::for_engine(pool.engine().context_size());
    match mode {
        Mode::Rag | Mode::RagExpanded => {
            on_stage(if indexes.len() > 1 { "Searching each book's index" } else { "Searching the book's index" });
            let found = search_each(&indexes, query, b.rag_chunks, mode, embedder, opts.only)?;
            let mut out = Reading { retrieved: found.len(), passages: chunk_passages(corpus, &with_neighbours(&indexes, &found, b.neighbour_chars)), ..Default::default() };
            out.ms = started.elapsed().as_secs_f64() * 1000.0;
            Ok(out)
        }
        Mode::RagKv => {
            on_stage(if indexes.len() > 1 { "Searching each book's index for the parts to read" } else { "Searching the book's index for the parts to read" });
            let found = search_each(&indexes, query, b.rag_kv_chunks, mode, embedder, opts.only)?;
            // the parts the best chunks are in, best first, then read in document order
            let mut parts: Vec<usize> = Vec::new();
            for f in &found {
                if !parts.contains(&f.part) && parts.len() < b.rag_kv_parts {
                    parts.push(f.part);
                }
            }
            parts.sort_unstable();
            // The passages the index found come first, in the book's own words: they are precise, and nothing a model
            // writes stands between them and the answer. (A first version handed over only its readings of the whole
            // parts, and lost a verse the plain passages had - a model's reading can blur or drop a line.) The parts
            // they point at are then read whole, and what those readings add comes after, as enrichment.
            let mut out = Reading { retrieved: found.len(), passages: chunk_passages(corpus, &found[..found.len().min(b.rag_chunks)]), ..Default::default() };
            let mut read = Reading::default();
            read_parts(pool, corpus, query, &parts, &[], opts, on_stage, &mut read)?;
            out.recalled = read.recalled;
            out.read_afresh = read.read_afresh;
            out.processed_tokens = read.processed_tokens;
            out.read_closely = read.read_closely;
            out.truncated = read.truncated;
            // Labelled as what they are: a model's reading, which can blur a number or join two things that are
            // apart in the book - where it differs from a quoted passage, the passage is right.
            out.passages.extend(read.passages.into_iter().map(|p| Passage { text: format!("(Notes from reading the whole part - a reading, not the book's words; where they differ from a passage above, the passage is right.)\n{}", p.text), ..p }));
            out.ms = started.elapsed().as_secs_f64() * 1000.0;
            Ok(out)
        }
        Mode::Kv => unreachable!("handled above"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_are_named_the_way_people_say_them() {
        for (s, m) in [("rag", Mode::Rag), ("Mode2", Mode::RagExpanded), ("hybrid", Mode::RagKv), ("lru", Mode::Kv), ("rag_kv", Mode::RagKv)] {
            assert_eq!(Mode::parse(s).unwrap(), m, "{s}");
        }
        assert!(Mode::parse("magic").is_err());
        assert_eq!(serde_json::to_string(&Mode::RagKv).unwrap(), "\"rag-kv\"");
        assert!(Mode::Kv.needs_saved_parts() && !Mode::Rag.needs_saved_parts() && Mode::RagKv.needs_index());
    }

    /// A crossover: each book's index is searched on its own and the results interleaved, so a question that both
    /// books answer gets both - and part numbers are the crossover's, not each book's.
    #[test]
    fn a_merged_corpus_searches_each_books_index_and_takes_their_best_in_turn() {
        let a = Index::plain(&["Krishna speaks of duty and action.", "Arjuna drops his bow."]);
        let b = Index::plain(&["Harry learns of duty at Hogwarts.", "Hagrid has a dragon."]);
        let shelves = [Shelf { index: &a, first_part: 0 }, Shelf { index: &b, first_part: 2 }];
        let found = search_each(&shelves, "what is said of duty", 4, Mode::Rag, None, None).unwrap();
        assert_eq!(found.iter().map(|f| f.part).collect::<Vec<_>>(), vec![0, 2], "the best of each book, in turn, placed in the crossover's parts");
        let only_b = search_each(&shelves, "what is said of duty", 4, Mode::Rag, None, Some(&[2, 3])).unwrap();
        assert_eq!(only_b.iter().map(|f| f.part).collect::<Vec<_>>(), vec![2], "a restriction to one book's parts is respected");
        let two = search_each(&shelves, "duty dragon bow", 2, Mode::Rag, None, None).unwrap();
        assert_eq!(two.len(), 2, "at most k in all");
    }

    /// A found chunk comes with the end of the one before and the start of the one after - from its own part only,
    /// and not when that neighbour was found too.
    #[test]
    fn a_found_chunk_brings_the_edges_of_its_neighbours() {
        let long = (0..6).map(|i| format!("Paragraph {i} begins. {} Paragraph {i} ends.", "Words of the book go on. ".repeat(60))).collect::<Vec<_>>().join("\n\n");
        let ix = Index::plain(&[long.as_str(), "Another part."]);
        let shelves = [Shelf { index: &ix, first_part: 0 }];
        let last_of_part = ix.chunks.iter().filter(|c| c.part == 0).map(|c| c.id).max().unwrap();
        assert!(last_of_part >= 2);
        let found = |c: usize| Found { part: ix.chunks[c].part, text: ix.chunks[c].text.clone(), shelf: 0, chunk: c };
        let got = with_neighbours(&shelves, &[found(1)], 40);
        assert!(got[0].text.starts_with('…') && got[0].text.ends_with('…') && got[0].text.contains(&ix.chunks[1].text));
        let both = with_neighbours(&shelves, &[found(1), found(2)], 40);
        assert!(!both[0].text.ends_with('…') && both[1].text.starts_with(&ix.chunks[2].text[..20]), "a neighbour found itself is not repeated");
        let edge = with_neighbours(&shelves, &[found(last_of_part)], 40);
        assert!(!edge[0].text.ends_with('…'), "nothing from the next part");
    }
}
