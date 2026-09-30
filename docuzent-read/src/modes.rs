//! The four ways of reading a document for a question.
//!
//! | mode | learning | a question |
//! |---|---|---|
//! | [`Mode::Rag`] | cut into chunks, indexed (words + vectors) | the best chunks, as they are |
//! | [`Mode::RagExpanded`] | ... and every chunk described by the model (context, people, setting, facts, questions) | the best chunks, found through any of those |
//! | [`Mode::RagKv`] | an index (expanded if it has been) **and** every part's KV state saved | the index points at the parts; those parts are restored and read closely, whole |
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
            Mode::RagKv => "RAG + saved parts",
            Mode::Kv => "saved parts (read every part)",
        }
    }

    pub fn needs_index(self) -> bool {
        self != Mode::Kv
    }

    pub fn needs_expansions(self) -> bool {
        self == Mode::RagExpanded
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
}

/// Searches every document's index on its own for its best `k` chunks, then takes them in turn - the best of each,
/// then the second best of each, and so on - up to `k` in all.
pub fn search_each(shelves: &[Shelf], query: &str, k: usize, mode: Mode, embedder: Option<&dyn Embedder>, only: Option<&[usize]>) -> Result<Vec<Found>> {
    let mut per: Vec<Vec<Found>> = Vec::with_capacity(shelves.len());
    for shelf in shelves {
        let how = if mode == Mode::Rag || !shelf.index.has_expansions() { Search::TextOnly } else { Search::Expanded };
        let local: Option<Vec<usize>> = only.map(|o| o.iter().filter(|p| **p >= shelf.first_part).map(|p| p - shelf.first_part).collect());
        if local.as_ref().is_some_and(|l| l.is_empty()) {
            per.push(Vec::new());
            continue;
        }
        let hits = shelf.index.search(query, k, how, embedder, local.as_deref())?;
        per.push(hits.into_iter().map(|(c, _)| &shelf.index.chunks[c]).map(|c| Found { part: c.part + shelf.first_part, text: c.text.clone() }).collect());
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
    match mode {
        Mode::Rag | Mode::RagExpanded => {
            on_stage(if indexes.len() > 1 { "Searching each book's index" } else { "Searching the book's index" });
            let found = search_each(&indexes, query, RAG_CHUNKS, mode, embedder, opts.only)?;
            let mut out = Reading { retrieved: found.len(), passages: chunk_passages(corpus, &found), ..Default::default() };
            out.ms = started.elapsed().as_secs_f64() * 1000.0;
            Ok(out)
        }
        Mode::RagKv => {
            on_stage(if indexes.len() > 1 { "Searching each book's index for the parts to read" } else { "Searching the book's index for the parts to read" });
            let found = search_each(&indexes, query, RAG_KV_CHUNKS, mode, embedder, opts.only)?;
            // the parts the best chunks are in, best first, then read in document order
            let mut parts: Vec<usize> = Vec::new();
            for f in &found {
                if !parts.contains(&f.part) && parts.len() < RAG_KV_PARTS {
                    parts.push(f.part);
                }
            }
            parts.sort_unstable();
            let mut out = Reading { retrieved: found.len(), ..Default::default() };
            read_parts(pool, corpus, query, &parts, &[], opts, on_stage, &mut out)?;
            if out.passages.is_empty() {
                // the parts, read whole, said nothing: the chunks that pointed at them are still the best evidence there is
                out.passages = chunk_passages(corpus, &found[..found.len().min(RAG_CHUNKS)]);
            }
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
}
