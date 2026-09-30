//! What is read: a document already split into parts, each with a saved KV state.

/// A document as the reader sees it. thebook's books, docuzent's own documents and the evaluator each implement it.
///
/// A part's *prefix* is the exact text its KV state was saved under - the reader asks its questions after it, so a
/// restored state is reused rather than the part being read again. It must never change for a saved part.
pub trait Corpus {
    fn part_count(&self) -> usize;
    /// The part's saved-state file name (in the KV store's directory).
    fn part_file(&self, i: usize) -> &str;
    /// Who owns the saved state in the LRU store (a book, a document) - eviction prefers other owners' files.
    fn part_owner(&self, i: usize) -> &str;
    /// The exact text the part's state was saved under.
    fn part_prefix(&self, i: usize) -> String;
    /// The part's own text.
    fn part_text(&self, i: usize) -> &str;
    /// In a corpus merged from several documents, which one the part comes from; empty otherwise.
    fn part_source(&self, _i: usize) -> String {
        String::new()
    }
    fn title(&self) -> &str;
    /// A few lines on what the whole document is about (its core ideas), used to put a question that finds nothing
    /// into the document's own words. Empty when there is none.
    fn gist(&self) -> String {
        String::new()
    }
}

/// A corpus held in memory - for tests, and for callers that build one on the fly.
pub struct MemCorpus {
    pub title: String,
    pub owner: String,
    /// `(saved-state file, prefix, text)` per part.
    pub parts: Vec<(String, String, String)>,
    pub gist: String,
}

impl Corpus for MemCorpus {
    fn part_count(&self) -> usize {
        self.parts.len()
    }
    fn part_file(&self, i: usize) -> &str {
        &self.parts[i].0
    }
    fn part_owner(&self, _i: usize) -> &str {
        &self.owner
    }
    fn part_prefix(&self, i: usize) -> String {
        self.parts[i].1.clone()
    }
    fn part_text(&self, i: usize) -> &str {
        &self.parts[i].2
    }
    fn title(&self) -> &str {
        &self.title
    }
    fn gist(&self) -> String {
        self.gist.clone()
    }
}
