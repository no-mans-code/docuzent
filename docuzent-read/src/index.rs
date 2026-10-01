//! The retrieval index for the RAG modes: chunks of the document, optional expansions of each (Mode 2), and a hybrid
//! search over them.
//!
//! **Chunks** never cross a part boundary, so every chunk points back to exactly one part - and so to that part's
//! saved KV state (Mode 3 reads those parts whole). Two ways of cutting them are kept, and measured against each
//! other (see the evaluation): [`Chunking::Plain`] - paragraphs gathered up to a size, a long paragraph split at
//! sentences - and [`Chunking::Guided`] - the model marks where scenes and topics change (see `expand::guided_cuts`).
//!
//! **Entries** are what is searched. Every chunk has one entry, its own text. With expansions (Mode 2) it has
//! more, each pointing back to the same chunk: the chunk with a sentence situating it in the document
//! ("contextual retrieval"), the people in it, its setting and objects, its facts, the questions it answers. A
//! question can then find a chunk however it is worded - and the model still answers from the chunk's own text.
//!
//! **Search** is hybrid: BM25 over words (exact names and rare terms - "Norbert", "Ridgeback") and, when there is an
//! embedder, cosine similarity over vectors (meaning - "how did he feel" finds "wept"). Each ranks chunks by their
//! best entry; the two rankings are fused by reciprocal rank (k = 60), which needs no tuning of scales.
//!
//! **Storage**: per document, in a directory of its own - `index.json` (chunks, entries, what made them) and
//! `vectors.f32` (the embeddings, raw little-endian floats). It is not size-bounded like the KV store: an index is
//! about a thousandth of the size of the document's saved KV states (measured in the evaluation).

use std::collections::HashMap;
use std::sync::OnceLock;
use std::io::{Read, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use docuzent_llm::{cosine_similarity, Embedder};
use serde::{Deserialize, Serialize};

use crate::text::terms;

/// Bumped when the index format or how it is made changes: an index made another way is rebuilt, not trusted.
pub const INDEX_VERSION: u32 = 1;
/// A chunk's size: about 300-400 tokens of prose, a paragraph or three.
pub const CHUNK_CHARS: usize = 1500;
/// Reciprocal-rank fusion constant (the usual value).
const RRF_K: f64 = 60.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Chunking {
    /// Paragraphs gathered up to [`CHUNK_CHARS`]; an over-long paragraph split at sentence ends.
    Plain,
    /// Cut where the model says a scene or topic changes, then held to [`CHUNK_CHARS`] the plain way.
    Guided,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chunk {
    pub id: usize,
    /// 0-based part.
    pub part: usize,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntryKind {
    /// The chunk's own text.
    Text,
    /// The chunk prefixed with a sentence situating it in the document (contextual retrieval).
    Context,
    /// Who is in it, and what each does.
    People,
    /// Where and when it happens; the objects, events and background.
    Setting,
    /// The facts it states, one per line.
    Facts,
    /// Questions it answers.
    Questions,
}

impl EntryKind {
    pub fn is_expansion(self) -> bool {
        self != EntryKind::Text
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub chunk: usize,
    pub kind: EntryKind,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    pub version: u32,
    pub chunking: Chunking,
    pub chunks: Vec<Chunk>,
    pub entries: Vec<Entry>,
    /// The chat model whose expansions (and guided cuts) these are; `None` for a plain index with no expansions.
    pub made_by: Option<String>,
    /// The embedding model of `vectors`; `None` when there are none (words only).
    pub embed_model: Option<String>,
    /// Chunks `made_by` could not describe even one at a time: found by their own text only, and not tried again
    /// on every load (a new model's expansions start afresh).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub undescribable: Vec<usize>,
    /// One vector per entry, when there is an embedder.
    #[serde(skip)]
    pub vectors: Vec<Vec<f32>>,
    /// The word index, built on first search (it is cheap to rebuild, so it is not saved).
    #[serde(skip)]
    bm25: OnceLock<Bm25>,
}

/// What the RAG modes search with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Search {
    /// The chunks' own text only (Mode 1).
    TextOnly,
    /// Every entry, expansions included (Mode 2, and Mode 3 when there are expansions).
    Expanded,
}

/// Paragraphs of `text`, then sentence pieces of any paragraph longer than `max`, gathered into chunks of at most
/// `max` characters (a single sentence longer than that is cut at a word).
pub fn plain_chunks(text: &str, max: usize) -> Vec<String> {
    let mut pieces: Vec<String> = Vec::new();
    for para in text.split("\n\n").map(str::trim).filter(|p| !p.is_empty()) {
        if para.chars().count() <= max {
            pieces.push(para.to_string());
        } else {
            let mut sentence = String::new();
            for (i, ch) in para.char_indices() {
                sentence.push(ch);
                let end = matches!(ch, '.' | '!' | '?' | '”' | '"') && para[i + ch.len_utf8()..].starts_with(' ');
                if end || sentence.chars().count() >= max {
                    pieces.push(sentence.trim().to_string());
                    sentence.clear();
                }
            }
            if !sentence.trim().is_empty() {
                pieces.push(sentence.trim().to_string());
            }
        }
    }
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for p in pieces {
        if !cur.is_empty() && cur.chars().count() + 2 + p.chars().count() > max {
            chunks.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push_str("\n\n");
        }
        cur.push_str(&p);
    }
    if !cur.trim().is_empty() {
        chunks.push(cur);
    }
    chunks
}

/// `text` cut at the given byte offsets (each the start of a new chunk), then held to `max` the plain way.
pub fn cut_at(text: &str, starts: &[usize], max: usize) -> Vec<String> {
    let mut bounds: Vec<usize> = starts.iter().copied().filter(|s| *s > 0 && *s < text.len() && text.is_char_boundary(*s)).collect();
    bounds.sort_unstable();
    bounds.dedup();
    let mut out = Vec::new();
    let mut prev = 0;
    for b in bounds.into_iter().chain(std::iter::once(text.len())) {
        let piece = text[prev..b].trim();
        if !piece.is_empty() {
            out.extend(plain_chunks(piece, max));
        }
        prev = b;
    }
    out
}

impl Index {
    /// An index of `parts` (their texts, in order), each cut by `cut` into chunks, with one text entry per chunk.
    pub fn new(chunking: Chunking, parts_chunks: Vec<Vec<String>>) -> Self {
        let mut chunks = Vec::new();
        for (part, texts) in parts_chunks.into_iter().enumerate() {
            for text in texts {
                chunks.push(Chunk { id: chunks.len(), part, text });
            }
        }
        let entries = chunks.iter().map(|c| Entry { chunk: c.id, kind: EntryKind::Text, text: c.text.clone() }).collect();
        Self { version: INDEX_VERSION, chunking, chunks, entries, made_by: None, embed_model: None, undescribable: Vec::new(), vectors: Vec::new(), bm25: OnceLock::new() }
    }

    /// Plain chunks of every part.
    pub fn plain(part_texts: &[&str]) -> Self {
        Self::new(Chunking::Plain, part_texts.iter().map(|t| plain_chunks(t, CHUNK_CHARS)).collect())
    }

    pub fn has_expansions(&self) -> bool {
        self.entries.iter().any(|e| e.kind.is_expansion())
    }

    /// Adds expansion entries (Mode 2), made by `model`.
    /// The chunks no expansion describes (in an index with expansions: the ones whose description failed).
    pub fn unexpanded(&self) -> Vec<usize> {
        let described: std::collections::HashSet<usize> = self.entries.iter().filter(|e| e.kind.is_expansion()).map(|e| e.chunk).collect();
        (0..self.chunks.len()).filter(|c| !described.contains(c) && !self.undescribable.contains(c)).collect()
    }

    pub fn add_expansions(&mut self, entries: Vec<Entry>, model: &str) {
        self.entries.retain(|e| !e.kind.is_expansion());
        self.entries.extend(entries.into_iter().filter(|e| e.kind.is_expansion() && e.chunk < self.chunks.len() && !e.text.trim().is_empty()));
        self.entries.sort_by_key(|e| (e.chunk, e.kind != EntryKind::Text));
        self.made_by = Some(model.to_string());
        self.vectors.clear();
        self.embed_model = None;
        self.bm25 = OnceLock::new();
    }

    /// Embeds every entry (in batches). Without this the index searches words only.
    pub fn embed(&mut self, embedder: &dyn Embedder, on_progress: &mut dyn FnMut(usize, usize)) -> Result<()> {
        let texts: Vec<String> = self.entries.iter().map(|e| e.text.clone()).collect();
        let mut vectors = Vec::with_capacity(texts.len());
        for (n, batch) in texts.chunks(64).enumerate() {
            on_progress((n * 64).min(texts.len()), texts.len());
            vectors.extend(embedder.embed_documents(batch)?);
        }
        on_progress(texts.len(), texts.len());
        anyhow::ensure!(vectors.len() == self.entries.len(), "the embedder returned {} vectors for {} entries", vectors.len(), self.entries.len());
        self.vectors = vectors;
        self.embed_model = Some(embedder.id());
        Ok(())
    }

    fn bm25(&self) -> &Bm25 {
        self.bm25.get_or_init(|| Bm25::new(&self.entries.iter().map(|e| e.text.as_str()).collect::<Vec<_>>()))
    }

    /// The best `k` chunks for `query`, best first, with their fused scores. `only` limits the search to chunks of
    /// those parts (0-based).
    pub fn search(&self, query: &str, k: usize, how: Search, embedder: Option<&dyn Embedder>, only: Option<&[usize]>) -> Result<Vec<(usize, f64)>> {
        let allowed = |e: &Entry, chunks: &[Chunk]| (how == Search::Expanded || !e.kind.is_expansion()) && only.is_none_or(|o| o.contains(&chunks[e.chunk].part));
        let (chunks, entries) = (&self.chunks, &self.entries);
        // words
        let word_scores = self.bm25().scores(query);
        let mut by_words: HashMap<usize, f64> = HashMap::new();
        for (i, s) in word_scores.into_iter().enumerate() {
            if s > 0.0 && allowed(&entries[i], chunks) {
                let best = by_words.entry(entries[i].chunk).or_insert(0.0);
                *best = best.max(s);
            }
        }
        // meaning
        let mut by_meaning: HashMap<usize, f64> = HashMap::new();
        if let Some(e) = embedder.filter(|_| !self.vectors.is_empty()) {
            if self.embed_model.as_deref() != Some(e.id().as_str()) {
                bail!("the index was embedded with `{}` but is searched with `{}` - rebuild it", self.embed_model.as_deref().unwrap_or("?"), e.id());
            }
            let q = e.embed_query(query)?;
            for (i, v) in self.vectors.iter().enumerate() {
                if allowed(&entries[i], chunks) {
                    let s = cosine_similarity(&q, v) as f64;
                    let best = by_meaning.entry(entries[i].chunk).or_insert(f64::MIN);
                    *best = best.max(s);
                }
            }
        }
        Ok(fuse(&[by_words, by_meaning], k))
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).with_context(|| format!("failed to create {}", dir.display()))?;
        std::fs::write(dir.join("index.json"), serde_json::to_vec(self)?)?;
        let mut f = std::io::BufWriter::new(std::fs::File::create(dir.join("vectors.f32"))?);
        let dims = self.vectors.first().map(Vec::len).unwrap_or(0);
        f.write_all(&(self.vectors.len() as u32).to_le_bytes())?;
        f.write_all(&(dims as u32).to_le_bytes())?;
        for v in &self.vectors {
            anyhow::ensure!(v.len() == dims, "vectors of different lengths");
            for x in v {
                f.write_all(&x.to_le_bytes())?;
            }
        }
        f.flush()?;
        Ok(())
    }

    /// The index saved in `dir`, if there is one of this [`INDEX_VERSION`].
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let Ok(raw) = std::fs::read(dir.join("index.json")) else { return Ok(None) };
        let mut index: Index = serde_json::from_slice(&raw).context("the saved index is unreadable")?;
        if index.version != INDEX_VERSION {
            return Ok(None);
        }
        if let Ok(mut f) = std::fs::File::open(dir.join("vectors.f32")) {
            let mut head = [0u8; 8];
            f.read_exact(&mut head)?;
            let n = u32::from_le_bytes(head[..4].try_into()?) as usize;
            let dims = u32::from_le_bytes(head[4..].try_into()?) as usize;
            let mut buf = vec![0u8; n * dims * 4];
            f.read_exact(&mut buf)?;
            index.vectors = buf.chunks_exact(dims.max(1) * 4).take(n).map(|v| v.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()).collect();
            if index.vectors.len() != index.entries.len() {
                index.vectors.clear();
                index.embed_model = None;
            }
        }
        Ok(Some(index))
    }

    /// Bytes the saved index takes on disk.
    pub fn disk_bytes(dir: &Path) -> u64 {
        ["index.json", "vectors.f32"].iter().filter_map(|f| std::fs::metadata(dir.join(f)).ok()).map(|m| m.len()).sum()
    }
}

/// Reciprocal-rank fusion of several chunk rankings (each a chunk -> score map, higher better).
fn fuse(rankings: &[HashMap<usize, f64>], k: usize) -> Vec<(usize, f64)> {
    let mut fused: HashMap<usize, f64> = HashMap::new();
    for r in rankings {
        let mut order: Vec<(usize, f64)> = r.iter().map(|(c, s)| (*c, *s)).collect();
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
        for (rank, (c, _)) in order.into_iter().enumerate() {
            *fused.entry(c).or_insert(0.0) += 1.0 / (RRF_K + rank as f64 + 1.0);
        }
    }
    let mut out: Vec<(usize, f64)> = fused.into_iter().collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    out.truncate(k);
    out
}

/// Okapi BM25 over the entries (k1 = 1.2, b = 0.75).
#[derive(Debug, Clone)]
struct Bm25 {
    docs: Vec<HashMap<String, f64>>,
    lens: Vec<f64>,
    avg: f64,
    df: HashMap<String, f64>,
}

impl Bm25 {
    fn new(texts: &[&str]) -> Self {
        let mut docs = Vec::with_capacity(texts.len());
        let mut lens = Vec::with_capacity(texts.len());
        let mut df: HashMap<String, f64> = HashMap::new();
        for t in texts {
            let ts = terms(t);
            lens.push(ts.len() as f64);
            let mut tf: HashMap<String, f64> = HashMap::new();
            for w in ts {
                *tf.entry(w).or_insert(0.0) += 1.0;
            }
            for w in tf.keys() {
                *df.entry(w.clone()).or_insert(0.0) += 1.0;
            }
            docs.push(tf);
        }
        let avg = if lens.is_empty() { 0.0 } else { lens.iter().sum::<f64>() / lens.len() as f64 };
        Self { docs, lens, avg, df }
    }

    fn scores(&self, query: &str) -> Vec<f64> {
        let (k1, b) = (1.2, 0.75);
        let n = self.docs.len() as f64;
        let mut q = terms(query);
        q.sort();
        q.dedup();
        self.docs
            .iter()
            .zip(&self.lens)
            .map(|(tf, len)| {
                q.iter()
                    .filter_map(|w| {
                        let f = *tf.get(w)?;
                        let df = self.df.get(w).copied().unwrap_or(0.0);
                        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                        Some(idf * f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * len / self.avg.max(1.0))))
                    })
                    .sum()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use docuzent_llm::SimEmbedder;

    fn parts() -> Vec<&'static str> {
        vec![
            "Harry went to the zoo with the Dursleys.\n\nA snake winked at him through the glass.",
            "At the final feast Dumbledore awarded ten points to Neville Longbottom.\n\nGryffindor won the House Cup.",
            "Hagrid won a dragon egg in a card game. He named the dragon Norbert; it was a Norwegian Ridgeback.",
        ]
    }

    #[test]
    fn chunks_never_cross_a_part_and_stay_within_their_size() {
        let text = "One sentence here. ".repeat(200);
        let chunks = plain_chunks(&text, 300);
        assert!(chunks.len() > 5 && chunks.iter().all(|c| c.chars().count() <= 300), "{:?}", chunks.iter().map(|c| c.len()).collect::<Vec<_>>());
        let idx = Index::plain(&parts());
        assert!(idx.chunks.iter().all(|c| parts()[c.part].contains(c.text.lines().next().unwrap())));
        assert_eq!(idx.chunks.iter().map(|c| c.part).collect::<Vec<_>>(), vec![0, 1, 2], "small parts: one chunk each");
    }

    #[test]
    fn cutting_at_given_points_keeps_every_word_in_order() {
        let t = "Scene one is here. It goes on.\n\nScene two starts now. More of it.";
        let cuts = cut_at(t, &[t.find("Scene two").unwrap()], 1000);
        assert_eq!(cuts, vec!["Scene one is here. It goes on.", "Scene two starts now. More of it."]);
    }

    #[test]
    fn words_find_rare_names_and_meaning_finds_the_rest_and_both_are_fused() {
        let mut idx = Index::plain(&parts());
        let hits = idx.search("What breed was Norbert?", 3, Search::TextOnly, None, None).unwrap();
        assert_eq!(hits[0].0, 2, "a rare name, found by its words");
        let e = SimEmbedder::default();
        idx.embed(&e, &mut |_, _| {}).unwrap();
        let hits = idx.search("who got ten points at the feast", 3, Search::TextOnly, Some(&e), None).unwrap();
        assert_eq!(hits[0].0, 1);
        let only = idx.search("who got ten points at the feast", 3, Search::TextOnly, Some(&e), Some(&[0, 2])).unwrap();
        assert!(only.iter().all(|(c, _)| *c != 1), "only the allowed parts are searched");
    }

    #[test]
    fn an_expansion_finds_a_chunk_its_own_words_do_not() {
        let mut idx = Index::plain(&parts());
        assert!(idx.search("reptile at the menagerie", 3, Search::TextOnly, None, None).unwrap().iter().all(|(c, _)| *c != 0), "the chunk never says 'reptile' or 'menagerie'");
        idx.add_expansions(vec![Entry { chunk: 0, kind: EntryKind::Setting, text: "the reptile house at a zoo (a menagerie of animals)".into() }], "model-a");
        let hits = idx.search("reptile at the menagerie", 3, Search::Expanded, None, None).unwrap();
        assert_eq!(hits[0].0, 0, "found through what it was expanded into");
        assert!(idx.search("reptile at the menagerie", 3, Search::TextOnly, None, None).unwrap().iter().all(|(c, _)| *c != 0), "Mode 1 ignores expansions");
        assert_eq!(idx.made_by.as_deref(), Some("model-a"));
    }

    #[test]
    fn an_index_is_saved_and_loaded_whole_and_a_mismatched_embedder_is_refused() {
        let dir = std::env::temp_dir().join(format!("docuzent-index-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut idx = Index::plain(&parts());
        let e = SimEmbedder::default();
        idx.embed(&e, &mut |_, _| {}).unwrap();
        idx.save(&dir).unwrap();
        let back = Index::load(&dir).unwrap().unwrap();
        assert_eq!((back.chunks.len(), back.entries.len(), back.vectors.len(), back.embed_model.clone()), (3, 3, 3, Some("sim-64".into())));
        assert_eq!(back.vectors, idx.vectors);
        assert!(Index::disk_bytes(&dir) > 0);
        let other = SimEmbedder { dims: 32 };
        assert!(back.search("zoo", 2, Search::TextOnly, Some(&other), None).unwrap_err().to_string().contains("rebuild"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reciprocal_rank_fusion_rewards_agreement() {
        let a: HashMap<usize, f64> = [(1, 9.0), (2, 5.0), (3, 1.0)].into();
        let b: HashMap<usize, f64> = [(2, 0.9), (3, 0.8), (1, 0.1)].into();
        let f = fuse(&[a, b], 3);
        assert_eq!(f[0].0, 2, "second in one, first in the other beats first in one, last in the other: {f:?}");
    }
}
