//! A document ready to be read, for tools that do not keep a library of their own (the CLI, the evaluator): its
//! parts, their saved KV states, and its index for each way of reading it - made once, then reused.

use std::path::Path;

use anyhow::Result;
use docuzent_doc::kvpool::KvPool;
use docuzent_doc::parts::{split_into_parts, Part, PART_TOKEN_LIMIT};
use docuzent_doc::hash;
use docuzent_kv::file_name;
use docuzent_llm::Embedder;

use crate::corpus::Corpus;
use crate::expand::{expand, guided_index};
use crate::index::{Chunking, Index};

pub struct Document {
    /// A short hash of the text: the owner of its saved states, and the name of its folder.
    pub id: String,
    pub title: String,
    pub author: Option<String>,
    /// `(saved-state file, part)`.
    pub parts: Vec<(String, Part)>,
}

impl Document {
    /// `text` split into parts that fit one context (`count` is the model's tokenizer), each named for its saved
    /// state under `model_id` - which should start with the KV store's scope, if it has one.
    pub fn from_text(text: &str, title: &str, author: Option<&str>, model_id: &str, count: &dyn Fn(&str) -> Result<usize>) -> Result<Self> {
        Self::from_text_sized(text, title, author, model_id, count, PART_TOKEN_LIMIT)
    }

    /// [`Document::from_text`] with parts of at most `part_tokens` (a model's [`crate::Budget::part_tokens`]: a
    /// small window reads a document in more, smaller parts).
    pub fn from_text_sized(text: &str, title: &str, author: Option<&str>, model_id: &str, count: &dyn Fn(&str) -> Result<usize>, part_tokens: usize) -> Result<Self> {
        let id: String = hash::hash_bytes(text.as_bytes()).chars().take(12).collect();
        let parts = split_into_parts(text, part_tokens, count)?;
        Ok(Self { parts: parts.into_iter().map(|p| (file_name(model_id, &id, &p.kv_kind()), p)).collect(), id, title: title.to_string(), author: author.map(str::to_string) })
    }

    /// Saves every part's KV state that is not saved yet; returns how many were.
    pub fn save_parts(&self, pool: &KvPool, on_progress: &mut dyn FnMut(usize, usize)) -> Result<usize> {
        let missing: Vec<usize> = (0..self.parts.len()).filter(|i| !pool.store().contains(&self.parts[*i].0)).collect();
        for (n, i) in missing.iter().enumerate() {
            on_progress(n, missing.len());
            pool.prime(&self.parts[*i].0, &self.id, &self.part_prefix(*i))?;
        }
        on_progress(missing.len(), missing.len());
        Ok(missing.len())
    }

    /// The index for `chunking` (with Mode 2's expansions by `model_id` when `expanded`), from `dir` if it was made
    /// before in the same way, else made now and kept there.
    pub fn index(&self, pool: &KvPool, dir: &Path, chunking: Chunking, expanded: bool, model_id: &str, embedder: Option<&dyn Embedder>, on_stage: &mut dyn FnMut(&str)) -> Result<Index> {
        // chunks sized for this model's window (an index made for another size is another index)
        let chunk_chars = crate::Budget::for_engine(pool.engine().context_size()).chunk_chars;
        let size = if chunk_chars == crate::index::CHUNK_CHARS { String::new() } else { format!("-c{chunk_chars}") };
        let key = format!("{}{}{size}-{model_id}", if chunking == Chunking::Guided { "guided" } else { "plain" }, if expanded { "-expanded" } else { "" });
        let dir = dir.join(format!("index-{key}"));
        if let Some(mut ix) = Index::load(&dir)? {
            if ix.embed_model == embedder.map(|e| e.id()) {
                // expansions this model made, with some chunks left undescribed: describe just those
                if !(expanded && ix.made_by.as_deref() == Some(model_id) && !ix.unexpanded().is_empty()) {
                    return Ok(ix);
                }
                on_stage("describing the passages the index is missing");
                expand(pool, self, &mut ix, model_id, &mut |_, _| {})?;
                if let Some(e) = embedder {
                    ix.embed(e, &mut |_, _| {})?;
                }
                ix.save(&dir)?;
                return Ok(ix);
            }
        }
        let mut ix = match chunking {
            Chunking::Plain => Index::plain_sized(&(0..self.parts.len()).map(|i| self.part_text(i)).collect::<Vec<_>>(), chunk_chars),
            Chunking::Guided => {
                on_stage("finding where scenes and topics change");
                guided_index(pool, self, &mut |_, _| {})?
            }
        };
        if expanded {
            on_stage("describing every passage (expansions)");
            expand(pool, self, &mut ix, model_id, &mut |_, _| {})?;
        }
        if let Some(e) = embedder {
            on_stage("embedding the index");
            ix.embed(e, &mut |_, _| {})?;
        }
        ix.save(&dir)?;
        Ok(ix)
    }
}

impl Corpus for Document {
    fn part_count(&self) -> usize {
        self.parts.len()
    }
    fn part_file(&self, i: usize) -> &str {
        &self.parts[i].0
    }
    fn part_owner(&self, _i: usize) -> &str {
        &self.id
    }
    fn part_prefix(&self, i: usize) -> String {
        self.parts[i].1.prefix(&self.title, self.author.as_deref(), self.parts.len())
    }
    fn part_text(&self, i: usize) -> &str {
        &self.parts[i].1.text
    }
    fn title(&self) -> &str {
        &self.title
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn a_document_is_split_saved_once_and_indexed_once() {
        let dir = std::env::temp_dir().join(format!("docuzent-document-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let engine = Arc::new(docuzent_llm::SimEngine::new(12288, |_| String::new()).in_dir(dir.join("kv")));
        let pool = KvPool::new(engine.clone(), Arc::new(docuzent_kv::KvStore::open(dir.join("kv"), 1 << 30).unwrap()));
        let text = "A paragraph about duty.\n\n".repeat(2000);
        let doc = Document::from_text(&text, "Duty", None, "m", &|s| Ok(s.len() / 4)).unwrap();
        assert!(doc.parts.len() > 1);
        assert_eq!(doc.save_parts(&pool, &mut |_, _| {}).unwrap(), doc.parts.len());
        assert_eq!(doc.save_parts(&pool, &mut |_, _| {}).unwrap(), 0, "saved once");
        let a = doc.index(&pool, &dir, Chunking::Plain, false, "m", None, &mut |_| {}).unwrap();
        assert!(dir.join("index-plain-m").join("index.json").is_file());
        let b = doc.index(&pool, &dir, Chunking::Plain, false, "m", None, &mut |_| panic!("made again")).unwrap();
        assert_eq!(a.chunks.len(), b.chunks.len());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
