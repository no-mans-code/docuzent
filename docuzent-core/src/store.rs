//! Embedded storage: chunk text + embedding + citation metadata, a
//! per-document mtime/size record (to skip re-ingesting unchanged files),
//! and a corpus version counter that only bumps when something actually
//! changed. Backed by `redb` (pure Rust, no C toolchain needed) rather
//! than SQLite - retrieval is brute-force cosine over every chunk regardless
//! (entirely adequate at the scale a local document folder produces, and
//! this project is accuracy-focused, not latency-focused), so no relational
//! query engine is needed, only get/scan/delete.

use std::path::Path;

use anyhow::{Context, Result};
use redb::{Database, ReadableTable, ReadableTableMetadata, TableDefinition};
use serde::{Deserialize, Serialize};

use crate::embed::cosine_similarity;

const DOCUMENTS: TableDefinition<&str, &str> = TableDefinition::new("documents");
const CHUNKS: TableDefinition<u64, &str> = TableDefinition::new("chunks");
const META: TableDefinition<&str, &str> = TableDefinition::new("meta");

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DocumentMeta {
    pub mtime: u64,
    pub size: u64,
    pub last_ingested_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChunkRecord {
    pub doc_path: String,
    pub chunk_index: usize,
    pub page: Option<u32>,
    pub text: String,
    pub embedding: Vec<f32>,
}

/// One chunk ready to be written, before it has a stored id.
pub struct NewChunk {
    pub chunk_index: usize,
    pub page: Option<u32>,
    pub text: String,
    pub embedding: Vec<f32>,
}

pub struct Store {
    db: Database,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        let db = Database::create(path)
            .with_context(|| format!("failed to open store at {}", path.display()))?;
        // Touch every table once so a brand-new database has them, rather
        // than deferring creation to the first real write.
        let txn = db.begin_write()?;
        txn.open_table(DOCUMENTS)?;
        txn.open_table(CHUNKS)?;
        txn.open_table(META)?;
        txn.commit()?;
        Ok(Self { db })
    }

    pub fn corpus_version(&self) -> Result<u64> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(META)?;
        Ok(table
            .get("corpus_version")?
            .and_then(|v| v.value().parse().ok())
            .unwrap_or(0))
    }

    pub fn document_meta(&self, doc_path: &str) -> Result<Option<DocumentMeta>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(DOCUMENTS)?;
        Ok(match table.get(doc_path)? {
            Some(v) => Some(serde_json::from_str(v.value())?),
            None => None,
        })
    }

    /// True if `doc_path` is already ingested with this exact mtime/size -
    /// the caller should skip re-extracting, re-embedding, and re-storing it.
    pub fn is_unchanged(&self, doc_path: &str, mtime: u64, size: u64) -> Result<bool> {
        Ok(match self.document_meta(doc_path)? {
            Some(meta) => meta.mtime == mtime && meta.size == size,
            None => false,
        })
    }

    /// Replaces every stored chunk for `doc_path` with `chunks`, updates its
    /// document record, and bumps the corpus version. Call only when the
    /// document is new or has actually changed (see [`Self::is_unchanged`]) -
    /// this always bumps the version, by design, so a stable corpus version
    /// after a no-op ingest depends on the caller checking first.
    pub fn replace_document(
        &self,
        doc_path: &str,
        mtime: u64,
        size: u64,
        ingested_at: u64,
        chunks: Vec<NewChunk>,
    ) -> Result<()> {
        let txn = self.db.begin_write()?;
        {
            let mut chunk_table = txn.open_table(CHUNKS)?;
            let mut to_remove = Vec::new();
            for entry in chunk_table.iter()? {
                let (k, v) = entry?;
                let record: ChunkRecord = serde_json::from_str(v.value())?;
                if record.doc_path == doc_path {
                    to_remove.push(k.value());
                }
            }
            for k in to_remove {
                chunk_table.remove(k)?;
            }

            let mut meta_table = txn.open_table(META)?;
            let mut next_id: u64 = meta_table
                .get("next_chunk_id")?
                .and_then(|v| v.value().parse().ok())
                .unwrap_or(0);

            for chunk in &chunks {
                let record = ChunkRecord {
                    doc_path: doc_path.to_string(),
                    chunk_index: chunk.chunk_index,
                    page: chunk.page,
                    text: chunk.text.clone(),
                    embedding: chunk.embedding.clone(),
                };
                let serialized = serde_json::to_string(&record)?;
                chunk_table.insert(next_id, serialized.as_str())?;
                next_id += 1;
            }
            meta_table.insert("next_chunk_id", next_id.to_string().as_str())?;

            let mut doc_table = txn.open_table(DOCUMENTS)?;
            let doc_meta = DocumentMeta { mtime, size, last_ingested_at: ingested_at };
            doc_table.insert(doc_path, serde_json::to_string(&doc_meta)?.as_str())?;

            let version: u64 = meta_table
                .get("corpus_version")?
                .and_then(|v| v.value().parse().ok())
                .unwrap_or(0);
            meta_table.insert("corpus_version", (version + 1).to_string().as_str())?;
        }
        txn.commit()?;
        Ok(())
    }

    /// Removes every chunk and the document record for `doc_path`, bumping
    /// the corpus version. A no-op (no version bump) if it was not present.
    pub fn remove_document(&self, doc_path: &str) -> Result<()> {
        let txn = self.db.begin_write()?;
        let mut removed_any = false;
        {
            let mut chunk_table = txn.open_table(CHUNKS)?;
            let mut to_remove = Vec::new();
            for entry in chunk_table.iter()? {
                let (k, v) = entry?;
                let record: ChunkRecord = serde_json::from_str(v.value())?;
                if record.doc_path == doc_path {
                    to_remove.push(k.value());
                }
            }
            for k in to_remove {
                chunk_table.remove(k)?;
                removed_any = true;
            }

            let mut doc_table = txn.open_table(DOCUMENTS)?;
            if doc_table.remove(doc_path)?.is_some() {
                removed_any = true;
            }

            if removed_any {
                let mut meta_table = txn.open_table(META)?;
                let version: u64 = meta_table
                    .get("corpus_version")?
                    .and_then(|v| v.value().parse().ok())
                    .unwrap_or(0);
                meta_table.insert("corpus_version", (version + 1).to_string().as_str())?;
            }
        }
        txn.commit()?;
        Ok(())
    }

    pub fn all_chunks(&self) -> Result<Vec<ChunkRecord>> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(CHUNKS)?;
        let mut out = Vec::new();
        for entry in table.iter()? {
            let (_, v) = entry?;
            out.push(serde_json::from_str(v.value())?);
        }
        Ok(out)
    }

    pub fn chunk_count(&self) -> Result<usize> {
        let txn = self.db.begin_read()?;
        let table = txn.open_table(CHUNKS)?;
        Ok(table.len()? as usize)
    }

    /// Brute-force cosine similarity over every stored chunk, highest first.
    pub fn retrieve(&self, query_embedding: &[f32], top_k: usize) -> Result<Vec<(ChunkRecord, f32)>> {
        let mut scored: Vec<(ChunkRecord, f32)> = self
            .all_chunks()?
            .into_iter()
            .map(|c| {
                let score = cosine_similarity(query_embedding, &c.embedding);
                (c, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// A throwaway store file under the OS temp dir, removed on drop.
    /// Written by hand rather than pulling in `tempfile`, which drags in a
    /// `getrandom` build script this environment's Application Control
    /// policy blocks - unnecessary weight for what a test needs, which is
    /// just a unique path.
    struct TempStore {
        store: Store,
        path: std::path::PathBuf,
    }

    impl std::ops::Deref for TempStore {
        type Target = Store;
        fn deref(&self) -> &Store {
            &self.store
        }
    }

    impl Drop for TempStore {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn open_temp() -> TempStore {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "docuzent-core-test-{}-{n}.redb",
            std::process::id()
        ));
        let store = Store::open(&path).unwrap();
        TempStore { store, path }
    }

    #[test]
    fn fresh_store_has_version_zero_and_no_chunks() {
        let store = open_temp();
        assert_eq!(store.corpus_version().unwrap(), 0);
        assert_eq!(store.chunk_count().unwrap(), 0);
    }

    #[test]
    fn replace_document_bumps_version_and_stores_chunks() {
        let store = open_temp();
        store
            .replace_document(
                "a.pdf",
                100,
                10,
                1000,
                vec![NewChunk { chunk_index: 0, page: Some(1), text: "hello".into(), embedding: vec![1.0, 0.0] }],
            )
            .unwrap();
        assert_eq!(store.corpus_version().unwrap(), 1);
        assert_eq!(store.chunk_count().unwrap(), 1);
        assert!(store.is_unchanged("a.pdf", 100, 10).unwrap());
        assert!(!store.is_unchanged("a.pdf", 200, 10).unwrap());
    }

    #[test]
    fn re_ingest_replaces_rather_than_accumulates() {
        let store = open_temp();
        store
            .replace_document("a.pdf", 100, 10, 1000, vec![NewChunk { chunk_index: 0, page: None, text: "v1".into(), embedding: vec![1.0] }])
            .unwrap();
        store
            .replace_document("a.pdf", 200, 20, 2000, vec![NewChunk { chunk_index: 0, page: None, text: "v2".into(), embedding: vec![1.0] }])
            .unwrap();
        assert_eq!(store.chunk_count().unwrap(), 1);
        assert_eq!(store.corpus_version().unwrap(), 2);
        assert_eq!(store.all_chunks().unwrap()[0].text, "v2");
    }

    #[test]
    fn retrieve_ranks_by_similarity() {
        let store = open_temp();
        store
            .replace_document(
                "a.pdf",
                1,
                1,
                1,
                vec![
                    NewChunk { chunk_index: 0, page: None, text: "close".into(), embedding: vec![1.0, 0.0] },
                    NewChunk { chunk_index: 1, page: None, text: "far".into(), embedding: vec![0.0, 1.0] },
                ],
            )
            .unwrap();
        let results = store.retrieve(&[1.0, 0.0], 2).unwrap();
        assert_eq!(results[0].0.text, "close");
        assert_eq!(results[1].0.text, "far");
    }
}
