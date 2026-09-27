//! Ties ingestion together: for each file in a folder, skip it if its
//! mtime/size match what is already stored, otherwise run it through
//! Docling, chunk the result by page, embed each chunk, and store it. Only
//! files that actually changed touch the model or the store, so a repeat
//! ingest of an unchanged folder leaves the corpus version stable.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use crate::chunk::{self, DEFAULT_MAX_CHARS};
use crate::embed::Embedder;
use crate::ingest::{self, IngestOptions};
use crate::store::{NewChunk, Store};

pub struct FileOutcome {
    pub doc_path: String,
    pub skipped_unchanged: bool,
    pub chunks_stored: usize,
}

/// Ingests every file directly inside `source` (non-recursive) into
/// `store`, using `work_dir` as scratch space for Docling's output.
pub fn ingest_folder(
    source: &Path,
    work_dir: &Path,
    store: &Store,
    embedder: &dyn Embedder,
    device: &str,
) -> Result<Vec<FileOutcome>> {
    std::fs::create_dir_all(work_dir)
        .with_context(|| format!("failed to create work dir {}", work_dir.display()))?;

    let mut outcomes = Vec::new();
    for entry in std::fs::read_dir(source)
        .with_context(|| format!("failed to read source dir {}", source.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let doc_path = path.to_string_lossy().into_owned();
        let metadata = entry.metadata()?;
        let mtime = metadata
            .modified()?
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let size = metadata.len();

        if store.is_unchanged(&doc_path, mtime, size)? {
            outcomes.push(FileOutcome { doc_path, skipped_unchanged: true, chunks_stored: 0 });
            continue;
        }

        let report = ingest::run(&IngestOptions {
            source: path.clone(),
            output: work_dir.to_path_buf(),
            to: "json".to_string(),
            device: device.to_string(),
            docling_bin: None,
        })?;

        let stem = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
        let json_name = report
            .produced_files
            .iter()
            .find(|f| f.starts_with(&stem))
            .cloned()
            .with_context(|| format!("docling produced no output for {}", path.display()))?;
        let json_path = work_dir.join(json_name);

        let chunk_inputs = chunk::chunk_docling_json(&json_path, DEFAULT_MAX_CHARS)?;
        let mut new_chunks = Vec::with_capacity(chunk_inputs.len());
        for c in chunk_inputs {
            let embedding = embedder
                .embed(&c.text)
                .with_context(|| format!("failed to embed chunk {} of {}", c.chunk_index, doc_path))?;
            new_chunks.push(NewChunk { chunk_index: c.chunk_index, page: c.page, text: c.text, embedding });
        }

        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        let chunks_stored = new_chunks.len();
        store.replace_document(&doc_path, mtime, size, now, new_chunks)?;

        outcomes.push(FileOutcome { doc_path, skipped_unchanged: false, chunks_stored });
    }
    Ok(outcomes)
}
