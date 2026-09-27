//! Real, non-mocked end-to-end check: Docling actually converts, Ollama
//! actually embeds, the store actually persists. Uses a 2-file subset of
//! `../temp-test` (one JPG, one small PDF) rather than the whole folder, so
//! this stays fast - each file spawns its own Docling subprocess, which
//! reloads its models from disk every time, unlike the batched CLI path.

use std::path::PathBuf;

use docuzent_core::embed::OllamaEmbedder;
use docuzent_core::pipeline::ingest_folder;
use docuzent_core::store::Store;

struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn unique_temp_dir(label: &str) -> TempDir {
    let path = std::env::temp_dir().join(format!("docuzent-core-it-{label}-{}", std::process::id()));
    std::fs::create_dir_all(&path).unwrap();
    TempDir(path)
}

#[test]
fn ingest_then_reingest_unchanged_is_stable() {
    let source_dir = unique_temp_dir("source");
    let work_dir = unique_temp_dir("work");
    let db_dir = unique_temp_dir("db");

    for name in ["jess1cc (1).jpg", "jess1a1 (4).pdf"] {
        std::fs::copy(
            PathBuf::from("../temp-test").join(name),
            source_dir.0.join(name),
        )
        .unwrap_or_else(|e| panic!("failed to stage fixture {name}: {e}"));
    }

    let store = Store::open(&db_dir.0.join("test.redb")).unwrap();
    let embedder = OllamaEmbedder::default_local();

    let first = ingest_folder(&source_dir.0, &work_dir.0, &store, &embedder, "cpu").unwrap();
    assert_eq!(first.len(), 2);
    for outcome in &first {
        assert!(!outcome.skipped_unchanged, "{} should not be skipped on first ingest", outcome.doc_path);
        assert!(outcome.chunks_stored > 0, "{} produced no chunks", outcome.doc_path);
    }
    let version_after_first = store.corpus_version().unwrap();
    assert_eq!(version_after_first, 2, "one version bump per newly-ingested file");
    let chunk_count_after_first = store.chunk_count().unwrap();
    assert!(chunk_count_after_first > 0);

    let second = ingest_folder(&source_dir.0, &work_dir.0, &store, &embedder, "cpu").unwrap();
    assert_eq!(second.len(), 2);
    for outcome in &second {
        assert!(outcome.skipped_unchanged, "{} should be skipped on unchanged re-ingest", outcome.doc_path);
        assert_eq!(outcome.chunks_stored, 0);
    }
    assert_eq!(
        store.corpus_version().unwrap(),
        version_after_first,
        "corpus version must stay stable when nothing changed"
    );
    assert_eq!(store.chunk_count().unwrap(), chunk_count_after_first);
}
