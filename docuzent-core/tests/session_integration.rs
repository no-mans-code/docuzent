//! Real, non-mocked check of the single-document Q&A session: real
//! Docling, real Ollama, real disk-persisted context surviving a brand
//! new `Session` (standing in for a process restart) for the same
//! document.

use std::path::PathBuf;

use docuzent_core::docling_cache::DoclingCache;
use docuzent_core::generate::OllamaClient;
use docuzent_core::session::{infer_context_length, Mode, Session};
use kvcache::Cache;

struct TempPath(PathBuf);
impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        let _ = std::fs::remove_dir_all(self.0.with_extension(""));
    }
}

fn temp_path(label: &str) -> TempPath {
    TempPath(std::env::temp_dir().join(format!("docuzent-core-session-it-{label}-{}.redb", std::process::id())))
}

fn temp_docling_path(label: &str) -> TempPath {
    TempPath(std::env::temp_dir().join(format!("docuzent-core-session-it-docling-{label}-{}.redb", std::process::id())))
}

const MODEL: &str = "qwen2.5:3b";
const HOST: &str = "http://localhost:11434";

#[test]
fn context_persists_to_disk_across_a_fresh_session() {
    let db_path = temp_path("persist");
    let docling_path = temp_docling_path("persist");
    let work_dir = std::env::temp_dir().join(format!("docuzent-core-session-it-work-{}", std::process::id()));

    let context_length = infer_context_length(HOST, MODEL).expect("ollama must be running with the model pulled");

    // Mode::Swap explicitly - this test exercises the disk-persistence
    // *mechanism* itself, not Adaptive's prediction logic (which, with no
    // prior speed data for this model, would cold-prime on its first real
    // decision regardless of what's on disk - see session.rs's own
    // `adaptive_mode_with_no_prior_speed_data_falls_back_to_a_cold_prime`).
    let cache1 = Cache::open(&db_path.0, 1_000_000_000).unwrap();
    let docling_cache1 = DoclingCache::open(&docling_path.0, 1_000_000_000).unwrap();
    let mut session1 =
        Session::open(OllamaClient::new(HOST, MODEL, context_length), MODEL, context_length, work_dir.clone(), Mode::Swap, cache1, docling_cache1).unwrap();

    let doc_path = PathBuf::from("../temp-test/jess1a1 (4).pdf");
    let load1 = session1.load_document(&doc_path).unwrap();
    assert!(!load1.warm_from_disk, "first load of a never-seen document should not be warm");
    assert!(load1.chars > 0);

    let answer1 = session1.ask("What is this document about, in one sentence?").unwrap();
    assert!(!answer1.answer.is_empty());
    assert_eq!(answer1.timings[0].label, "prime");
    drop(session1); // release the redb file handle before reopening it

    // Second session: brand new Session, same DB file, same document -
    // simulates a fresh process picking the cache back up.
    let cache2 = Cache::open(&db_path.0, 1_000_000_000).unwrap();
    let docling_cache2 = DoclingCache::open(&docling_path.0, 1_000_000_000).unwrap();
    let mut session2 =
        Session::open(OllamaClient::new(HOST, MODEL, context_length), MODEL, context_length, work_dir.clone(), Mode::Swap, cache2, docling_cache2).unwrap();
    let load2 = session2.load_document(&doc_path).unwrap();
    assert!(load2.warm_from_disk, "the same document, model, and context length should hit the disk cache");

    let answer2 = session2.ask("Who might use this document?").unwrap();
    assert!(!answer2.answer.is_empty());
    // No "prime" call needed - the disk-persisted context from session 1 was reused.
    assert_eq!(answer2.timings.len(), 1);
    assert_eq!(answer2.timings[0].label, "answer");

    std::fs::remove_dir_all(&work_dir).ok();
}
