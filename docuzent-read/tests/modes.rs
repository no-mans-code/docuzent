//! The four reading modes end to end, over a real KV pool and store with a simulated model: what each one does,
//! and - as much as what it finds - what it costs (which calls it makes, which parts it touches).

use std::sync::Arc;

use docuzent_doc::kvpool::KvPool;
use docuzent_kv::KvStore;
use docuzent_llm::{chatml, SimEmbedder, SimEngine};
use docuzent_read::expand::{expand, guided_index};
use docuzent_read::{read, Chunking, Index, MemCorpus, Mode, ReadOptions, Sources};

const PARTS: [&str; 5] = [
    "Mr and Mrs Dursley of number four, Privet Drive, were proud to say they were perfectly normal.\n\nThey had a small son called Dudley.",
    "Harry went to the zoo. A snake winked at him through the glass, and then the glass vanished.",
    "Hagrid won a dragon egg in a card game at the pub. He named the dragon Norbert; it was a Norwegian Ridgeback.",
    "At the final feast Dumbledore awarded ten points to Neville Longbottom, and Gryffindor won the House Cup.",
    "Harry went home to the Dursleys for the summer.",
];

fn question_of(prompt: &str) -> String {
    prompt.rsplit("A reader asks: \"").next().and_then(|s| s.split('"').next()).unwrap_or("").to_lowercase()
}

/// The part a prompt is about: the last saved part text it contains.
fn part_in(prompt: &str) -> Option<usize> {
    PARTS.iter().position(|p| prompt.contains(&p[..40]))
}

/// A model that scores a part 9 when it shares a word with the question, quotes the part when asked to extract,
/// describes passages with a telling word, and cuts parts at their second paragraph.
fn responder(prompt: &str) -> String {
    if prompt.contains("How much of this part of the book bears on that question") {
        let q = question_of(prompt);
        let part = part_in(prompt).map(|i| PARTS[i].to_lowercase()).unwrap_or_default();
        let hit = q.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 3).any(|w| part.contains(w));
        return if hit { "9" } else { "0" }.into();
    }
    if prompt.contains("Now read only the second half") {
        return "NONE".into();
    }
    if prompt.contains("tell me everything in it that bears on that question") {
        return part_in(prompt).map(|i| format!("It says: \"{}\"", PARTS[i])).unwrap_or_else(|| "NONE".into());
    }
    if prompt.contains("Below are passages from this part of the book") {
        // the zoo passage is described as a "reptile house", a word it never uses
        let n = prompt.matches("\n[").count().max(1);
        let items: Vec<String> = (1..=n).map(|i| format!(r#"{{"n": {i}, "context": "", "people": "", "setting": "{}", "facts": [], "questions": []}}"#, if prompt.contains("snake winked") { "a reptile house at a zoo" } else { "somewhere" })).collect();
        return format!("[{}]", items.join(","));
    }
    if prompt.contains("change scene, time, place or topic") {
        return if prompt.contains("They had a small son") { "They had a small son called Dudley.".into() } else { "NONE".into() };
    }
    String::new()
}

struct Rig {
    engine: Arc<SimEngine>,
    pool: KvPool,
    corpus: MemCorpus,
    dir: std::path::PathBuf,
}

fn rig(label: &str) -> Rig {
    let dir = std::env::temp_dir().join(format!("docuzent-modes-{label}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let engine = Arc::new(SimEngine::new(12288, responder).in_dir(&dir));
    let store = Arc::new(KvStore::open(&dir, 1 << 30).unwrap());
    let pool = KvPool::new(engine.clone(), store);
    let parts = PARTS.iter().enumerate().map(|(i, t)| (format!("m-book-p{i}.kv"), chatml::system(&format!("Part {} of the book.\n\n{t}", i + 1)), t.to_string())).collect();
    let corpus = MemCorpus { title: "The Book".into(), owner: "book".into(), parts, gist: String::new() };
    for i in 0..PARTS.len() {
        pool.prime(&corpus.parts[i].0, "book", &corpus.parts[i].1).unwrap();
    }
    engine.clear_events();
    Rig { engine, pool, corpus, dir }
}

fn scored(r: &Rig) -> usize {
    r.engine.events().iter().filter(|e| matches!(e, docuzent_llm::sim::Event::Complete { .. })).count()
}

#[test]
fn mode_4_scores_every_part_and_reads_the_relevant_ones_closely() {
    let r = rig("kv");
    let reading = read(Mode::Kv, Sources { pool: &r.pool, corpus: &r.corpus, indexes: Vec::new(), embedder: None }, "What breed was the dragon Norbert?", ReadOptions::default(), &|_| {}).unwrap();
    assert_eq!(reading.scores.len(), PARTS.len(), "every part scored");
    assert!(reading.passages.iter().any(|p| p.part == 3 && p.text.contains("Norwegian Ridgeback")));
    assert_eq!(reading.read_afresh, 0, "every part restored from its saved state");
    let _ = std::fs::remove_dir_all(&r.dir);
}

#[test]
fn mode_1_answers_from_chunks_without_calling_the_model() {
    let r = rig("rag");
    let mut index = Index::plain(&PARTS);
    let e = SimEmbedder::default();
    index.embed(&e, &mut |_, _| {}).unwrap();
    let reading = read(Mode::Rag, Sources { pool: &r.pool, corpus: &r.corpus, indexes: vec![docuzent_read::Shelf { index: &index, first_part: 0 }], embedder: Some(&e) }, "What breed was the dragon Norbert?", ReadOptions::default(), &|_| {}).unwrap();
    assert_eq!(scored(&r), 0, "no model call at all");
    assert!(reading.passages.iter().any(|p| p.part == 3 && p.text.contains("Ridgeback")));
    assert!(reading.retrieved > 0 && reading.scores.is_empty());
    let _ = std::fs::remove_dir_all(&r.dir);
}

#[test]
fn mode_2_finds_a_chunk_through_its_expansions_where_mode_1_cannot() {
    let r = rig("expanded");
    let mut index = Index::plain(&PARTS);
    expand(&r.pool, &r.corpus, &mut index, "sim-model", &mut |_, _| {}).unwrap();
    assert!(index.has_expansions() && index.made_by.as_deref() == Some("sim-model"));
    r.engine.clear_events();
    let q = "What happened at the reptile house?";
    let plain = read(Mode::Rag, Sources { pool: &r.pool, corpus: &r.corpus, indexes: vec![docuzent_read::Shelf { index: &index, first_part: 0 }], embedder: None }, q, ReadOptions::default(), &|_| {}).unwrap();
    assert!(!plain.passages.iter().any(|p| p.part == 2), "the zoo chunk never says 'reptile'");
    let expanded = read(Mode::RagExpanded, Sources { pool: &r.pool, corpus: &r.corpus, indexes: vec![docuzent_read::Shelf { index: &index, first_part: 0 }], embedder: None }, q, ReadOptions::default(), &|_| {}).unwrap();
    assert!(expanded.passages.iter().any(|p| p.part == 2 && p.text.contains("snake winked")), "found through its expansion, answered from its own text");
    assert_eq!(scored(&r), 0, "searching costs no model call");
    let _ = std::fs::remove_dir_all(&r.dir);
}

#[test]
fn mode_3_reads_only_the_parts_the_index_points_at_whole_without_scoring() {
    let r = rig("ragkv");
    let index = Index::plain(&PARTS);
    let reading = read(Mode::RagKv, Sources { pool: &r.pool, corpus: &r.corpus, indexes: vec![docuzent_read::Shelf { index: &index, first_part: 0 }], embedder: None }, "Who was awarded ten points at the final feast?", ReadOptions::default(), &|_| {}).unwrap();
    let prompts: Vec<String> = r.engine.events().iter().filter_map(|e| if let docuzent_llm::sim::Event::Complete { prompt_head, .. } = e { Some(prompt_head.clone()) } else { None }).collect();
    assert!(reading.scores.is_empty(), "no part was scored");
    assert!(reading.read_closely >= 1 && reading.read_closely <= docuzent_read::modes::RAG_KV_PARTS, "{}", reading.read_closely);
    assert!(reading.passages.iter().any(|p| p.part == 4 && p.text.contains("Neville Longbottom")), "{:?}", reading.passages);
    // the index's passages come first, in the book's own words; the whole-part reading enriches after them
    // ... labelled as a reading, so the answer prefers the book's words where the two differ
    let is_reading = |p: &docuzent_read::Passage| p.text.starts_with("(Notes from reading the whole part") && p.text.contains("It says:");
    let first_read = reading.passages.iter().position(is_reading).expect("the part's reading is there");
    assert!(first_read > 0 && reading.passages[..first_read].iter().all(|p| !is_reading(p)), "{:?}", reading.passages);
    assert!(prompts.len() <= 2 * docuzent_read::modes::RAG_KV_PARTS, "an extraction and a second-half look per part, nothing more: {}", prompts.len());
    let _ = std::fs::remove_dir_all(&r.dir);
}

#[test]
fn every_mode_respects_a_restriction_to_some_parts() {
    let r = rig("only");
    let index = Index::plain(&PARTS);
    for mode in Mode::ALL {
        let reading = read(mode, Sources { pool: &r.pool, corpus: &r.corpus, indexes: vec![docuzent_read::Shelf { index: &index, first_part: 0 }], embedder: None }, "Where did the Dursleys live, and what breed was Norbert?", ReadOptions { only: Some(&[0]), ..Default::default() }, &|_| {}).unwrap();
        assert!(reading.passages.iter().all(|p| p.part == 1), "{mode}: {:?}", reading.passages.iter().map(|p| p.part).collect::<Vec<_>>());
    }
    let _ = std::fs::remove_dir_all(&r.dir);
}

#[test]
fn a_rag_mode_without_an_index_reads_every_part_and_says_so() {
    let r = rig("noindex");
    let said = std::sync::Mutex::new(Vec::new());
    let reading = read(Mode::Rag, Sources { pool: &r.pool, corpus: &r.corpus, indexes: Vec::new(), embedder: None }, "What breed was Norbert?", ReadOptions::default(), &|s| said.lock().unwrap().push(s.to_string())).unwrap();
    assert_eq!(reading.scores.len(), PARTS.len());
    assert!(said.lock().unwrap().iter().any(|s| s.contains("No index")));
    let _ = std::fs::remove_dir_all(&r.dir);
}

#[test]
fn guided_chunking_cuts_where_the_model_says_a_section_starts() {
    let r = rig("guided");
    let index = guided_index(&r.pool, &r.corpus, &mut |_, _| {}).unwrap();
    assert_eq!(index.chunking, Chunking::Guided);
    let first: Vec<&str> = index.chunks.iter().filter(|c| c.part == 0).map(|c| c.text.as_str()).collect();
    assert_eq!(first, vec!["Mr and Mrs Dursley of number four, Privet Drive, were proud to say they were perfectly normal.", "They had a small son called Dudley."]);
    assert_eq!(index.chunks.iter().filter(|c| c.part == 1).count(), 1, "a part the model calls one section stays whole");
    let _ = std::fs::remove_dir_all(&r.dir);
}

/// A model that is cut off whenever it describes more than one passage: it finishes the first and stops mid-way
/// through the second. Every passage still gets described - the ones it was cut off on, on their own.
fn cut_off_responder(prompt: &str) -> String {
    if prompt.contains("Below are passages from this part of the book") {
        let one = r#"{"n": 1, "context": "", "people": "", "setting": "a described passage", "facts": [], "questions": []}"#;
        return if prompt.contains("\n[2]\n") { format!(r#"[{one}, {{"n": 2, "context": "and then the tok"#) } else { format!("[{one}]") };
    }
    String::new()
}

#[test]
fn a_batch_cut_off_mid_way_is_described_again_passage_by_passage_and_a_second_pass_repairs_only_what_is_missing() {
    let dir = std::env::temp_dir().join(format!("docuzent-modes-cutoff-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let engine = Arc::new(SimEngine::new(12288, cut_off_responder).in_dir(&dir));
    let pool = KvPool::new(engine.clone(), Arc::new(KvStore::open(&dir, 1 << 30).unwrap()));
    // one long part: several chunks, so passages are described three at a time
    let text = (0..7).map(|i| format!("Paragraph {i}. {}", "Something happens here, and then something else. ".repeat(30))).collect::<Vec<_>>().join("\n\n");
    let corpus = MemCorpus { title: "The Book".into(), owner: "book".into(), parts: vec![("m-long-p0.kv".into(), chatml::system(&format!("Part 1 of the book.\n\n{text}")), text.clone())], gist: String::new() };
    pool.prime(&corpus.parts[0].0, "book", &corpus.parts[0].1).unwrap();
    let mut index = Index::plain(&[text.as_str()]);
    assert!(index.chunks.len() >= 4, "{} chunks", index.chunks.len());
    let left = expand(&pool, &corpus, &mut index, "sim-model", &mut |_, _| {}).unwrap();
    assert_eq!((left, index.unexpanded()), (0, Vec::<usize>::new()), "every chunk described");

    // an index with a chunk's descriptions lost is repaired: only that chunk is described again
    let lost = 1;
    index.add_expansions(index.entries.iter().filter(|e| e.kind.is_expansion() && e.chunk != lost).cloned().collect(), "sim-model");
    assert_eq!(index.unexpanded(), vec![lost]);
    engine.clear_events();
    assert_eq!(expand(&pool, &corpus, &mut index, "sim-model", &mut |_, _| {}).unwrap(), 0);
    assert!(index.unexpanded().is_empty());
    let calls = engine.events().iter().filter(|e| matches!(e, docuzent_llm::sim::Event::Complete { .. })).count();
    assert_eq!(calls, 1, "one call, for the one chunk that was missing");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A model that never manages to describe one passage (here: any prompt holding it). The passage is tried on its
/// own, recorded as undescribable, and not tried again on the next load - it is still found by its own text.
fn stubborn_responder(prompt: &str) -> String {
    if prompt.contains("Below are passages from this part of the book") {
        if prompt.contains("THE STUBBORN ONE") {
            return "I cannot describe this.".into();
        }
        let n = prompt.matches("\n[").count().max(1);
        let items: Vec<String> = (1..=n).map(|i| format!(r#"{{"n": {i}, "context": "", "people": "", "setting": "described", "facts": [], "questions": []}}"#)).collect();
        return format!("[{}]", items.join(","));
    }
    String::new()
}

#[test]
fn a_passage_the_model_cannot_describe_is_recorded_once_and_not_retried_on_every_load() {
    let dir = std::env::temp_dir().join(format!("docuzent-modes-stubborn-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let engine = Arc::new(SimEngine::new(12288, stubborn_responder).in_dir(&dir));
    let pool = KvPool::new(engine.clone(), Arc::new(KvStore::open(&dir, 1 << 30).unwrap()));
    let parts = ["A plain first part about a garden.", "THE STUBBORN ONE: a passage the model will not describe.", "A plain third part about a river."];
    let corpus = MemCorpus { title: "The Book".into(), owner: "book".into(), parts: parts.iter().enumerate().map(|(i, t)| (format!("m-stub-p{i}.kv"), chatml::system(&format!("Part {}.\n\n{t}", i + 1)), t.to_string())).collect(), gist: String::new() };
    let mut index = Index::plain(&parts);
    assert_eq!(expand(&pool, &corpus, &mut index, "sim-model", &mut |_, _| {}).unwrap(), 1);
    assert_eq!((index.undescribable.clone(), index.unexpanded()), (vec![1], Vec::<usize>::new()));
    engine.clear_events();
    assert_eq!(expand(&pool, &corpus, &mut index, "sim-model", &mut |_, _| {}).unwrap(), 0, "nothing left to try");
    assert_eq!(engine.events().iter().filter(|e| matches!(e, docuzent_llm::sim::Event::Complete { .. })).count(), 0, "no call for it again");
    let fresh = expand(&pool, &corpus, &mut index, "another-model", &mut |_, _| {}).unwrap();
    assert_eq!((fresh, index.undescribable.clone()), (1, vec![1]), "a new model starts afresh (and fails afresh)");
    let _ = std::fs::remove_dir_all(&dir);
}
