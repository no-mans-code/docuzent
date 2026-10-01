//! What the model adds to an index: where scenes change (guided chunking), and, for every chunk, the other ways it
//! can be found (Mode 2's expansions).
//!
//! Both are done with the part's saved KV state resident, so the model has the whole part in view while it
//! describes one passage of it - which is the point of *contextual* retrieval (Provider, 2024: a chunk prefixed
//! with a sentence situating it in its document; they report 49% fewer failed retrievals, 67% with reranking) -
//! at the cost of a restore, not a re-read of the part.
//!
//! The expansions follow published work:
//! - **context**: a sentence placing the passage in the story (contextual retrieval);
//! - **people** and **setting**: the entities it involves - people, places, objects, events - so a question about
//!   any of them reaches it (the entity index of GraphRAG, Edge et al., 2024, without the graph);
//! - **facts**: the atomic statements it makes (propositions - *Dense X Retrieval*, Chen et al., 2023);
//! - **questions** it answers (doc2query - Nogueira et al., 2019).

use anyhow::Result;
use docuzent_doc::kvpool::KvPool;
use docuzent_llm::{chatml, json, Sampling};
use serde_json::Value;

use crate::corpus::Corpus;
use crate::index::{cut_at, Chunking, Entry, EntryKind, Index};

/// Chunks described in one call: enough to amortise the call, few enough that the reply stays well-formed.
/// At the reference window (smaller windows describe one at a time - [`crate::Budget::expand_batch`]). A reply cut
/// off at its token limit loses only the passage it was writing (what it finished is kept), and that passage is
/// described again on its own.
pub const EXPAND_BATCH: usize = 3;
const CUT_TOKENS: i32 = 300;

pub fn cut_prompt() -> String {
    "Where does this part of the book change scene, time, place or topic? For each new section after the first, copy out its first six to eight words, exactly as written, one per line, in order. Reply with those lines only; if the part is one continuous section, reply with exactly: NONE".to_string()
}

/// Where the text `words` begins in `text`, allowing for differences in spacing and line breaks.
pub fn locate(text: &str, words: &str) -> Option<usize> {
    let words: Vec<&str> = words.trim().trim_matches(|c: char| c == '"' || c == '\'' || c == '-' || c == '*' || c == '“' || c == '”').split_whitespace().collect();
    if words.len() < 3 {
        return None;
    }
    for take in [words.len(), 5, 4] {
        let want: Vec<&str> = words.iter().take(take.min(words.len())).copied().collect();
        // walk the text word by word, comparing the next `want.len()` words
        let spans: Vec<(usize, &str)> = text.split_whitespace().map(|w| (w.as_ptr() as usize - text.as_ptr() as usize, w)).collect();
        for i in 0..spans.len().saturating_sub(want.len() - 1) {
            if spans[i..i + want.len()].iter().zip(&want).all(|((_, a), b)| a.trim_matches(|c: char| !c.is_alphanumeric()).eq_ignore_ascii_case(b.trim_matches(|c: char| !c.is_alphanumeric()))) {
                return Some(spans[i].0);
            }
        }
    }
    None
}

/// Guided chunking: the model, with each part resident, marks where its scenes and topics change; the part is cut
/// there, and each section held to [`CHUNK_CHARS`].
pub fn guided_index(pool: &KvPool, corpus: &dyn Corpus, on_progress: &mut dyn FnMut(usize, usize)) -> Result<Index> {
    let n = corpus.part_count();
    let mut parts_chunks = Vec::with_capacity(n);
    for i in 0..n {
        on_progress(i, n);
        let prefix = corpus.part_prefix(i);
        let (c, _) = pool.with_resident(corpus.part_file(i), corpus.part_owner(i), &prefix, |llm| llm.complete(&chatml::ask(&prefix, &cut_prompt()), &Sampling::precise(CUT_TOKENS), &mut |_| {}))?;
        let text = corpus.part_text(i);
        let starts: Vec<usize> = if crate::reader::is_none(&c.text) { Vec::new() } else { c.text.lines().filter_map(|l| locate(text, l.trim_start_matches(|ch: char| ch.is_ascii_digit() || ch == '.' || ch == ')' || ch == ' '))).collect() };
        parts_chunks.push(cut_at(text, &starts, crate::Budget::for_engine(pool.engine().context_size()).chunk_chars));
    }
    on_progress(n, n);
    Ok(Index::new(Chunking::Guided, parts_chunks))
}

pub fn expand_prompt(passages: &[&str]) -> String {
    let mut p = String::from(
        "Below are passages from this part of the book, numbered. For each one, write:\n\
- \"context\": one sentence placing it in the book: who is there, where and when, what is happening and what led to it\n\
- \"people\": each named person present or mentioned, and what they do or say in it\n\
- \"setting\": the place, the time, and the objects, creatures and events it involves\n\
- \"facts\": the facts it states, each short, with names and numbers exactly as written\n\
- \"questions\": three questions a reader could ask that it answers\n\
Use only this part of the book. Inside the JSON, quote anyone's words with single quotes ('like this'), never double quotes. Reply with a JSON array, one object per passage, in order: [{\"n\": 1, \"context\": \"...\", \"people\": \"...\", \"setting\": \"...\", \"facts\": [\"...\"], \"questions\": [\"...\"]}]\n\n",
    );
    for (i, t) in passages.iter().enumerate() {
        p.push_str(&format!("[{}]\n{}\n\n", i + 1, t));
    }
    p
}

fn text_of(v: &Value) -> String {
    match v {
        Value::String(s) => s.trim().to_string(),
        Value::Array(a) => a.iter().map(text_of).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n"),
        Value::Object(o) => o.iter().map(|(k, v)| format!("{k}: {}", text_of(v))).collect::<Vec<_>>().join("\n"),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// The expansion entries a reply describes, for the chunks `ids` (in the order they were numbered). A reply that is
/// not JSON, or leaves a passage out, gives fewer entries - never wrong ones.
pub fn parse_expansions(reply: &str, ids: &[usize], chunk_texts: &[&str]) -> Vec<Entry> {
    // the whole array - or, from a reply cut off mid-way (or otherwise broken), every object it did finish
    let items: Vec<Value> = json::first_array(reply).and_then(|a| serde_json::from_str(a).ok()).unwrap_or_else(|| json::complete_objects(reply).into_iter().filter_map(|o| serde_json::from_str(o).ok()).collect());
    let mut out = Vec::new();
    for (pos, item) in items.iter().enumerate() {
        let n = item["n"].as_u64().map(|n| n as usize).filter(|n| (1..=ids.len()).contains(n)).unwrap_or(pos + 1);
        let Some(&chunk) = ids.get(n - 1) else { continue };
        let context = text_of(&item["context"]);
        if !context.is_empty() {
            out.push(Entry { chunk, kind: EntryKind::Context, text: format!("{context}\n\n{}", chunk_texts[n - 1]) });
        }
        for (key, kind) in [("people", EntryKind::People), ("setting", EntryKind::Setting), ("facts", EntryKind::Facts), ("questions", EntryKind::Questions)] {
            let t = text_of(&item[key]);
            if !t.is_empty() {
                out.push(Entry { chunk, kind, text: t });
            }
        }
    }
    out
}

/// Mode 2: every chunk described by the model (with its part resident), the descriptions added to the index as
/// entries pointing back to the chunk. `model` is recorded, so a change of model is known to call for a rebuild.
///
/// A chunk the batch reply left out (cut off at its token limit, or not JSON) is described again on its own. On an
/// index this model already expanded, only the chunks without descriptions are described - a repair, not a rebuild.
/// Returns how many chunks are still undescribed (0 when every one is).
pub fn expand(pool: &KvPool, corpus: &dyn Corpus, index: &mut Index, model: &str, on_progress: &mut dyn FnMut(usize, usize)) -> Result<usize> {
    let repair = index.made_by.as_deref() == Some(model) && index.has_expansions();
    let todo: Vec<usize> = if repair { index.unexpanded() } else { (0..index.chunks.len()).collect() };
    let mut entries: Vec<Entry> = if repair { index.entries.iter().filter(|e| e.kind.is_expansion()).cloned().collect() } else { Vec::new() };
    let total = todo.len();
    let mut done = 0;
    // passages per call and what a call may write, for this model's window (3 and 1,600 tokens at 12,288)
    let b = crate::Budget::for_engine(pool.engine().context_size());
    // described alone and still not: kept by their own text, not tried again on every load
    let mut failed_alone: Vec<usize> = Vec::new();
    for part in 0..corpus.part_count() {
        let ids: Vec<usize> = todo.iter().copied().filter(|c| index.chunks[*c].part == part).collect();
        if ids.is_empty() {
            continue;
        }
        let prefix = corpus.part_prefix(part);
        let describe = |batch: &[usize]| -> Result<Vec<Entry>> {
            let texts: Vec<&str> = batch.iter().map(|id| index.chunks[*id].text.as_str()).collect();
            let prompt = chatml::ask(&prefix, &expand_prompt(&texts));
            let (c, _) = pool.with_resident(corpus.part_file(part), corpus.part_owner(part), &prefix, |llm| llm.complete(&prompt, &Sampling::precise(b.expand_tokens), &mut |_| {}))?;
            Ok(parse_expansions(&c.text, batch, &texts))
        };
        for batch in ids.chunks(b.expand_batch) {
            on_progress(done, total);
            let got = describe(batch)?;
            let missing: Vec<usize> = batch.iter().copied().filter(|id| !got.iter().any(|e| e.chunk == *id)).collect();
            entries.extend(got);
            if batch.len() > 1 {
                for id in missing {
                    let alone = describe(&[id])?;
                    if alone.is_empty() {
                        failed_alone.push(id);
                    }
                    entries.extend(alone);
                }
            } else {
                failed_alone.extend(missing);
            }
            done += batch.len();
        }
    }
    on_progress(total, total);
    let left_out = failed_alone.len();
    if !repair {
        index.undescribable.clear();
    }
    index.undescribable.extend(failed_alone);
    index.add_expansions(entries, model);
    Ok(left_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_marked_section_start_is_found_despite_spacing_and_punctuation() {
        let text = "It was late.\n\nThe next   morning, Harry woke\nearly and went down.";
        assert_eq!(locate(text, "\"The next morning, Harry woke early\""), Some(text.find("The next").unwrap()));
        assert_eq!(locate(text, "Nothing like this here at all"), None);
        assert_eq!(locate(text, "too short"), None);
    }

    #[test]
    fn expansions_point_back_to_their_chunks_and_a_bad_reply_gives_none() {
        let reply = r#"Here you go: [{"n": 1, "context": "At the Dursleys' breakfast.", "people": "Harry: cooks bacon", "setting": "kitchen, morning", "facts": ["Dudley has 36 presents"], "questions": ["How many presents did Dudley get?"]},
                       {"n": 2, "context": "", "people": ["Hagrid: arrives"], "setting": null, "facts": [], "questions": []}]"#;
        let e = parse_expansions(reply, &[7, 8], &["chunk seven", "chunk eight"]);
        let kinds: Vec<(usize, EntryKind)> = e.iter().map(|x| (x.chunk, x.kind)).collect();
        assert_eq!(kinds, vec![(7, EntryKind::Context), (7, EntryKind::People), (7, EntryKind::Setting), (7, EntryKind::Facts), (7, EntryKind::Questions), (8, EntryKind::People)]);
        assert!(e[0].text.starts_with("At the Dursleys' breakfast.") && e[0].text.ends_with("chunk seven"), "context entries carry the chunk itself");
        assert!(parse_expansions("not json", &[1], &["x"]).is_empty());
    }

    #[test]
    fn a_reply_cut_off_mid_way_keeps_the_passages_it_finished() {
        let cut = r#"[{"n": 1, "context": "At the zoo.", "people": "Harry", "setting": "", "facts": [], "questions": []}, {"n": 2, "context": "On the train, wh"#;
        let e = parse_expansions(cut, &[4, 5], &["four", "five"]);
        assert!(!e.is_empty() && e.iter().all(|x| x.chunk == 4), "passage 1 kept, the unfinished passage 2 not invented");
    }

    #[test]
    fn the_expansion_prompt_numbers_its_passages_and_asks_for_every_kind() {
        let p = expand_prompt(&["first", "second"]);
        assert!(p.contains("[1]\nfirst") && p.contains("[2]\nsecond"));
        for k in ["context", "people", "setting", "facts", "questions"] {
            assert!(p.contains(k));
        }
    }
}
