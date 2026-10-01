//! Retrieval alone, without a model: for each question, does what a search hands over contain what a right answer
//! must say? Separates a retrieval miss from an answering one, and compares ways of searching in seconds.
//!
//!   retrieval --index <index dir> --questions <set.json> [--embed-url URL] [--embed-model M]
//!
//! A question's evidence is its `must` patterns and its `any_of` group (as one): the share of them found in the text
//! handed over is its coverage; "complete" when every one is there. Questions with no evidence (the ones the book
//! cannot answer) are skipped. Patterns that are worked out rather than quoted (a sum, a number in figures where the
//! book spells it) count as missing for every variant alike - the comparison between variants is what this is for.

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use docuzent_llm::{cosine_similarity, Embedder, OpenAiEmbedder};
use docuzent_read::modes::{with_neighbours, Found};
use docuzent_read::{Index, Search, Shelf};
use regex::RegexBuilder;
use serde::Deserialize;

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    index: PathBuf,
    #[arg(long)]
    questions: PathBuf,
    #[arg(long, default_value = "http://embed:8080")]
    embed_url: String,
    #[arg(long, default_value = "nomic-embed-text")]
    embed_model: String,
}

#[derive(Deserialize)]
struct Set {
    book: String,
    questions: Vec<Question>,
}

#[derive(Deserialize)]
struct Question {
    q: String,
    #[serde(default)]
    must: Vec<String>,
    #[serde(default)]
    any_of: Vec<String>,
}

/// Chunks best first: reciprocal-rank fusion of a words and a meaning ranking with these weights.
fn fuse(words: &[usize], meaning: &[usize], ww: f64, wm: f64) -> Vec<usize> {
    let mut s: HashMap<usize, f64> = HashMap::new();
    for (r, c) in words.iter().enumerate() {
        *s.entry(*c).or_default() += ww / (60.0 + r as f64 + 1.0);
    }
    for (r, c) in meaning.iter().enumerate() {
        *s.entry(*c).or_default() += wm / (60.0 + r as f64 + 1.0);
    }
    let mut v: Vec<(usize, f64)> = s.into_iter().collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
    v.into_iter().map(|(c, _)| c).collect()
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let index = Index::load(&cli.index)?.with_context(|| format!("no index in {}", cli.index.display()))?;
    let set: Set = serde_json::from_slice(&std::fs::read(&cli.questions)?)?;
    let embedder = OpenAiEmbedder::new(cli.embed_url.as_str(), cli.embed_model.as_str());
    let expanded = index.entries.iter().any(|e| e.kind.is_expansion());
    let how = if expanded { Search::Expanded } else { Search::TextOnly };
    let shelves = [Shelf { index: &index, first_part: 0 }];
    let re = |p: &str| RegexBuilder::new(p).case_insensitive(true).build();

    // variant name -> (sum of coverage, complete count)
    let variants = ["fused (now)", "words only", "meaning only", "words x2", "meaning x2", "fused, 12 chunks", "fused + neighbours", "words x2 + neighbours"];
    let mut score: Vec<(f64, usize)> = vec![(0.0, 0); variants.len()];
    let mut asked = 0;
    let mut misses: Vec<String> = Vec::new();
    for q in &set.questions {
        let mut evidence: Vec<String> = q.must.clone();
        if !q.any_of.is_empty() {
            evidence.push(q.any_of.join("|"));
        }
        // a question the book cannot answer has nothing to find ("the document does not say" is the right answer)
        if evidence.is_empty() || evidence.iter().any(|p| p.contains("does not") || p.contains("not (say")) {
            continue;
        }
        asked += 1;
        let words: Vec<usize> = index.search(&q.q, usize::MAX, how, None, None)?.into_iter().map(|(c, _)| c).collect();
        let qv = embedder.embed_query(&q.q)?;
        let mut best: HashMap<usize, f32> = HashMap::new();
        for (e, v) in index.entries.iter().zip(&index.vectors) {
            if expanded || !e.kind.is_expansion() {
                let s = cosine_similarity(&qv, v);
                let b = best.entry(e.chunk).or_insert(f32::MIN);
                *b = b.max(s);
            }
        }
        let mut meaning: Vec<(usize, f32)> = best.into_iter().collect();
        meaning.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        let meaning: Vec<usize> = meaning.into_iter().map(|(c, _)| c).collect();

        let texts = |chunks: &[usize], neighbours: bool| -> String {
            let found: Vec<Found> = chunks.iter().map(|c| Found { part: index.chunks[*c].part, text: index.chunks[*c].text.clone(), shelf: 0, chunk: *c }).collect();
            let found = if neighbours { with_neighbours(&shelves, &found, docuzent_read::modes::NEIGHBOUR_CHARS) } else { found };
            found.into_iter().map(|f| f.text).collect::<Vec<_>>().join("\n\n")
        };
        let take = |v: Vec<usize>, k: usize| v.into_iter().take(k).collect::<Vec<_>>();
        let handed: Vec<String> = vec![
            texts(&take(fuse(&words, &meaning, 1.0, 1.0), 8), false),
            texts(&take(words.clone(), 8), false),
            texts(&take(meaning.clone(), 8), false),
            texts(&take(fuse(&words, &meaning, 2.0, 1.0), 8), false),
            texts(&take(fuse(&words, &meaning, 1.0, 2.0), 8), false),
            texts(&take(fuse(&words, &meaning, 1.0, 1.0), 12), false),
            texts(&take(fuse(&words, &meaning, 1.0, 1.0), 8), true),
            texts(&take(fuse(&words, &meaning, 2.0, 1.0), 8), true),
        ];
        for (i, text) in handed.iter().enumerate() {
            let found = evidence.iter().filter(|p| re(p).map(|r| r.is_match(text)).unwrap_or(false)).count();
            score[i].0 += found as f64 / evidence.len() as f64;
            score[i].1 += (found == evidence.len()) as usize;
            if i == 0 && found < evidence.len() {
                misses.push(q.q.clone());
            }
        }
    }
    println!("## {} - {} chunks, {} index; {asked} questions with evidence", set.book, index.chunks.len(), if expanded { "expanded" } else { "plain" });
    println!("| variant | evidence found | complete |\n|---|---|---|");
    for (i, v) in variants.iter().enumerate() {
        println!("| {v} | {:.0}% | {}/{asked} |", 100.0 * score[i].0 / asked.max(1) as f64, score[i].1);
    }
    println!("incomplete now: {}", misses.join(" / "));
    Ok(())
}
