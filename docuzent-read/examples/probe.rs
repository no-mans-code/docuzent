//! Where a saved index ranks the passages that answer a question - to tell a retrieval miss from an answering one.
//!
//!   cargo run --release -p docuzent-read --example probe -- <index dir> <embed url> <embed model> "<question>|<needle>[||<needle>...]" ...
//!
//! For each question: the chunks a search hands over (the best 8), and where the chunks holding any needle (plain
//! text, case-insensitive) rank - fused (words + meaning) and by words alone.

use docuzent_llm::OpenAiEmbedder;
use docuzent_read::{Index, Search};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let index = Index::load(std::path::Path::new(&args[0]))?.ok_or_else(|| anyhow::anyhow!("no index in {}", args[0]))?;
    let embedder = OpenAiEmbedder::new(args[1].as_str(), args[2].as_str());
    let how = if index.entries.iter().any(|e| e.kind.is_expansion()) { Search::Expanded } else { Search::TextOnly };
    println!("{} chunks, {} entries, searched {how:?}", index.chunks.len(), index.entries.len());
    for spec in &args[3..] {
        let (question, needles) = spec.split_once('|').unwrap_or((spec, ""));
        let needles: Vec<String> = needles.split("||").filter(|n| !n.is_empty()).map(str::to_lowercase).collect();
        let holds = |c: usize| needles.iter().any(|n| index.chunks[c].text.to_lowercase().contains(n));
        let fused = index.search(question, usize::MAX, how, Some(&embedder), None)?;
        let words = index.search(question, usize::MAX, how, None, None)?;
        println!("\n## {question}");
        for (rank, (c, _)) in fused.iter().take(8).enumerate() {
            let text: String = index.chunks[*c].text.chars().take(90).collect::<String>().replace('\n', " ");
            println!("  {}{:>2}. chunk {:>3} (part {:>2}) {text}", if holds(*c) { "*" } else { " " }, rank + 1, c, index.chunks[*c].part + 1);
        }
        let answering: Vec<usize> = (0..index.chunks.len()).filter(|c| holds(*c)).collect();
        for c in answering {
            let f = fused.iter().position(|(x, _)| *x == c).map_or("-".into(), |p| (p + 1).to_string());
            let w = words.iter().position(|(x, _)| *x == c).map_or("-".into(), |p| (p + 1).to_string());
            println!("  answer in chunk {c} (part {}): fused rank {f}, words rank {w}", index.chunks[c].part + 1);
        }
    }
    Ok(())
}
