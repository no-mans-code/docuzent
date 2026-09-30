//! Text into vectors, for retrieval. An [`Embedder`] keeps storage and search decoupled from any one backend:
//!
//! * [`OpenAiEmbedder`] - the OpenAI-compatible `/v1/embeddings` endpoint, which both Ollama and llama.cpp's server
//!   (started with `--embeddings`) provide. Batched. The default.
//! * [`OllamaEmbedder`] - Ollama's own older `/api/embeddings`, one text per call (what docuzent-core has always used).
//! * [`SimEmbedder`] - deterministic hashed bag-of-words vectors, for tests without a model.
//!
//! Some embedding models are trained with a task prefix and retrieve markedly worse without it: nomic-embed-text
//! wants `search_document: ` on what is stored and `search_query: ` on what is searched for. [`Embedder::embed_query`]
//! and [`Embedder::embed_documents`] add it when the model is one of those.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub trait Embedder: Send + Sync {
    /// One text, as is.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Several texts. The default embeds them one by one; a backend with a batch endpoint overrides it.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Which model makes the vectors - an index records it, and is rebuilt when it changes (vectors from two
    /// models are not comparable).
    fn id(&self) -> String {
        "unknown".into()
    }

    /// Texts to be stored and searched, with the model's document prefix if it has one.
    fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match task_prefixes(&self.id()) {
            Some((doc, _)) => self.embed_batch(&texts.iter().map(|t| format!("{doc}{t}")).collect::<Vec<_>>()),
            None => self.embed_batch(texts),
        }
    }

    /// A question, with the model's query prefix if it has one.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        match task_prefixes(&self.id()) {
            Some((_, q)) => self.embed(&format!("{q}{text}")),
            None => self.embed(text),
        }
    }
}

/// The document and query prefixes a model was trained with, if any.
pub fn task_prefixes(model: &str) -> Option<(&'static str, &'static str)> {
    let m = model.to_ascii_lowercase();
    if m.contains("nomic-embed") {
        Some(("search_document: ", "search_query: "))
    } else if m.contains("mxbai-embed") {
        Some(("", "Represent this sentence for searching relevant passages: "))
    } else {
        None
    }
}

/// `/v1/embeddings` - Ollama (`http://host:11434`) and llama.cpp's server with `--embeddings` both speak it.
pub struct OpenAiEmbedder {
    base: String,
    model: String,
    agent: ureq::Agent,
    batch: usize,
}

impl OpenAiEmbedder {
    pub fn new(base: impl Into<String>, model: impl Into<String>) -> Self {
        let agent = ureq::AgentBuilder::new().timeout_connect(std::time::Duration::from_secs(5)).timeout_read(std::time::Duration::from_secs(300)).build();
        Self { base: base.into().trim_end_matches('/').to_string(), model: model.into(), agent, batch: 32 }
    }
}

impl Embedder for OpenAiEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.embed_batch(&[text.to_string()])?.pop().unwrap_or_default())
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.batch.max(1)) {
            let url = format!("{}/v1/embeddings", self.base);
            let v: Value = self
                .agent
                .post(&url)
                .send_json(json!({ "model": self.model, "input": batch }))
                .map_err(|e| anyhow::anyhow!("embeddings request to {url} failed ({e}) - is the embedding model `{}` available there?", self.model))?
                .into_json()
                .context("the embeddings response was not JSON")?;
            let mut data: Vec<(usize, Vec<f32>)> = v["data"]
                .as_array()
                .context("the embeddings response has no `data`")?
                .iter()
                .enumerate()
                .map(|(i, d)| (d["index"].as_u64().map(|n| n as usize).unwrap_or(i), d["embedding"].as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).map(|x| x as f32).collect()).unwrap_or_default()))
                .collect();
            data.sort_by_key(|(i, _)| *i);
            anyhow::ensure!(data.len() == batch.len(), "asked for {} embeddings, got {}", batch.len(), data.len());
            out.extend(data.into_iter().map(|(_, e)| e));
        }
        Ok(out)
    }

    fn id(&self) -> String {
        self.model.clone()
    }
}

/// Calls a local Ollama server's `/api/embeddings` endpoint, one text at a time.
pub struct OllamaEmbedder {
    host: String,
    model: String,
}

impl OllamaEmbedder {
    pub fn new(host: impl Into<String>, model: impl Into<String>) -> Self {
        Self { host: host.into(), model: model.into() }
    }

    /// `nomic-embed-text` on the default local Ollama endpoint.
    pub fn default_local() -> Self {
        Self::new("http://localhost:11434", "nomic-embed-text")
    }
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

#[derive(Deserialize)]
struct EmbedResponse {
    embedding: Vec<f32>,
}

impl Embedder for OllamaEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.host.trim_end_matches('/'));
        let resp: EmbedResponse = ureq::post(&url)
            .send_json(EmbedRequest { model: &self.model, prompt: text })
            .with_context(|| format!("ollama embeddings request to {url} failed - is `ollama serve` running?"))?
            .into_json()
            .context("failed to parse ollama embeddings response")?;
        Ok(resp.embedding)
    }

    fn id(&self) -> String {
        self.model.clone()
    }
}

/// Deterministic vectors without a model: each word hashed into one of `dims` buckets, then normalised. Texts
/// sharing words are similar, which is all a test needs.
pub struct SimEmbedder {
    pub dims: usize,
}

impl Default for SimEmbedder {
    fn default() -> Self {
        Self { dims: 64 }
    }
}

impl Embedder for SimEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = vec![0f32; self.dims];
        for w in text.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2) {
            let h = w.bytes().fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211));
            v[(h % self.dims as u64) as usize] += 1.0;
        }
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            v.iter_mut().for_each(|x| *x /= n);
        }
        Ok(v)
    }

    fn id(&self) -> String {
        format!("sim-{}", self.dims)
    }
}

/// Cosine similarity between two vectors; 0.0 for mismatched lengths or a zero vector rather than panicking or
/// dividing by zero.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    #[test]
    fn identical_vectors_have_similarity_one() {
        let v = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn orthogonal_vectors_have_similarity_zero() {
        assert!((cosine_similarity(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
    }

    #[test]
    fn mismatched_lengths_are_zero_not_a_panic() {
        assert_eq!(cosine_similarity(&[1.0, 2.0], &[1.0]), 0.0);
    }

    #[test]
    fn the_simulated_embedder_makes_texts_that_share_words_close() {
        let e = SimEmbedder::default();
        let a = e.embed("the house cup feast points").unwrap();
        let b = e.embed("points at the house cup").unwrap();
        let c = e.embed("a dragon egg in the hut").unwrap();
        assert!(cosine_similarity(&a, &b) > cosine_similarity(&a, &c));
    }

    #[test]
    fn a_model_trained_with_task_prefixes_gets_them() {
        assert_eq!(task_prefixes("nomic-embed-text:latest"), Some(("search_document: ", "search_query: ")));
        assert_eq!(task_prefixes("bge-m3"), None);
    }

    #[test]
    fn the_openai_endpoint_is_asked_in_batches_and_answers_are_put_back_in_order() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { break };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 8192];
                loop {
                    let n = s.read(&mut chunk).unwrap_or(0);
                    buf.extend_from_slice(&chunk[..n]);
                    let t = String::from_utf8_lossy(&buf).to_string();
                    if let Some(p) = t.find("\r\n\r\n") {
                        let len = t[..p].to_lowercase().lines().find_map(|l| l.strip_prefix("content-length:").and_then(|v| v.trim().parse::<usize>().ok())).unwrap_or(0);
                        if buf.len() >= p + 4 + len {
                            break;
                        }
                    }
                    if n == 0 {
                        break;
                    }
                }
                let t = String::from_utf8_lossy(&buf).to_string();
                let body: Value = serde_json::from_str(&t[t.find("\r\n\r\n").unwrap() + 4..]).unwrap();
                let inputs = body["input"].as_array().unwrap().len();
                // answered out of order, as a server may
                let data: Vec<Value> = (0..inputs).rev().map(|i| json!({ "index": i, "embedding": [i as f32, 1.0] })).collect();
                let payload = json!({ "data": data }).to_string();
                let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}", payload.len());
            }
        });
        let mut e = OpenAiEmbedder::new(&base, "nomic-embed-text");
        e.batch = 2;
        let texts: Vec<String> = (0..5).map(|i| format!("text {i}")).collect();
        let v = e.embed_batch(&texts).unwrap();
        assert_eq!(v.len(), 5);
        assert_eq!(v.iter().map(|x| x[0]).collect::<Vec<_>>(), vec![0.0, 1.0, 0.0, 1.0, 0.0], "each batch back in order");
    }
}
