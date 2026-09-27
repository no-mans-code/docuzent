//! Turns chunk text into vectors. An `Embedder` trait keeps storage and
//! retrieval decoupled from any one backend; [`OllamaEmbedder`] is the
//! default, matching this project's "local, no external API keys" design.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

pub trait Embedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// Calls a local Ollama server's `/api/embeddings` endpoint.
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
}

/// Cosine similarity between two vectors; 0.0 for mismatched lengths or a
/// zero vector rather than panicking or dividing by zero.
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
}
