//! Queries a local Ollama server for a model's real context window, rather
//! than hardcoding one - `/api/show`'s `model_info` carries it under a
//! family-prefixed key (`"qwen2.context_length"`, `"llama.context_length"`,
//! ...), so this looks for any key ending in `.context_length` instead of
//! guessing the family.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct ShowResponse {
    #[serde(default)]
    model_info: serde_json::Map<String, Value>,
    #[serde(default)]
    details: Option<Details>,
}

#[derive(Deserialize)]
struct Details {
    #[serde(default)]
    parameter_size: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelInfo {
    pub context_length: u32,
    /// e.g. "3.1B" - as Ollama reports it, not parsed further.
    pub parameter_size: Option<String>,
}

/// Fetches `model`'s real context window (and parameter size, if Ollama
/// reports one) from `host`'s `/api/show`.
pub fn model_info(host: &str, model: &str) -> Result<ModelInfo> {
    let url = format!("{}/api/show", host.trim_end_matches('/'));
    let resp: ShowResponse = ureq::post(&url)
        .send_json(serde_json::json!({ "model": model }))
        .with_context(|| format!("ollama show request to {url} failed - is `ollama serve` running?"))?
        .into_json()
        .context("failed to parse ollama show response")?;

    let context_length = resp
        .model_info
        .iter()
        .find(|(k, _)| k.ends_with(".context_length"))
        .and_then(|(_, v)| v.as_u64())
        .with_context(|| format!("no *.context_length field in model_info for `{model}`"))?;
    if context_length == 0 || context_length > u32::MAX as u64 {
        bail!("implausible context_length {context_length} reported for `{model}`");
    }

    Ok(ModelInfo {
        context_length: context_length as u32,
        parameter_size: resp.details.and_then(|d| d.parameter_size),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_context_length_regardless_of_family_prefix() {
        let mut model_info_map = serde_json::Map::new();
        model_info_map.insert("qwen2.context_length".to_string(), Value::from(32768));
        model_info_map.insert("qwen2.embedding_length".to_string(), Value::from(2048));
        let resp = ShowResponse {
            model_info: model_info_map,
            details: Some(Details { parameter_size: Some("3.1B".to_string()) }),
        };
        let context_length = resp
            .model_info
            .iter()
            .find(|(k, _)| k.ends_with(".context_length"))
            .and_then(|(_, v)| v.as_u64())
            .unwrap();
        assert_eq!(context_length, 32768);
        assert_eq!(resp.details.unwrap().parameter_size, Some("3.1B".to_string()));
    }
}
