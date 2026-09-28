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

#[derive(Debug, Clone, PartialEq)]
pub struct ModelInfo {
    /// The window Ollama reports via `*.context_length` - for many models
    /// this is a RoPE-scaling extrapolation far beyond what the model was
    /// actually trained on, not a safe default to use as-is. See
    /// `original_context_length`, and
    /// https://github.com/no-mans-code/docuzent/issues/30.
    pub context_length: u32,
    /// The model's real trained context, from
    /// `*.rope.scaling.original_context_length`, when the model reports
    /// one (only models using RoPE scaling do - e.g. YaRN). `None` for a
    /// model that doesn't report this field, in which case
    /// `context_length` itself is the best available number.
    pub original_context_length: Option<u32>,
    /// The real per-head attention dimension, from
    /// `*.attention.key_length`, when the model reports it directly.
    /// `None` for a model that doesn't (in which case
    /// `embedding_length / head_count` is the best available
    /// approximation - what `ollama_kv_profiler::ollama::Client::
    /// architecture_info` falls back to). This exists because that
    /// approximation is measurably *wrong* for at least two real,
    /// currently-relevant architectures: confirmed via real `/api/show`
    /// payloads, `mistral3` (devstral) reports `embedding_length` 5120 /
    /// `head_count` 32 = a derived 160, but its own `attention.key_length`
    /// is 128 (a 25% KV-cache overestimate); `gemma4` derives 176 but
    /// reports 512 (a 2.9x underestimate). Averaged across a per-layer
    /// array the same way `original_context_length` is, for architectures
    /// that vary it per layer.
    pub attention_key_length: Option<f64>,
    /// e.g. "3.1B" - as Ollama reports it, not parsed further.
    pub parameter_size: Option<String>,
}

#[derive(Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagsModel>,
}

#[derive(Deserialize)]
struct TagsModel {
    name: String,
    #[serde(default)]
    size: u64,
}

/// Lists every model this Ollama server has locally pulled - for a UI to
/// offer a real choice from, rather than a model being fixed at startup.
pub fn list_models(host: &str) -> Result<Vec<String>> {
    Ok(list_tags(host)?.into_iter().map(|m| m.name).collect())
}

/// `model`'s real on-disk weight size in bytes, as Ollama itself reports
/// it via `/api/tags` - the dominant term in how much VRAM a model needs
/// beyond its KV cache, and the only place this is available without
/// loading the model first (`/api/show` doesn't include it).
pub fn model_weight_bytes(host: &str, model: &str) -> Result<u64> {
    list_tags(host)?
        .into_iter()
        .find(|m| m.name == model)
        .map(|m| m.size)
        .with_context(|| format!("`{model}` is not in `/api/tags` - has it been pulled?"))
}

fn list_tags(host: &str) -> Result<Vec<TagsModel>> {
    let url = format!("{}/api/tags", host.trim_end_matches('/'));
    let resp: TagsResponse = ureq::get(&url)
        .call()
        .with_context(|| format!("ollama tags request to {url} failed - is `ollama serve` running?"))?
        .into_json()
        .context("failed to parse ollama tags response")?;
    Ok(resp.models)
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
    parse_show_response(resp, model)
}

/// The pure, live-server-free half of [`model_info`] - separated out so
/// real `/api/show` payloads (captured once from a live server) can be
/// replayed as regression tests without needing Ollama running.
fn parse_show_response(resp: ShowResponse, model: &str) -> Result<ModelInfo> {
    let context_length = resp
        .model_info
        .iter()
        .find(|(k, _)| k.ends_with(".context_length"))
        .and_then(|(_, v)| v.as_u64())
        .with_context(|| format!("no *.context_length field in model_info for `{model}`"))?;
    if context_length == 0 || context_length > u32::MAX as u64 {
        bail!("implausible context_length {context_length} reported for `{model}`");
    }

    let original_context_length = resp
        .model_info
        .iter()
        .find(|(k, _)| k.ends_with(".rope.scaling.original_context_length"))
        .and_then(|(_, v)| v.as_u64())
        .filter(|&n| n > 0 && n <= u32::MAX as u64)
        .map(|n| n as u32);

    let attention_key_length = resp
        .model_info
        .iter()
        .find(|(k, _)| k.ends_with(".attention.key_length"))
        .and_then(|(_, v)| avg_of_value(v));

    Ok(ModelInfo {
        context_length: context_length as u32,
        original_context_length,
        attention_key_length,
        parameter_size: resp.details.and_then(|d| d.parameter_size),
    })
}

/// A field can be one scalar or a per-layer array (some architectures vary
/// it per layer - see `attention_key_length`'s doc comment); this reads
/// either shape and returns the average, matching how `original_context_length`-
/// style scalar fields and per-layer arrays both need handling uniformly.
fn avg_of_value(v: &Value) -> Option<f64> {
    if let Some(n) = v.as_u64() {
        return Some(n as f64);
    }
    let arr = v.as_array()?;
    if arr.is_empty() {
        return None;
    }
    let sum: u64 = arr.iter().filter_map(|x| x.as_u64()).sum();
    Some(sum as f64 / arr.len() as f64)
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

    /// Real field names/values captured from a live `/api/show` for
    /// `devstral-small-2:24b` - the exact model that motivated
    /// `original_context_length` in the first place (see
    /// https://github.com/no-mans-code/docuzent/issues/30): it reports a
    /// nominal 393,216-token window, but was actually trained on 8,192.
    #[test]
    fn parses_original_context_length_for_a_real_yarn_scaled_model() {
        let mut model_info_map = serde_json::Map::new();
        model_info_map.insert("mistral3.context_length".to_string(), Value::from(393_216u64));
        model_info_map.insert("mistral3.rope.scaling.original_context_length".to_string(), Value::from(8192u64));
        model_info_map.insert("mistral3.rope.scaling.type".to_string(), Value::from("yarn"));
        let resp = ShowResponse { model_info: model_info_map, details: None };

        let info = parse_show_response(resp, "devstral-small-2:24b").unwrap();

        assert_eq!(info.context_length, 393_216);
        assert_eq!(info.original_context_length, Some(8192));
    }

    /// Real field values from the same live devstral payload: its real
    /// per-head attention dimension (128) differs from what
    /// `embedding_length / head_count` (5120 / 32 = 160) would derive -
    /// see `ModelInfo::attention_key_length`'s doc comment for why this
    /// field exists at all.
    #[test]
    fn parses_attention_key_length_when_a_model_reports_it_directly() {
        let mut model_info_map = serde_json::Map::new();
        model_info_map.insert("mistral3.context_length".to_string(), Value::from(393_216u64));
        model_info_map.insert("mistral3.attention.key_length".to_string(), Value::from(128u64));
        let resp = ShowResponse { model_info: model_info_map, details: None };

        let info = parse_show_response(resp, "devstral-small-2:24b").unwrap();

        assert_eq!(info.attention_key_length, Some(128.0));
    }

    /// gemma4's real interleaved local/global attention reports
    /// `head_count_kv` as a per-layer array in real `/api/show` output -
    /// `attention_key_length` (and any similar field) must average an
    /// array rather than fail to parse it, the same convention
    /// `ollama_kv_profiler::ollama::Client::architecture_info` uses.
    #[test]
    fn averages_a_per_layer_array_field_instead_of_failing_to_parse_it() {
        let mut model_info_map = serde_json::Map::new();
        model_info_map.insert("gemma4.context_length".to_string(), Value::from(262_144u64));
        model_info_map.insert(
            "gemma4.attention.key_length".to_string(),
            Value::from(vec![512u64, 512, 256]), // e.g. mixed global/local layers
        );
        let resp = ShowResponse { model_info: model_info_map, details: None };

        let info = parse_show_response(resp, "gemma4:26b").unwrap();

        assert_eq!(info.attention_key_length, Some((512.0 + 512.0 + 256.0) / 3.0));
    }

    /// Most models don't use RoPE scaling at all - `original_context_length`
    /// must stay `None` rather than fabricating a value, so callers can
    /// tell "not reported" apart from "reported and equal to the nominal
    /// window."
    #[test]
    fn original_context_length_is_none_when_the_model_does_not_report_it() {
        let mut model_info_map = serde_json::Map::new();
        model_info_map.insert("qwen2.context_length".to_string(), Value::from(32768u64));
        let resp = ShowResponse { model_info: model_info_map, details: None };

        let info = parse_show_response(resp, "qwen2.5:3b").unwrap();

        assert_eq!(info.context_length, 32768);
        assert_eq!(info.original_context_length, None);
    }
}
