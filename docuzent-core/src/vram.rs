//! Estimates whether a model + context size fits in free VRAM, and picks
//! a default context length that doesn't need a manual override to be
//! safe - see https://github.com/no-mans-code/docuzent/issues/30 for the
//! real, already-observed failure this exists to prevent: trusting
//! `devstral-small-2:24b`'s nominal 393,216-token window (its real
//! trained context is 8,192 - the rest is RoPE-scaling extrapolation)
//! sized a single prime call to need roughly 64GB of KV cache alone,
//! which crashed Ollama outright rather than just answering worse.
//!
//! Two layers, deliberately, because a *prediction* and a *measurement*
//! answer different questions:
//! - Predictive ([`estimate_for_model`]): before loading anything, "would
//!   this context size fit in free VRAM right now, and what's a safe
//!   default." Built from real GGUF architecture metadata Ollama reports
//!   (via `ollama_kv_profiler::ollama::Client::architecture_info` and
//!   `ollama_kv_profiler::hardware::kv_bytes_per_token` - reused, not
//!   reimplemented, matching how `Session`'s adaptive mode already reuses
//!   that project's `predictor`/`hardware` modules) and the model's real
//!   on-disk weight size. It's an estimate, not ground truth: Ollama's
//!   KV-cache dtype is assumed f16 (its default), the VRAM headroom
//!   Ollama needs beyond weights+KV is a fixed guessed margin, and free
//!   VRAM can change between this estimate and a real load.
//! - Ground truth ([`real_vram_fraction`]): after a real call, Ollama's
//!   own `/api/ps` reports exactly how much of the *currently loaded*
//!   model sits in VRAM vs. was pushed to system RAM - no estimation
//!   involved, this is what actually happened.

use anyhow::Result;
use ollama_kv_profiler::hardware;
use ollama_kv_profiler::ollama::Client as ProfilerClient;

use crate::model_info;

/// Ollama's default KV-cache element size (f16) - 2 bytes per value.
/// Assumed, not queried: Ollama doesn't expose which KV-cache dtype a
/// given server is configured with over its API. A server configured
/// with `OLLAMA_KV_CACHE_TYPE=q8_0` or smaller will have a real footprint
/// below this estimate - a conservative bias (overestimates VRAM need),
/// not an optimistic one.
const KV_CACHE_DTYPE_BYTES: u32 = 2;

/// A fixed guess at the VRAM Ollama needs beyond raw weights + KV cache
/// (CUDA context, activation buffers, etc.) - not measured per model,
/// just a safety margin so the estimate doesn't undercount right at the
/// edge. Deliberately round and conservative.
pub const FIXED_OVERHEAD_BYTES: u64 = 1_000_000_000;

/// How much of a model's real, reported VRAM to treat as "usable" when
/// picking a default context size - leaves headroom for the OS/desktop/
/// other processes rather than planning to consume 100% of a card most
/// systems already have something else running on.
pub const DEFAULT_VRAM_SAFETY_FRACTION: f64 = 0.85;

/// Never shrink the safe default below this, regardless of how little
/// free VRAM is measured - a context this small is still usable (map-
/// reduce chunks accordingly), and a near-zero window isn't a helpful
/// "safe" answer.
pub const MIN_SAFE_CONTEXT_LENGTH: u32 = 4096;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelVramEstimate {
    /// The raw window Ollama reports via `*.context_length` - may be a
    /// RoPE-scaling extrapolation far beyond the model's real trained
    /// context (see `original_context_length`). This is the slider's
    /// absolute maximum in the web UI - available for an explicit
    /// override, never used silently as the default.
    pub nominal_context_length: u32,
    /// The model's real trained context, when reported (see
    /// `docuzent_core::model_info::ModelInfo::original_context_length`).
    pub original_context_length: Option<u32>,
    /// The recommended default: bounded above by `original_context_length`
    /// (when known) and by what fits in `vram_free_bytes` at
    /// `DEFAULT_VRAM_SAFETY_FRACTION`, floored at
    /// [`MIN_SAFE_CONTEXT_LENGTH`].
    pub safe_context_length: u32,
    /// The model's real on-disk weight size (`/api/tags`), the dominant
    /// term in VRAM need before any KV cache is counted.
    pub weight_bytes: u64,
    /// KV-cache bytes per token of context, from real architecture
    /// metadata - multiply by a candidate context length to get that
    /// context's KV-cache footprint.
    pub kv_bytes_per_token: u64,
    /// Real, currently-free VRAM in bytes - `None` when it can't be
    /// measured (no NVIDIA GPU, or `nvidia-smi` unavailable; see
    /// [`vram_free_bytes`]). When `None`, `safe_context_length` falls
    /// back to `original_context_length.unwrap_or(nominal_context_length)`
    /// - there's nothing to safely compare an estimate against.
    pub vram_free_bytes: Option<u64>,
}

/// Real (`nvidia-smi`-backed) free VRAM in bytes - NVIDIA-only, `None` on
/// any other GPU vendor or if `nvidia-smi` isn't on `PATH`. The same
/// underlying command as `ollama_kv_profiler::hardware::vram_free_fraction`
/// (which returns a *fraction*) - this project also needs an absolute
/// byte budget to compare an absolute VRAM-need estimate against, so this
/// reads both numbers directly rather than trying to recover bytes from a
/// fraction with no total to multiply against.
pub fn vram_free_bytes() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let (used, total) = text.trim().lines().next()?.split_once(',')?;
    let used: u64 = used.trim().parse().ok()?;
    let total: u64 = total.trim().parse().ok()?;
    Some(total.saturating_sub(used) * 1024 * 1024) // nvidia-smi reports MiB
}

/// The largest context length, at or below `ceiling`, whose estimated
/// VRAM need (`weight_bytes + kv_bytes_per_token * tokens +
/// FIXED_OVERHEAD_BYTES`) fits within `vram_safety_fraction` of
/// `free_bytes` - floored at [`MIN_SAFE_CONTEXT_LENGTH`]. Pure function
/// (no I/O) so it's directly unit-testable against hand-computed numbers.
fn fit_context_length(ceiling: u32, weight_bytes: u64, kv_bytes_per_token: u64, free_bytes: u64, vram_safety_fraction: f64) -> u32 {
    if kv_bytes_per_token == 0 {
        return ceiling;
    }
    let budget = (free_bytes as f64 * vram_safety_fraction) as u64;
    let available_for_kv = budget.saturating_sub(weight_bytes).saturating_sub(FIXED_OVERHEAD_BYTES);
    let max_tokens_for_vram = available_for_kv / kv_bytes_per_token;
    ceiling.min(max_tokens_for_vram.max(MIN_SAFE_CONTEXT_LENGTH as u64) as u32)
}

/// Full VRAM picture for `model` on `host`: its real nominal/trained
/// context lengths, real weight size, real per-token KV-cache cost, real
/// free VRAM (if measurable), and the safe default context length that
/// falls out of combining all of those. One function, one round-trip set,
/// reused by the CLI's default-selection, the web UI's slider estimate,
/// and the MCP server's startup - see module docs for why this isn't
/// reimplemented three times.
pub fn estimate_for_model(host: &str, model: &str) -> Result<ModelVramEstimate> {
    let info = model_info::model_info(host, model)?;
    let weight_bytes = model_info::model_weight_bytes(host, model).unwrap_or(0);
    let arch = ProfilerClient::new(host).architecture_info(model)?;
    // `arch.head_dim` (embedding_length / head_count) is a real,
    // measurable underestimate or overestimate for at least two current
    // architectures - see `ModelInfo::attention_key_length`'s doc comment
    // for the real numbers (mistral3/devstral: 160 derived vs. 128 real;
    // gemma4: 176 derived vs. 512 real). Prefer the model's own directly-
    // reported value when it has one.
    let head_dim = info.attention_key_length.unwrap_or(arch.head_dim);
    let kv_bytes_per_token = hardware::kv_bytes_per_token(arch.num_layers, arch.num_kv_heads, head_dim, KV_CACHE_DTYPE_BYTES);

    let trained_ceiling = info.original_context_length.unwrap_or(info.context_length);
    let free_bytes = vram_free_bytes();
    let safe_context_length = match free_bytes {
        Some(free) => fit_context_length(trained_ceiling, weight_bytes, kv_bytes_per_token, free, DEFAULT_VRAM_SAFETY_FRACTION),
        None => trained_ceiling,
    };

    Ok(ModelVramEstimate {
        nominal_context_length: info.context_length,
        original_context_length: info.original_context_length,
        safe_context_length,
        weight_bytes,
        kv_bytes_per_token,
        vram_free_bytes: free_bytes,
    })
}

/// Picks the context length a session should actually open with: an
/// explicit `override_value` if the caller gave one (full user control,
/// bypassing every safety check here - see
/// https://github.com/no-mans-code/docuzent/issues/29), otherwise
/// [`estimate_for_model`]'s VRAM/trained-context-aware safe default.
/// Returns the full estimate alongside the chosen value either way, so a
/// caller can report/log the real numbers even when they were overridden.
pub fn resolve_context_length(host: &str, model: &str, override_value: Option<u32>) -> Result<(u32, ModelVramEstimate)> {
    let estimate = estimate_for_model(host, model)?;
    let context_length = override_value.unwrap_or(estimate.safe_context_length);
    Ok((context_length, estimate))
}

/// Ground truth, from Ollama's own `/api/ps`: what fraction of `model`,
/// as currently loaded, actually sits in VRAM (1.0 = fully resident, 0.0
/// = fully offloaded to system RAM). `None` if the model isn't currently
/// loaded (nothing to measure yet - Ollama loads lazily on first real
/// call) or `/api/ps` is unreachable.
pub fn real_vram_fraction(host: &str, model: &str) -> Option<f64> {
    ProfilerClient::new(host).vram_fraction(model).ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real numbers from a live `/api/show` for `devstral-small-2:24b`:
    /// block_count=40, head_count_kv=8, key/value_length=128 -> f16 KV
    /// cache = 2*40*8*128*2 = 163,840 bytes/token. Real weight size
    /// (`/api/tags`) ~15.18 GB. This is the exact real case that caused
    /// the crash issue #30 exists to prevent: at the nominal 393,216
    /// window this needs ~64GB of KV cache alone.
    #[test]
    fn devstral_nominal_context_would_need_about_64gb_of_kv_cache_alone() {
        let kv_bytes_per_token = hardware::kv_bytes_per_token(40, 8.0, 128.0, KV_CACHE_DTYPE_BYTES);
        assert_eq!(kv_bytes_per_token, 163_840);
        let kv_at_nominal = kv_bytes_per_token * 393_216;
        assert!(kv_at_nominal > 64_000_000_000, "expected ~64GB, got {kv_at_nominal}");
    }

    /// The same real devstral numbers, fit against a card with real
    /// headroom (a 24GB card) - the trained ceiling (8192, from
    /// `rope.scaling.original_context_length`) fits comfortably there, so
    /// the safe context length should land exactly at it, not be
    /// artificially shrunk further.
    #[test]
    fn devstral_trained_context_fits_comfortably_on_a_24gb_card() {
        let weight_bytes = 15_177_374_099u64; // real /api/tags size for devstral-small-2:24b
        let kv_bytes_per_token = hardware::kv_bytes_per_token(40, 8.0, 128.0, KV_CACHE_DTYPE_BYTES);
        let free_bytes = 22_000_000_000u64; // ~22GB free on a 24GB card
        let safe = fit_context_length(8192, weight_bytes, kv_bytes_per_token, free_bytes, DEFAULT_VRAM_SAFETY_FRACTION);
        assert_eq!(safe, 8192, "the real trained context should fit without needing to shrink further");
    }

    /// The real, already-observed case from this project's own prior
    /// testing (see the README's "hardware limits" finding): on a 16GB
    /// card, devstral-small-2:24b's ~15.18GB of weights alone leave too
    /// little of the 85%-safety-margin budget for *any* meaningful KV
    /// cache - even the model's real trained context (8192) doesn't fit
    /// with headroom to spare. The formula must shrink all the way to the
    /// floor here, not silently claim 8192 is safe when it measurably
    /// wasn't (that 24B-on-16GB run took 20+ minutes for one question,
    /// consistent with real RAM offload).
    #[test]
    fn devstral_on_a_real_16gb_card_correctly_shrinks_to_the_floor() {
        let weight_bytes = 15_177_374_099u64;
        let kv_bytes_per_token = hardware::kv_bytes_per_token(40, 8.0, 128.0, KV_CACHE_DTYPE_BYTES);
        let free_bytes = 14_000_000_000u64; // ~14GB free on a 16GB card
        let safe = fit_context_length(8192, weight_bytes, kv_bytes_per_token, free_bytes, DEFAULT_VRAM_SAFETY_FRACTION);
        assert_eq!(safe, MIN_SAFE_CONTEXT_LENGTH, "weights alone exceed the safety-margin budget on this card - correctly floors rather than claiming a false fit");
    }

    /// A genuinely VRAM-constrained case: not enough free VRAM even for
    /// the model's weights at the safety fraction - the safe context
    /// length must floor at `MIN_SAFE_CONTEXT_LENGTH`, not go to zero or
    /// underflow.
    #[test]
    fn an_extremely_constrained_gpu_floors_at_the_minimum_rather_than_underflowing() {
        let weight_bytes = 15_177_374_099u64;
        let kv_bytes_per_token = 163_840u64;
        let free_bytes = 2_000_000_000u64; // 2GB free - not even enough for the weights alone
        let safe = fit_context_length(393_216, weight_bytes, kv_bytes_per_token, free_bytes, DEFAULT_VRAM_SAFETY_FRACTION);
        assert_eq!(safe, MIN_SAFE_CONTEXT_LENGTH);
    }

    /// Plenty of free VRAM (a real high-end-card scenario): the fit
    /// should be bounded by `ceiling` (the trained/nominal context),
    /// never expanded past it just because VRAM allows more.
    #[test]
    fn abundant_vram_is_still_capped_at_the_ceiling_not_expanded_past_it() {
        let weight_bytes = 1_000_000_000u64;
        let kv_bytes_per_token = 36_864u64; // real qwen2.5:3b number
        let free_bytes = 80_000_000_000u64; // an 80GB card
        let safe = fit_context_length(32_768, weight_bytes, kv_bytes_per_token, free_bytes, DEFAULT_VRAM_SAFETY_FRACTION);
        assert_eq!(safe, 32_768);
    }

    /// A real regression this formula must never produce: dividing by a
    /// zero KV-cache-per-token (e.g. architecture metadata Ollama didn't
    /// report cleanly) must fall back to the ceiling, not panic on a
    /// divide-by-zero or silently return 0.
    #[test]
    fn zero_kv_bytes_per_token_falls_back_to_the_ceiling_instead_of_dividing_by_zero() {
        let safe = fit_context_length(32_768, 1_000_000_000, 0, 10_000_000_000, DEFAULT_VRAM_SAFETY_FRACTION);
        assert_eq!(safe, 32_768);
    }

    // Real-Ollama check below - `#[ignore]`d by default (run with
    // `cargo test -- --ignored`), needs a live `ollama serve` with
    // `devstral-small-2:24b` pulled and a real NVIDIA GPU.

    const REAL_HOST: &str = "http://localhost:11434";

    #[test]
    #[ignore]
    fn real_estimate_for_devstral_matches_this_projects_own_documented_finding() {
        let estimate = estimate_for_model(REAL_HOST, "devstral-small-2:24b").unwrap();
        println!("real devstral estimate: {estimate:#?}");

        assert_eq!(estimate.nominal_context_length, 393_216, "real nominal window, RoPE-extrapolated");
        assert_eq!(estimate.original_context_length, Some(8192), "real trained context");
        assert_eq!(estimate.kv_bytes_per_token, 163_840, "real KV-cache-per-token for this architecture");
        assert!(estimate.weight_bytes > 14_000_000_000, "real on-disk weight size should be ~15GB, got {}", estimate.weight_bytes);

        // The safe default must never exceed the trained ceiling,
        // regardless of how much VRAM this machine happens to have free
        // right now.
        assert!(estimate.safe_context_length <= 8192);
    }

    #[test]
    #[ignore]
    fn real_vram_fraction_is_none_before_the_model_has_ever_been_loaded() {
        // A model name Ollama has definitely never served in this test
        // run has nothing in `/api/ps` to report.
        let f = real_vram_fraction(REAL_HOST, "a-model-name-nobody-pulled:latest");
        assert_eq!(f, None);
    }
}
