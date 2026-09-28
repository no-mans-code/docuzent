//! A small, self-calibrating per-`(model, context_length)` speed profile
//! for `Session`'s adaptive mode - updated opportunistically from real
//! generate calls the session is already making, not dedicated
//! calibration round-trips (unlike `ollama-kv-profiler`'s own `validate`,
//! which can afford ~10 extra calls per run since it's a research tool;
//! a real product answering a real question shouldn't pay for that on
//! every document load). Persisted as a small JSON blob in the same
//! on-disk `kvcache::Cache` the LLM-context cache already uses, under a
//! distinct key prefix - no separate cache file needed.
//!
//! `disk_bytes_per_token` deliberately isn't part of this profile: unlike
//! the other fields, it doesn't need estimating at all - it's computed
//! exactly from a real cached context's serialized size whenever one is
//! on hand (see `Session`'s adaptive-mode logic).
//!
//! Running averages are capped at [`EFFECTIVE_SAMPLE_CAP`] effective
//! samples rather than growing unbounded - real conditions genuinely
//! drift (a different document mix, different system load) over a
//! profile's long lifetime, so recent measurements should carry more
//! weight than one from months ago, without needing to tune an explicit
//! decay constant.

use anyhow::Result;
use kvcache::Cache;
use serde::{Deserialize, Serialize};

/// Once a running average has this many effective samples, a new
/// observation is blended in at a fixed weight instead of an
/// ever-shrinking one - bounds how much a single old measurement can keep
/// dominating a long-lived profile.
const EFFECTIVE_SAMPLE_CAP: u32 = 20;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SpeedProfile {
    pub prefill_tokens_per_sec: f64,
    pub eval_tokens_per_sec: f64,
    pub expected_answer_tokens: f64,
    pub fixed_overhead_ms: f64,
    ingest_samples: u32,
    swap_samples: u32,
}

fn blend(old: f64, new: f64, samples: u32) -> f64 {
    let weight = samples.min(EFFECTIVE_SAMPLE_CAP) as f64;
    old + (new - old) / (weight + 1.0)
}

impl SpeedProfile {
    fn key(model: &str, context_length: u32) -> String {
        format!("profile|{model}|{context_length}")
    }

    /// `None` if this `(model, context_length)` has never been observed -
    /// the caller's job to decide what to do with no data yet (typically:
    /// just cold-ingest, there's nothing to predict against regardless).
    pub fn load(cache: &Cache, model: &str, context_length: u32) -> Result<Option<Self>> {
        match cache.get(&Self::key(model, context_length))? {
            Some(bytes) => Ok(Some(serde_json::from_slice(&bytes)?)),
            None => Ok(None),
        }
    }

    pub fn save(&self, cache: &Cache, model: &str, context_length: u32) -> Result<()> {
        cache.put(&Self::key(model, context_length), &serde_json::to_vec(self)?)
    }

    /// Blends in a real cold-ingest call's own measured prefill/eval
    /// speed. Called after *every* real ingest, swap or raw alike - both
    /// paths still pay for a real prefill+eval measurement as a side
    /// effect of doing the work at all.
    pub fn update_from_ingest(&mut self, prefill_tokens_per_sec: f64, eval_tokens_per_sec: f64) {
        self.prefill_tokens_per_sec = blend(self.prefill_tokens_per_sec, prefill_tokens_per_sec, self.ingest_samples);
        self.eval_tokens_per_sec = blend(self.eval_tokens_per_sec, eval_tokens_per_sec, self.ingest_samples);
        self.ingest_samples = self.ingest_samples.saturating_add(1);
    }

    /// Blends in a real swap-shaped call's own answer length and residual
    /// overhead (`wall_ms` minus both reported durations) - called after
    /// any real call that reused a disk- or memory-cached context and
    /// produced a real answer.
    pub fn update_from_swap(&mut self, answer_tokens: f64, overhead_ms: f64) {
        self.expected_answer_tokens = blend(self.expected_answer_tokens, answer_tokens, self.swap_samples);
        self.fixed_overhead_ms = blend(self.fixed_overhead_ms, overhead_ms.max(0.0), self.swap_samples);
        self.swap_samples = self.swap_samples.saturating_add(1);
    }

    /// A fresh profile seeded entirely from one real observation - used
    /// the first time a `(model, context_length)` is ever seen.
    pub fn seed_from_ingest(prefill_tokens_per_sec: f64, eval_tokens_per_sec: f64) -> Self {
        Self {
            prefill_tokens_per_sec,
            eval_tokens_per_sec,
            expected_answer_tokens: 0.0,
            fixed_overhead_ms: 0.0,
            ingest_samples: 1,
            swap_samples: 0,
        }
    }

    /// Whether enough real swap-shaped data exists yet to trust
    /// `expected_answer_tokens`/`fixed_overhead_ms` for a prediction.
    pub fn has_swap_data(&self) -> bool {
        self.swap_samples > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cache(label: &str) -> (Cache, std::path::PathBuf) {
        let path = std::env::temp_dir().join(format!("speed-profile-test-{label}-{}.redb", std::process::id()));
        (Cache::open(&path, 1_000_000).unwrap(), path)
    }

    #[test]
    fn load_on_an_unseen_model_returns_none() {
        let (cache, path) = temp_cache("unseen");
        assert!(SpeedProfile::load(&cache, "some-model", 4096).unwrap().is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_then_load_round_trips() {
        let (cache, path) = temp_cache("roundtrip");
        let profile = SpeedProfile::seed_from_ingest(1000.0, 50.0);
        profile.save(&cache, "m", 4096).unwrap();
        let loaded = SpeedProfile::load(&cache, "m", 4096).unwrap().unwrap();
        assert_eq!(loaded.prefill_tokens_per_sec, 1000.0);
        assert_eq!(loaded.eval_tokens_per_sec, 50.0);
        assert!(!loaded.has_swap_data());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn different_context_lengths_get_independent_profiles() {
        let (cache, path) = temp_cache("independent");
        SpeedProfile::seed_from_ingest(1000.0, 50.0).save(&cache, "m", 4096).unwrap();
        SpeedProfile::seed_from_ingest(2000.0, 80.0).save(&cache, "m", 8192).unwrap();
        assert_eq!(SpeedProfile::load(&cache, "m", 4096).unwrap().unwrap().prefill_tokens_per_sec, 1000.0);
        assert_eq!(SpeedProfile::load(&cache, "m", 8192).unwrap().unwrap().prefill_tokens_per_sec, 2000.0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn update_from_ingest_moves_the_running_average_toward_new_observations() {
        let mut profile = SpeedProfile::seed_from_ingest(1000.0, 50.0);
        profile.update_from_ingest(2000.0, 50.0);
        // Second sample blended in at weight 1/(1+1) = 0.5: (1000+2000)/2 = 1500
        assert!((profile.prefill_tokens_per_sec - 1500.0).abs() < 0.001);
    }

    #[test]
    fn update_from_swap_records_real_answer_length_and_overhead() {
        let mut profile = SpeedProfile::seed_from_ingest(1000.0, 50.0);
        assert!(!profile.has_swap_data());
        profile.update_from_swap(12.0, 40.0);
        assert!(profile.has_swap_data());
        assert_eq!(profile.expected_answer_tokens, 12.0);
        assert_eq!(profile.fixed_overhead_ms, 40.0);
    }

    #[test]
    fn later_samples_carry_bounded_weight_not_ever_shrinking() {
        let mut profile = SpeedProfile::seed_from_ingest(1000.0, 50.0);
        for _ in 0..100 {
            profile.update_from_ingest(1000.0, 50.0);
        }
        // A single big outlier after many samples should still move the
        // average by a real, bounded amount - not an infinitesimal one.
        let before = profile.prefill_tokens_per_sec;
        profile.update_from_ingest(2000.0, 50.0);
        let moved = profile.prefill_tokens_per_sec - before;
        assert!(moved > 1.0, "outlier should still move a long-lived average by a non-trivial amount, moved {moved}");
    }
}
