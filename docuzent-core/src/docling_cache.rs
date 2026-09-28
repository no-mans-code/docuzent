//! A separate on-disk cache for Docling's own (slow, GPU/CPU-bound)
//! parsing output - independent of which model or LLM-cache mode is in
//! play, since parsing a file's text is the same work no matter what's
//! later done with it. Keyed per source file (its content hash), so a file
//! reused across different document sets is parsed once and reused, not
//! re-cached per grouping.
//!
//! Capacity is caller-supplied (see [`DEFAULT_CAPACITY_BYTES`] for the
//! deliberate 10GB reservation callers should default to) rather than the
//! dynamic "half of available disk space" [`kvcache::default_capacity_bytes`]
//! computes for the LLM-context cache - this cache's budget is a stated
//! reservation, not derived from whatever free space happens to exist.

use std::path::Path;

use anyhow::Result;
use kvcache::Cache;

/// A deliberate reservation, not derived from available disk space.
pub const DEFAULT_CAPACITY_BYTES: u64 = 10 * 1024 * 1024 * 1024;

pub struct DoclingCache {
    cache: Cache,
    capacity_bytes: u64,
}

impl DoclingCache {
    pub fn open(path: &Path, capacity_bytes: u64) -> Result<Self> {
        Ok(Self { cache: Cache::open(path, capacity_bytes)?, capacity_bytes })
    }

    /// Returns the previously-parsed, post-chunking combined text for a
    /// file with this content hash, if this exact file has been parsed
    /// before.
    pub fn get(&self, file_hash: &str) -> Result<Option<String>> {
        match self.cache.get(file_hash)? {
            Some(bytes) => Ok(Some(String::from_utf8(bytes)?)),
            None => Ok(None),
        }
    }

    /// Caches `text` under `file_hash` - unless `text` alone is bigger
    /// than the whole cache capacity, in which case this is a deliberate
    /// no-op (never evict everything just to fit one oversized document;
    /// see module doc comment).
    pub fn put(&self, file_hash: &str, text: &str) -> Result<()> {
        if text.len() as u64 > self.capacity_bytes {
            return Ok(());
        }
        self.cache.put(file_hash, text.as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("docling-cache-test-{label}-{}.redb", std::process::id()))
    }

    #[test]
    fn miss_then_hit_after_a_real_put() {
        let path = temp_path("hit");
        let cache = DoclingCache::open(&path, 1_000_000).unwrap();
        assert_eq!(cache.get("hash1").unwrap(), None);
        cache.put("hash1", "some parsed document text").unwrap();
        assert_eq!(cache.get("hash1").unwrap(), Some("some parsed document text".to_string()));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn an_oversized_value_is_never_written_but_smaller_ones_still_are() {
        let path = temp_path("oversized");
        // A tiny capacity makes the guard genuinely exercisable in a test,
        // rather than allocating a real multi-GB string.
        let cache = DoclingCache::open(&path, 10).unwrap();
        cache.put("small", "tiny").unwrap(); // 4 bytes, fits
        cache.put("big", "this text is definitely over ten bytes").unwrap(); // over cap, refused
        assert_eq!(cache.get("small").unwrap(), Some("tiny".to_string()));
        assert_eq!(cache.get("big").unwrap(), None, "oversized value should never have been written");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn default_capacity_is_ten_gb() {
        assert_eq!(DEFAULT_CAPACITY_BYTES, 10 * 1024 * 1024 * 1024);
    }
}
