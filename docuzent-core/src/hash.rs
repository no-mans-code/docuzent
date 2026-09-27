//! Content hashing for cache keys - not a security boundary, just a cheap,
//! reliable way to notice "is this the same document."

use std::path::Path;

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};

/// Hex-encoded SHA-256 of a file's bytes.
pub fn hash_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    Ok(hash_bytes(&bytes))
}

pub fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_encode(&hasher.finalize())
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_bytes_hash_the_same() {
        assert_eq!(hash_bytes(b"hello"), hash_bytes(b"hello"));
    }

    #[test]
    fn different_bytes_hash_differently() {
        assert_ne!(hash_bytes(b"hello"), hash_bytes(b"world"));
    }

    #[test]
    fn produces_64_hex_chars_for_sha256() {
        assert_eq!(hash_bytes(b"anything").len(), 64);
    }
}
