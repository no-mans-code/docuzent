//! Zip extraction, deliberately shallow: only the top level of a zip is
//! ever extracted. An entry that is itself a zip is left alone rather than
//! recursed into - the standard defense against a zip bomb (a zip
//! containing a zip containing a zip...) without needing to track
//! decompressed-size budgets or nesting depth ourselves.

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Extracts every entry directly inside `zip_path` into `dest_dir`
/// (created if needed). Entries whose own name ends in `.zip` are written
/// out as plain files, not extracted further. Returns the paths written.
pub fn extract_one_level(zip_path: &Path, dest_dir: &Path) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dest_dir)
        .with_context(|| format!("failed to create {}", dest_dir.display()))?;

    let file = File::open(zip_path).with_context(|| format!("failed to open {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("failed to read {} as a zip", zip_path.display()))?;

    let mut written = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        if entry.is_dir() {
            continue;
        }

        let out_path = match entry.enclosed_name() {
            Some(p) => dest_dir.join(p),
            None => continue, // unsafe path (e.g. absolute or traversal) - skip it
        };
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut out_file = File::create(&out_path)
            .with_context(|| format!("failed to create {}", out_path.display()))?;
        std::io::copy(&mut entry, &mut out_file)
            .with_context(|| format!("failed to extract {}", out_path.display()))?;

        // A nested zip is written to disk as plain bytes but never opened
        // as an archive itself - that's the one-level limit. Whatever
        // later tries to parse it (Docling) will simply skip a file it
        // doesn't recognize as a real document.
        written.push(out_path);
    }
    Ok(written)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_dir(label: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("docuzent-core-archive-test-{label}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    fn make_zip(path: &Path, entries: &[(&str, &[u8])]) {
        let file = File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        for (name, contents) in entries {
            zip.start_file(*name, zip::write::SimpleFileOptions::default()).unwrap();
            zip.write_all(contents).unwrap();
        }
        zip.finish().unwrap();
    }

    #[test]
    fn extracts_top_level_files() {
        let dir = temp_dir("flat");
        let zip_path = dir.join("archive.zip");
        make_zip(&zip_path, &[("a.txt", b"hello"), ("b.txt", b"world")]);

        let dest = dir.join("out");
        let written = extract_one_level(&zip_path, &dest).unwrap();
        assert_eq!(written.len(), 2);
        assert_eq!(std::fs::read(dest.join("a.txt")).unwrap(), b"hello");
        assert_eq!(std::fs::read(dest.join("b.txt")).unwrap(), b"world");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn nested_zip_is_written_but_not_extracted() {
        let dir = temp_dir("nested");
        let inner_zip = dir.join("inner.zip");
        make_zip(&inner_zip, &[("secret.txt", b"should not be auto-extracted")]);
        let inner_bytes = std::fs::read(&inner_zip).unwrap();

        let outer_zip = dir.join("outer.zip");
        make_zip(&outer_zip, &[("nested.zip", &inner_bytes), ("readme.txt", b"top level file")]);

        let dest = dir.join("out");
        let written = extract_one_level(&outer_zip, &dest).unwrap();
        assert_eq!(written.len(), 2);
        // The nested zip exists on disk as a plain file...
        assert!(dest.join("nested.zip").exists());
        // ...but its contents were never extracted anywhere.
        assert!(!dest.join("secret.txt").exists());

        std::fs::remove_dir_all(&dir).ok();
    }
}
