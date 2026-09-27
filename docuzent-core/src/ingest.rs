use std::path::PathBuf;
use std::process::Command;

use anyhow::{bail, Context, Result};

/// Parameters for one Docling conversion run.
pub struct IngestOptions {
    pub source: PathBuf,
    pub output: PathBuf,
    pub to: String,
    pub device: String,
    pub docling_bin: Option<String>,
}

/// What actually happened, for the caller to report however it likes.
pub struct IngestReport {
    pub docling_bin: String,
    pub produced_files: Vec<String>,
}

/// Resolves the docling executable: explicit override, then `$DOCLING_BIN`,
/// then a project-local `.venv`, then bare `docling` on PATH.
pub fn resolve_docling_bin(explicit: Option<&str>) -> String {
    explicit
        .map(|s| s.to_string())
        .or_else(|| std::env::var("DOCLING_BIN").ok())
        .or_else(|| find_venv_docling().map(|p| p.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "docling".to_string())
}

/// Look for a `.venv` in the current directory or its parent (covers running
/// from the repo root or from a workspace member directory) and return its
/// docling executable.
fn find_venv_docling() -> Option<PathBuf> {
    let rel = if cfg!(windows) {
        "Scripts/docling.exe"
    } else {
        "bin/docling"
    };
    for base in [PathBuf::from("."), PathBuf::from(".."), PathBuf::from("../..")] {
        let candidate = base.join(".venv").join(rel);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Runs Docling's `convert` over `opts.source`, writing results to
/// `opts.output`. Does not print anything - that's the caller's job.
pub fn run(opts: &IngestOptions) -> Result<IngestReport> {
    let docling_bin = resolve_docling_bin(opts.docling_bin.as_deref());

    if !opts.source.exists() {
        bail!("source path does not exist: {}", opts.source.display());
    }

    std::fs::create_dir_all(&opts.output)
        .with_context(|| format!("failed to create output dir {}", opts.output.display()))?;

    let status = Command::new(&docling_bin)
        .arg("convert")
        .arg(&opts.source)
        .arg("--to")
        .arg(&opts.to)
        .arg("--output")
        .arg(&opts.output)
        .arg("--device")
        .arg(&opts.device)
        .status()
        .with_context(|| {
            format!(
                "failed to launch `{docling_bin}` - is Docling installed and on PATH? \
                 Set --docling-bin or the DOCLING_BIN env var to point at it."
            )
        })?;

    if !status.success() {
        bail!("docling exited with status: {status}");
    }

    let produced_files: Vec<_> = std::fs::read_dir(&opts.output)
        .with_context(|| format!("failed to read output dir {}", opts.output.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();

    Ok(IngestReport { docling_bin, produced_files })
}
