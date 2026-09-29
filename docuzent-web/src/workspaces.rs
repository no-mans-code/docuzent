//! Persisted, named document workspaces for the web UI - a thin save/
//! reload layer over `Session`'s existing `load_documents`/
//! `load_text_files` and on-disk context cache, not a new architecture.
//! A saved workspace just remembers which files (and which model/mode/
//! context length) to reload - the actual document content and LLM
//! context stay in the caches that already exist for that purpose, so
//! reloading a workspace is a real cache hit whenever the disk cache
//! still has it, not a special case. See
//! https://github.com/no-mans-code/docuzent/issues/38.

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Workspace {
    /// The exact file paths to reload - today these are paths under
    /// `docuzent-web`'s own upload directory (`std::env::temp_dir()
    /// .join("docuzent-web-uploads")`), which persists across requests
    /// as long as the server process keeps running. A real "re-upload a
    /// file that's gone" flow is future work, not this issue's scope.
    pub paths: Vec<String>,
    pub text_only: bool,
    pub model: String,
    pub mode: String,
    pub context_length: u32,
    pub map_reduce_context_fraction: f32,
    /// RFC 3339 - when this workspace was last saved (created or updated).
    pub updated_at: String,
}

#[derive(Default, Serialize, Deserialize)]
struct WorkspaceStore {
    #[serde(default)]
    workspaces: BTreeMap<String, Workspace>,
}

/// A thin wrapper over one small JSON file - workspaces are a handful of
/// named records, not a dataset that needs a real database.
pub struct WorkspaceManager {
    path: PathBuf,
}

impl WorkspaceManager {
    pub fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn load(&self) -> Result<WorkspaceStore> {
        match std::fs::read(&self.path) {
            Ok(bytes) => serde_json::from_slice(&bytes).context("failed to parse workspaces.json"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(WorkspaceStore::default()),
            Err(e) => Err(e).context("failed to read workspaces.json"),
        }
    }

    fn save(&self, store: &WorkspaceStore) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let bytes = serde_json::to_vec_pretty(store).context("failed to serialize workspaces")?;
        std::fs::write(&self.path, bytes).context("failed to write workspaces.json")
    }

    pub fn list(&self) -> Result<BTreeMap<String, Workspace>> {
        Ok(self.load()?.workspaces)
    }

    pub fn get(&self, name: &str) -> Result<Option<Workspace>> {
        Ok(self.load()?.workspaces.get(name).cloned())
    }

    /// Creates or overwrites the workspace named `name` - saving again
    /// under the same name is how a workspace's file set or model/mode
    /// gets updated, not a separate "edit" operation.
    pub fn save_workspace(&self, name: &str, workspace: Workspace) -> Result<()> {
        let mut store = self.load()?;
        store.workspaces.insert(name.to_string(), workspace);
        self.save(&store)
    }

    pub fn delete(&self, name: &str) -> Result<bool> {
        let mut store = self.load()?;
        let existed = store.workspaces.remove(name).is_some();
        if existed {
            self.save(&store)?;
        }
        Ok(existed)
    }
}

/// A workspace name must be non-empty, not absurdly long, and made only
/// of characters that are safe as a JSON map key and as UI text -
/// permissive enough for a real name ("NCERT chapter 3", "my-notes_v2")
/// without needing any escaping downstream. Never used to build a
/// filesystem path (the file names to actually load are stored inside
/// the `Workspace` record, never derived from its name), but kept
/// conservative anyway rather than trusting arbitrary input.
pub fn sanitize_name(name: &str) -> Option<String> {
    let trimmed = name.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 100 {
        return None;
    }
    let safe: String = trimmed.chars().filter(|c| c.is_alphanumeric() || matches!(c, ' ' | '-' | '_')).collect();
    let safe = safe.trim();
    if safe.is_empty() {
        None
    } else {
        Some(safe.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!("docuzent-web-workspaces-test-{label}-{}.json", std::process::id()))
    }

    fn sample(paths: Vec<&str>) -> Workspace {
        Workspace {
            paths: paths.into_iter().map(String::from).collect(),
            text_only: false,
            model: "qwen2.5:3b".to_string(),
            mode: "adaptive".to_string(),
            context_length: 32768,
            map_reduce_context_fraction: 0.25,
            updated_at: "2026-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn an_unseen_store_starts_empty() {
        let path = temp_path("empty");
        let mgr = WorkspaceManager::new(path.clone());
        assert!(mgr.list().unwrap().is_empty());
        assert_eq!(mgr.get("anything").unwrap(), None);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn save_then_get_round_trips() {
        let path = temp_path("roundtrip");
        let mgr = WorkspaceManager::new(path.clone());
        mgr.save_workspace("my notes", sample(vec!["a.txt", "b.txt"])).unwrap();

        let got = mgr.get("my notes").unwrap().unwrap();
        assert_eq!(got.paths, vec!["a.txt".to_string(), "b.txt".to_string()]);
        assert_eq!(got.model, "qwen2.5:3b");

        let listed = mgr.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert!(listed.contains_key("my notes"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn saving_again_under_the_same_name_overwrites_not_duplicates() {
        let path = temp_path("overwrite");
        let mgr = WorkspaceManager::new(path.clone());
        mgr.save_workspace("ws", sample(vec!["old.txt"])).unwrap();
        mgr.save_workspace("ws", sample(vec!["new.txt", "new2.txt"])).unwrap();

        assert_eq!(mgr.list().unwrap().len(), 1, "must overwrite, not accumulate a second entry");
        assert_eq!(mgr.get("ws").unwrap().unwrap().paths, vec!["new.txt".to_string(), "new2.txt".to_string()]);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_real_process_restart_survives_via_the_json_file() {
        let path = temp_path("restart");
        {
            let mgr = WorkspaceManager::new(path.clone());
            mgr.save_workspace("persisted", sample(vec!["doc.pdf"])).unwrap();
        }
        // A fresh manager over the same path, simulating a new process.
        let mgr2 = WorkspaceManager::new(path.clone());
        assert_eq!(mgr2.get("persisted").unwrap().unwrap().paths, vec!["doc.pdf".to_string()]);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn delete_removes_it_and_reports_whether_it_existed() {
        let path = temp_path("delete");
        let mgr = WorkspaceManager::new(path.clone());
        mgr.save_workspace("ws", sample(vec!["a.txt"])).unwrap();

        assert!(mgr.delete("ws").unwrap());
        assert_eq!(mgr.get("ws").unwrap(), None);
        assert!(!mgr.delete("ws").unwrap(), "deleting something already gone reports false, not an error");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sanitize_name_accepts_a_real_name_with_spaces_and_punctuation() {
        assert_eq!(sanitize_name("NCERT chapter 3"), Some("NCERT chapter 3".to_string()));
        assert_eq!(sanitize_name("my-notes_v2"), Some("my-notes_v2".to_string()));
    }

    #[test]
    fn sanitize_name_rejects_empty_or_whitespace_only() {
        assert_eq!(sanitize_name(""), None);
        assert_eq!(sanitize_name("   "), None);
    }

    #[test]
    fn sanitize_name_rejects_an_absurdly_long_name() {
        let too_long = "a".repeat(200);
        assert_eq!(sanitize_name(&too_long), None);
    }

    #[test]
    fn sanitize_name_strips_unsafe_characters_rather_than_rejecting_outright() {
        assert_eq!(sanitize_name("../../etc/passwd"), Some(".. ..etcpasswd".to_string()).map(|_| "etcpasswd".to_string()).or(sanitize_name("../../etc/passwd")));
        // The real property that matters: no path separators or dots
        // survive into the sanitized name.
        let sanitized = sanitize_name("../../etc/passwd").unwrap();
        assert!(!sanitized.contains('/'));
        assert!(!sanitized.contains('.'));
    }
}
