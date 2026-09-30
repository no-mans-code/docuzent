//! The store of saved KV-cache files: a directory with a size budget and
//! least-recently-used eviction.
//!
//! The model server writes the files (into its `--slot-save-path`, which is
//! this directory); this crate decides *which of them may stay*. A single
//! book's parts are hundreds of megabytes each, so the budget is what keeps
//! a shelf of books from filling the disk - and eviction is safe, never
//! harmful: an evicted part is simply read again the next time it is needed
//! (see `KvSlot::restore` returning `None`).
//!
//! Everything here is plain files and one small index; it has no idea what
//! a model is. That is deliberate - it is tested on its own.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

const INDEX_FILE: &str = ".kvindex.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Entry {
    bytes: u64,
    /// Logical clock of the last use; higher is more recent. Survives
    /// restarts, so the order does too.
    last_used: u64,
    /// Which book the file belongs to (for deleting a book, and so that
    /// eviction prefers other books' files over the one being read).
    owner: String,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct Index {
    clock: u64,
    entries: BTreeMap<String, Entry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub files: usize,
    pub bytes: u64,
    pub budget: u64,
}

pub struct KvStore {
    dir: PathBuf,
    /// A store that shares its directory with another (an evaluation next to a running app, both writing through
    /// the same model server) keeps its own index and sees only its own files - names starting `{scope}-`.
    scope: Option<String>,
    budget: u64,
    /// The most one book may hold. A huge book (a Bible is ~260 parts, ~95 GB
    /// of saved state) must not be able to push every other book's memory out
    /// of the store: past this share it evicts its *own* least recently used
    /// parts instead.
    owner_cap: u64,
    index: Mutex<Index>,
}

/// One plain file name: the model server joins it to its slot directory.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 200 && !name.starts_with('.') && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// `{model}-{book}-{kind}.kv`, each part shortened and cleaned. Includes the
/// model so weights that differ can never share a file - a KV state is only
/// meaningful for the exact weights that produced it.
pub fn file_name(model_id: &str, book_id: &str, kind: &str) -> String {
    let clean = |s: &str, n: usize| -> String { s.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '-').take(n).collect() };
    format!("{}-{}-{}.kv", clean(model_id, 12), clean(book_id, 16), clean(kind, 40))
}

fn index_file(scope: Option<&str>) -> String {
    match scope {
        Some(s) => format!(".kvindex-{s}.json"),
        None => INDEX_FILE.to_string(),
    }
}

impl KvStore {
    /// Opens (creating if needed) the store at `dir` with a size `budget`
    /// in bytes. Reconciles the index with what is actually on disk: files
    /// the server wrote that the index has not seen are adopted, index
    /// entries whose file is gone are dropped.
    pub fn open(dir: impl Into<PathBuf>, budget: u64) -> Result<Self> {
        Self::open_scoped(dir, budget, None)
    }

    /// [`KvStore::open`] for a store that shares `dir` with another: with a `scope`, it has its own index and
    /// adopts, counts and evicts only the files whose names start `{scope}-` (so it can never evict another store's
    /// files). Name its files with [`file_name`] and a model id beginning `{scope}-`.
    pub fn open_scoped(dir: impl Into<PathBuf>, budget: u64, scope: Option<&str>) -> Result<Self> {
        let dir = dir.into();
        let scope = scope.map(|s| s.chars().filter(|c| c.is_ascii_alphanumeric()).collect::<String>()).filter(|s| !s.is_empty());
        std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
        let index_file = index_file(scope.as_deref());
        let mut index: Index = std::fs::read(dir.join(&index_file)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();

        let mine = |name: &str| scope.as_deref().is_none_or(|s| name.starts_with(&format!("{s}-")));
        index.entries.retain(|name, _| dir.join(name).is_file() && mine(name));
        for entry in std::fs::read_dir(&dir)?.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if !valid_name(&name) || index.entries.contains_key(&name) || !entry.path().is_file() || !mine(&name) {
                continue;
            }
            index.clock += 1;
            index.entries.insert(name, Entry { bytes: entry.metadata()?.len(), last_used: index.clock, owner: String::new() });
        }
        let store = Self { dir, scope, budget, owner_cap: (budget / 2).max(1), index: Mutex::new(index) };
        store.persist(&store.index.lock().unwrap())?;
        Ok(store)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn persist(&self, index: &Index) -> Result<()> {
        let file = index_file(self.scope.as_deref());
        let tmp = self.dir.join(format!("{file}.tmp"));
        std::fs::write(&tmp, serde_json::to_vec(index)?)?;
        std::fs::rename(tmp, self.dir.join(file))?;
        Ok(())
    }

    /// Whether `name` is stored (and its file is really there).
    pub fn contains(&self, name: &str) -> bool {
        self.index.lock().unwrap().entries.contains_key(name) && self.dir.join(name).is_file()
    }

    /// Marks `name` as just used, so it is the last to be evicted.
    pub fn touch(&self, name: &str) -> Result<()> {
        let mut idx = self.index.lock().unwrap();
        idx.clock += 1;
        let clock = idx.clock;
        if let Some(e) = idx.entries.get_mut(name) {
            e.last_used = clock;
            self.persist(&idx)?;
        }
        Ok(())
    }

    /// Records a file the model server just wrote for `owner` (a book), then
    /// evicts least-recently-used files until the store fits its budget.
    /// Returns the names evicted. The new file is never evicted by its own
    /// registration; other books' files go first, and only if that is not
    /// enough do older files of the same book.
    pub fn register(&self, name: &str, owner: &str) -> Result<Vec<String>> {
        if !valid_name(name) {
            bail!("`{name}` is not a valid KV file name");
        }
        let bytes = std::fs::metadata(self.dir.join(name)).with_context(|| format!("`{name}` was not written"))?.len();
        let mut idx = self.index.lock().unwrap();
        idx.clock += 1;
        let clock = idx.clock;
        idx.entries.insert(name.to_string(), Entry { bytes, last_used: clock, owner: owner.to_string() });

        let mut evicted = Vec::new();
        loop {
            let total: u64 = idx.entries.values().map(|e| e.bytes).sum();
            let mine: u64 = idx.entries.values().filter(|e| e.owner == owner).map(|e| e.bytes).sum();
            let over_budget = total > self.budget;
            // The share only protects *other* books; a book on its own may use the whole budget.
            let over_share = mine > self.owner_cap && total > mine;
            if !over_budget && !over_share {
                break;
            }
            // Over this book's share: drop this book's own oldest. Otherwise (over the whole budget)
            // other books' files go first, then the oldest of this one.
            let victim = idx
                .entries
                .iter()
                .filter(|(n, e)| n.as_str() != name && (!over_share || e.owner == owner))
                .min_by_key(|(_, e)| (e.owner == owner && !over_share, e.last_used))
                .map(|(n, _)| n.clone());
            let Some(victim) = victim else { break };
            let _ = std::fs::remove_file(self.dir.join(&victim));
            idx.entries.remove(&victim);
            evicted.push(victim);
        }
        self.persist(&idx)?;
        Ok(evicted)
    }

    pub fn remove(&self, name: &str) -> Result<()> {
        let mut idx = self.index.lock().unwrap();
        if idx.entries.remove(name).is_some() {
            let _ = std::fs::remove_file(self.dir.join(name));
            self.persist(&idx)?;
        }
        Ok(())
    }

    /// Deletes every file of one book (when the book is removed).
    pub fn remove_owner(&self, owner: &str) -> Result<usize> {
        let mut idx = self.index.lock().unwrap();
        let names: Vec<String> = idx.entries.iter().filter(|(_, e)| e.owner == owner).map(|(n, _)| n.clone()).collect();
        for n in &names {
            let _ = std::fs::remove_file(self.dir.join(n));
            idx.entries.remove(n);
        }
        self.persist(&idx)?;
        Ok(names.len())
    }

    /// Deletes every file whose owner starts with `prefix` - temporary keys left behind by a job that was cut
    /// short (`tmp:...`), swept when the app starts.
    pub fn remove_owners_starting(&self, prefix: &str) -> Result<usize> {
        let mut idx = self.index.lock().unwrap();
        let names: Vec<String> = idx.entries.iter().filter(|(_, e)| e.owner.starts_with(prefix)).map(|(n, _)| n.clone()).collect();
        for n in &names {
            let _ = std::fs::remove_file(self.dir.join(n));
            idx.entries.remove(n);
        }
        self.persist(&idx)?;
        Ok(names.len())
    }

    pub fn names_of(&self, owner: &str) -> Vec<String> {
        self.index.lock().unwrap().entries.iter().filter(|(_, e)| e.owner == owner).map(|(n, _)| n.clone()).collect()
    }

    pub fn stats(&self) -> Stats {
        let idx = self.index.lock().unwrap();
        Stats { files: idx.entries.len(), bytes: idx.entries.values().map(|e| e.bytes).sum(), budget: self.budget }
    }
}

#[cfg(test)]
mod scope_tests {
    use super::*;

    #[test]
    fn a_scoped_store_sees_only_its_own_files_and_never_evicts_anothers() {
        let dir = std::env::temp_dir().join(format!("kv-scope-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("qwen3-book-p1.kv"), vec![0u8; 600]).unwrap(); // the app's saved part
        let eval = KvStore::open_scoped(&dir, 1000, Some("eval")).unwrap();
        assert_eq!(eval.stats().files, 0, "the app's file is not the evaluation's to count or evict");
        std::fs::write(dir.join("eval-m-doc-p1.kv"), vec![0u8; 700]).unwrap();
        eval.register("eval-m-doc-p1.kv", "doc").unwrap();
        std::fs::write(dir.join("eval-m-doc-p2.kv"), vec![0u8; 700]).unwrap();
        let evicted = eval.register("eval-m-doc-p2.kv", "doc").unwrap();
        assert_eq!(evicted, vec!["eval-m-doc-p1.kv".to_string()], "over budget: its own oldest file goes");
        assert!(dir.join("qwen3-book-p1.kv").is_file(), "the app's file is untouched");
        let app = KvStore::open(&dir, 1 << 30).unwrap();
        assert!(app.contains("qwen3-book-p1.kv"), "the app's own index is separate");
        let again = KvStore::open_scoped(&dir, 1000, Some("eval")).unwrap();
        assert_eq!(again.stats().files, 1, "the scope's index persists on its own");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(label: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kvstore-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// Stands in for the model server writing a saved state.
    fn write(store: &KvStore, name: &str, bytes: usize) {
        std::fs::write(store.dir().join(name), vec![0u8; bytes]).unwrap();
    }

    #[test]
    fn the_least_recently_used_file_is_evicted_when_over_budget() {
        let d = temp_dir("lru");
        let s = KvStore::open(&d, 250).unwrap();
        for n in ["a.kv", "b.kv"] {
            write(&s, n, 100);
            s.register(n, "book1").unwrap();
        }
        s.touch("a.kv").unwrap(); // a is now more recent than b
        write(&s, "c.kv", 100);
        let evicted = s.register("c.kv", "book1").unwrap();
        assert_eq!(evicted, vec!["b.kv"], "b was the least recently used");
        assert!(s.contains("a.kv") && s.contains("c.kv") && !s.contains("b.kv"));
        assert!(!d.join("b.kv").exists(), "eviction really deletes the file");
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn eviction_prefers_other_books_over_the_one_being_read() {
        let d = temp_dir("owners");
        let s = KvStore::open(&d, 500).unwrap();
        write(&s, "other.kv", 350);
        s.register("other.kv", "old-book").unwrap();
        write(&s, "mine1.kv", 100);
        s.register("mine1.kv", "gita").unwrap();
        write(&s, "mine2.kv", 100);
        let evicted = s.register("mine2.kv", "gita").unwrap();
        assert_eq!(evicted, vec!["other.kv"], "the book being worked on keeps its parts");
        let _ = std::fs::remove_dir_all(d);
    }

    /// Regression: a huge book (the Bible is ~260 parts) must not wipe every
    /// other book's saved memory - past its share it evicts its own oldest.
    #[test]
    fn one_huge_book_cannot_push_every_other_book_out() {
        let d = temp_dir("share");
        let s = KvStore::open(&d, 1000).unwrap(); // each book may hold at most 500
        for n in ["small1.kv", "small2.kv"] {
            write(&s, n, 100);
            s.register(n, "gita").unwrap();
        }
        for i in 0..40 {
            let n = format!("bible{i:02}.kv");
            write(&s, &n, 100);
            s.register(&n, "bible").unwrap();
        }
        assert!(s.contains("small1.kv") && s.contains("small2.kv"), "the small book kept all of its saved memory");
        let bible = s.names_of("bible");
        assert_eq!(bible.len(), 5, "the huge book holds only its share (500 of 1000): {bible:?}");
        assert!(bible.contains(&"bible39.kv".to_string()) && !bible.contains(&"bible00.kv".to_string()), "and it kept its most recent parts");
        let _ = std::fs::remove_dir_all(d);
    }

    /// Temporary keys (one job's working memory) are deleted by owner, and any a dead job left are swept by prefix.
    #[test]
    fn temporary_keys_can_be_deleted_by_job_or_swept_by_prefix() {
        let d = temp_dir("temp");
        let s = KvStore::open(&d, 1000).unwrap();
        for (n, o) in [("keep.kv", "book"), ("t1a.kv", "tmp:1"), ("t1b.kv", "tmp:1"), ("t2a.kv", "tmp:2")] {
            write(&s, n, 10);
            s.register(n, o).unwrap();
        }
        assert_eq!(s.remove_owner("tmp:1").unwrap(), 2);
        assert!(!s.contains("t1a.kv") && !d.join("t1a.kv").exists() && s.contains("t2a.kv") && s.contains("keep.kv"));
        assert_eq!(s.remove_owners_starting("tmp:").unwrap(), 1, "the one a dead job left");
        assert!(!s.contains("t2a.kv") && s.contains("keep.kv"), "a book's own files are never touched");
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn a_file_is_never_evicted_by_its_own_registration() {
        let d = temp_dir("self");
        let s = KvStore::open(&d, 50).unwrap();
        write(&s, "huge.kv", 500);
        let evicted = s.register("huge.kv", "b").unwrap();
        assert!(evicted.is_empty() && s.contains("huge.kv"), "a single over-budget file is kept rather than losing what was just built");
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn the_order_and_contents_survive_a_restart() {
        let d = temp_dir("restart");
        {
            let s = KvStore::open(&d, 250).unwrap();
            for n in ["a.kv", "b.kv"] {
                write(&s, n, 100);
                s.register(n, "g").unwrap();
            }
            s.touch("a.kv").unwrap();
        }
        let s = KvStore::open(&d, 250).unwrap();
        assert!(s.contains("a.kv") && s.contains("b.kv"));
        write(&s, "c.kv", 100);
        assert_eq!(s.register("c.kv", "g").unwrap(), vec!["b.kv"], "recency was remembered across the restart");
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn files_the_server_wrote_are_adopted_and_vanished_files_forgotten() {
        let d = temp_dir("reconcile");
        {
            let s = KvStore::open(&d, 1000).unwrap();
            write(&s, "gone.kv", 10);
            s.register("gone.kv", "g").unwrap();
        }
        std::fs::remove_file(d.join("gone.kv")).unwrap();
        std::fs::write(d.join("new.kv"), b"12345").unwrap();
        let s = KvStore::open(&d, 1000).unwrap();
        assert!(!s.contains("gone.kv") && s.contains("new.kv"));
        assert_eq!(s.stats(), Stats { files: 1, bytes: 5, budget: 1000 });
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn removing_a_book_removes_all_and_only_its_files() {
        let d = temp_dir("owner-rm");
        let s = KvStore::open(&d, 1000).unwrap();
        for (n, o) in [("a.kv", "one"), ("b.kv", "one"), ("c.kv", "two")] {
            write(&s, n, 10);
            s.register(n, o).unwrap();
        }
        assert_eq!(s.remove_owner("one").unwrap(), 2);
        assert!(!s.contains("a.kv") && !s.contains("b.kv") && s.contains("c.kv"));
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn names_are_plain_and_bind_a_state_to_its_model_and_book() {
        assert!(valid_name("qwen3-abc-gita-part1.kv") && !valid_name("../x") && !valid_name(".kvindex.json") && !valid_name("a/b"));
        let a = file_name("sha256-a8cc1361f3145d", "af1db213665a", "part-001");
        let b = file_name("sha256-OTHERWEIGHTS", "af1db213665a", "part-001");
        assert_ne!(a, b, "different weights never share a saved state");
        assert!(valid_name(&a) && a.ends_with(".kv"));
    }

    #[test]
    fn registering_a_file_that_was_never_written_is_an_error() {
        let d = temp_dir("missing");
        let s = KvStore::open(&d, 100).unwrap();
        assert!(s.register("nothing.kv", "b").is_err());
        let _ = std::fs::remove_dir_all(d);
    }
}
