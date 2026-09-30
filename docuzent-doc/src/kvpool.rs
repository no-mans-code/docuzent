//! Swapping saved KV states in and out of the model's one slot.
//!
//! The GPU holds a single working KV state; a book's parts each have a
//! saved one on disk. [`KvPool::with_resident`] makes a named prefix the
//! slot's contents - already there, restored from its file (~0.1 s), or, if
//! the file was evicted, read again and saved anew - and runs a closure
//! against the model while it is. Priming ([`KvPool::prime`]) is how a
//! state is created in the first place, at book creation.
//!
//! One caller at a time: the slot is a single shared resource, so the whole
//! "make resident, then generate" step holds a lock.

use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use anyhow::Result;
use docuzent_kv::KvStore;
use docuzent_llm::{Engine, Llm, Sampling};

/// How a prefix came to be in the slot.
#[derive(Debug, Clone, PartialEq)]
pub enum Swap {
    /// It already was - nothing to do.
    AlreadyResident,
    /// Loaded from its saved file.
    Restored { tokens: usize, ms: f64 },
    /// No usable saved file: the text was read again (and saved).
    Reprimed { tokens: usize, prefill_ms: f64 },
    /// The engine cannot save states (Ollama): the prefix is read as part of the prompt, every time.
    NotSaved,
}

/// What creating a saved state cost.
#[derive(Debug, Clone, PartialEq)]
pub struct Primed {
    pub tokens: usize,
    pub prefill_ms: f64,
    pub save_ms: f64,
    pub bytes: u64,
    /// Files the LRU store evicted to make room.
    pub evicted: Vec<String>,
}

pub struct KvPool {
    /// Swappable: the model can be changed while the app runs (between jobs - see [`KvPool::set_engine`]).
    engine: RwLock<Arc<dyn Engine>>,
    store: Arc<KvStore>,
    /// Which saved state the slot's contents start with, if known.
    resident: Mutex<Option<String>>,
    /// Serializes use of the slot.
    slot: Mutex<()>,
}

impl KvPool {
    pub fn new(engine: Arc<dyn Engine>, store: Arc<KvStore>) -> Self {
        Self { engine: RwLock::new(engine), store, resident: Mutex::new(None), slot: Mutex::new(()) }
    }

    pub fn store(&self) -> &Arc<KvStore> {
        &self.store
    }

    pub fn engine(&self) -> Arc<dyn Engine> {
        self.engine.read().unwrap().clone()
    }

    /// Runs on another model from now on. Waits for the slot (no job is cut short), and forgets what was resident:
    /// the new model has its own saved states, under its own names.
    pub fn set_engine(&self, engine: Arc<dyn Engine>) {
        let _slot = self.slot.lock().unwrap();
        *self.engine.write().unwrap() = engine;
        *self.resident.lock().unwrap() = None;
    }

    /// Reads `prefix` into a clean slot, saves the resulting state as
    /// `file`, and registers it (owned by `owner`) with the LRU store.
    pub fn prime(&self, file: &str, owner: &str, prefix: &str) -> Result<Primed> {
        let _slot = self.slot.lock().unwrap();
        self.prime_locked(file, owner, prefix)
    }

    fn prime_locked(&self, file: &str, owner: &str, prefix: &str) -> Result<Primed> {
        let engine = self.engine();
        if !engine.persists() {
            // Nothing can be saved: reading the prefix now would only be thrown away.
            return Ok(Primed { tokens: 0, prefill_ms: 0.0, save_ms: 0.0, bytes: 0, evicted: Vec::new() });
        }
        engine.erase()?;
        *self.resident.lock().unwrap() = None;
        let t0 = Instant::now();
        // One token is generated only because a completion must; the state that matters is the prefix.
        let c = engine.complete(prefix, &Sampling::precise(1), &mut |_| {})?;
        let prefill_ms = if c.prompt_ms > 0.0 { c.prompt_ms } else { t0.elapsed().as_secs_f64() * 1000.0 };
        let io = engine.save(file)?;
        let evicted = self.store.register(file, owner)?;
        *self.resident.lock().unwrap() = Some(file.to_string());
        Ok(Primed { tokens: c.prompt_tokens + c.cached_tokens, prefill_ms, save_ms: io.ms, bytes: io.bytes, evicted })
    }

    /// Makes `file`'s state the slot's contents (priming from `prefix` if
    /// the file is gone) and runs `f` with the model. Returns `f`'s result
    /// and how the state got there.
    pub fn with_resident<R>(&self, file: &str, owner: &str, prefix: &str, f: impl FnOnce(&dyn Llm) -> Result<R>) -> Result<(R, Swap)> {
        let _slot = self.slot.lock().unwrap();
        let engine = self.engine();
        if !engine.persists() {
            let out = f(engine.as_ref());
            return Ok((out?, Swap::NotSaved));
        }
        let already = self.resident.lock().unwrap().as_deref() == Some(file);
        let swap = if already {
            Swap::AlreadyResident
        } else {
            let restored = if self.store.contains(file) { engine.restore(file)? } else { None };
            match restored {
                Some(io) => {
                    self.store.touch(file)?;
                    *self.resident.lock().unwrap() = Some(file.to_string());
                    Swap::Restored { tokens: io.tokens, ms: io.ms }
                }
                None => {
                    // Evicted, or written by other weights: read the text again.
                    let _ = self.store.remove(file);
                    let p = self.prime_locked(file, owner, prefix)?;
                    Swap::Reprimed { tokens: p.tokens, prefill_ms: p.prefill_ms }
                }
            }
        };
        if already {
            self.store.touch(file)?;
        }
        let out = f(engine.as_ref());
        // Whatever `f` did, the slot still *begins* with `file`'s prefix, but
        // an error may have left it in an unknown state: forget, to be safe.
        if out.is_err() {
            *self.resident.lock().unwrap() = None;
        }
        Ok((out?, swap))
    }

    /// Runs `f` against the model for work that is not about any part of a
    /// book (merging ideas, deriving speakers). It will overwrite the slot,
    /// so nothing is trusted to be resident afterwards.
    pub fn with_scratch<R>(&self, f: impl FnOnce(&dyn Llm) -> Result<R>) -> Result<R> {
        let _slot = self.slot.lock().unwrap();
        *self.resident.lock().unwrap() = None;
        let engine = self.engine();
        f(engine.as_ref())
    }

    /// The slot's contents are unknown (something else used it).
    pub fn forget_resident(&self) {
        *self.resident.lock().unwrap() = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use docuzent_llm::{chatml, SimEngine};

    fn pool(label: &str) -> (Arc<SimEngine>, KvPool, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("kvpool-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let store = Arc::new(KvStore::open(&dir, 10 * 1024 * 1024 * 1024).unwrap());
        let engine = Arc::new(SimEngine::new(12288, |_| String::new()).in_dir(&dir));
        (engine.clone(), KvPool::new(engine, store), dir)
    }

    fn prefix(n: usize) -> String {
        chatml::system(&format!("part {n}: {}", "text ".repeat(800)))
    }

    #[test]
    fn a_primed_part_is_restored_not_reread_and_a_question_costs_only_its_own_tokens() {
        let (sim, pool, dir) = pool("restore");
        let p = prefix(1);
        let primed = pool.prime("p1.kv", "book", &p).unwrap();
        assert!(primed.tokens > 900, "priming read the whole part");
        // Another part is used in between, so part 1 is no longer resident.
        pool.prime("p2.kv", "book", &prefix(2)).unwrap();
        sim.clear_events();
        let (answer_cost, swap) = pool
            .with_resident("p1.kv", "book", &p, |llm| {
                let c = llm.complete(&chatml::ask(&p, "what does it say?"), &Sampling::precise(20), &mut |_| {})?;
                Ok(c.prompt_tokens)
            })
            .unwrap();
        assert!(matches!(swap, Swap::Restored { .. }), "{swap:?}");
        assert!(answer_cost < 40, "only the question was processed, not the part: {answer_cost} tokens");
        assert_eq!(sim.restore_hits(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn asking_twice_of_the_same_part_needs_no_second_swap() {
        let (sim, pool, dir) = pool("resident");
        let p = prefix(1);
        pool.prime("p1.kv", "book", &p).unwrap();
        sim.clear_events();
        for _ in 0..2 {
            let (_, swap) = pool.with_resident("p1.kv", "book", &p, |_| Ok(())).unwrap();
            assert_eq!(swap, Swap::AlreadyResident);
        }
        assert_eq!(sim.restore_hits() + sim.restore_misses(), 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Regression: an evicted or damaged saved state must fall back to
    /// reading the text again - the answer is slower, never wrong or missing.
    #[test]
    fn a_missing_saved_state_is_transparently_rebuilt() {
        let (sim, pool, dir) = pool("evicted");
        let p = prefix(1);
        pool.prime("p1.kv", "book", &p).unwrap();
        pool.prime("p2.kv", "book", &prefix(2)).unwrap();
        sim.lose_file("p1.kv"); // the LRU store lost it
        let (_, swap) = pool.with_resident("p1.kv", "book", &p, |_| Ok(())).unwrap();
        assert!(matches!(swap, Swap::Reprimed { tokens, .. } if tokens > 900), "{swap:?}");
        assert!(pool.store().contains("p1.kv"), "and it is saved again for next time");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn an_error_inside_the_closure_forgets_what_is_resident() {
        let (_, pool, dir) = pool("error");
        let p = prefix(1);
        pool.prime("p1.kv", "book", &p).unwrap();
        assert!(pool.with_resident::<()>("p1.kv", "book", &p, |_| anyhow::bail!("boom")).is_err());
        let (_, swap) = pool.with_resident("p1.kv", "book", &p, |_| Ok(())).unwrap();
        assert!(matches!(swap, Swap::Restored { .. }), "the slot was not trusted after a failure: {swap:?}");
        let _ = std::fs::remove_dir_all(dir);
    }
}
