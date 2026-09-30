//! A simulated engine: no GPU, no server, but the *same accounting*.
//!
//! It models what matters for testing KV-cache behaviour - one slot holding
//! a prefix, prompts that only cost the tokens beyond that prefix, saved
//! states that can be restored (or have gone missing) - so a test can assert
//! "the second question processed fewer than 50 tokens" or "restoring a part
//! never re-read its text", against exactly the traits the real
//! [`crate::LlamaServer`] implements. What it "says" comes from a
//! caller-supplied responder, so a test controls the model's output.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::Result;

use crate::{Completion, KvSlot, Llm, Sampling, SlotIo};

/// Simulated tokenizer: four characters per token, rounded up.
pub fn sim_tokens(chars: usize) -> usize {
    chars.div_ceil(4)
}

/// One thing the engine was asked to do, in order.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Complete { processed: usize, cached: usize, prompt_head: String },
    Save { file: String },
    Restore { file: String, hit: bool },
    Erase,
}

#[derive(Default)]
struct State {
    /// The text whose KV state the slot currently holds.
    resident: String,
    files: HashMap<String, String>,
    events: Vec<Event>,
    processed_total: usize,
    /// `n_predict` of every completion, in order.
    n_predicts: Vec<i32>,
}

pub struct SimEngine {
    /// When set, `save` writes a small real file here (as the real server
    /// does), so a `KvStore` over the same directory sees it.
    dir: Option<std::path::PathBuf>,
    state: Mutex<State>,
    responder: Box<dyn Fn(&str) -> String + Send + Sync>,
    ctx: usize,
}

impl SimEngine {
    /// `responder` maps the full prompt to the model's reply (empty for a
    /// priming call).
    pub fn new(ctx: usize, responder: impl Fn(&str) -> String + Send + Sync + 'static) -> Self {
        Self { dir: None, state: Mutex::new(State::default()), responder: Box::new(responder), ctx }
    }

    /// Saved states are also written to `dir` as real files.
    pub fn in_dir(mut self, dir: impl Into<std::path::PathBuf>) -> Self {
        self.dir = Some(dir.into());
        self
    }

    pub fn events(&self) -> Vec<Event> {
        self.state.lock().unwrap().events.clone()
    }

    /// Every completion's prompt tokens that had to be processed, summed.
    pub fn processed_tokens(&self) -> usize {
        self.state.lock().unwrap().processed_total
    }

    pub fn completions(&self) -> Vec<(usize, usize, String)> {
        self.state.lock().unwrap().events.iter().filter_map(|e| if let Event::Complete { processed, cached, prompt_head } = e { Some((*processed, *cached, prompt_head.clone())) } else { None }).collect()
    }

    pub fn n_predicts(&self) -> Vec<i32> {
        self.state.lock().unwrap().n_predicts.clone()
    }

    pub fn restore_hits(&self) -> usize {
        self.state.lock().unwrap().events.iter().filter(|e| matches!(e, Event::Restore { hit: true, .. })).count()
    }

    pub fn restore_misses(&self) -> usize {
        self.state.lock().unwrap().events.iter().filter(|e| matches!(e, Event::Restore { hit: false, .. })).count()
    }

    pub fn saved_files(&self) -> Vec<String> {
        let mut v: Vec<String> = self.state.lock().unwrap().files.keys().cloned().collect();
        v.sort();
        v
    }

    /// Simulates the LRU store evicting a saved state.
    pub fn lose_file(&self, file: &str) {
        self.state.lock().unwrap().files.remove(file);
        if let Some(dir) = &self.dir {
            let _ = std::fs::remove_file(dir.join(file));
        }
    }

    pub fn resident_text(&self) -> String {
        self.state.lock().unwrap().resident.clone()
    }

    pub fn clear_events(&self) {
        let mut s = self.state.lock().unwrap();
        s.events.clear();
        s.n_predicts.clear();
        s.processed_total = 0;
    }
}

fn common_prefix_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

impl Llm for SimEngine {
    fn complete(&self, prompt: &str, s: &Sampling, on_token: &mut dyn FnMut(&str)) -> Result<Completion> {
        let reply = (self.responder)(prompt);
        let mut st = self.state.lock().unwrap();
        let common = common_prefix_len(&st.resident, prompt);
        let cached = sim_tokens(common);
        let total = sim_tokens(prompt.chars().count());
        let processed = total.saturating_sub(cached);
        st.processed_total += processed;
        st.n_predicts.push(s.n_predict);
        st.events.push(Event::Complete { processed, cached, prompt_head: prompt.chars().take(60).collect() });

        // Honour n_predict the way a real model would: cut the reply off.
        let max_chars = (s.n_predict.max(0) as usize) * 4;
        let (text, truncated) = if reply.chars().count() > max_chars { (reply.chars().take(max_chars).collect::<String>(), true) } else { (reply, false) };
        for w in text.split_inclusive(' ') {
            on_token(w);
        }
        st.resident = format!("{prompt}{text}");
        Ok(Completion { gen_tokens: sim_tokens(text.chars().count()), prompt_tokens: processed, cached_tokens: cached, prompt_ms: processed as f64 * 0.5, gen_ms: text.chars().count() as f64 * 6.0, text, truncated })
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(sim_tokens(text.chars().count()))
    }
}

impl KvSlot for SimEngine {
    fn save(&self, file: &str) -> Result<SlotIo> {
        let mut st = self.state.lock().unwrap();
        let resident = st.resident.clone();
        let tokens = sim_tokens(resident.chars().count());
        if let Some(dir) = &self.dir {
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join(file), vec![0u8; 1024])?;
        }
        st.files.insert(file.to_string(), resident);
        st.events.push(Event::Save { file: file.to_string() });
        Ok(SlotIo { tokens, bytes: tokens as u64 * 80 * 1024, ms: 1.0 })
    }

    fn restore(&self, file: &str) -> Result<Option<SlotIo>> {
        let mut st = self.state.lock().unwrap();
        let on_disk = self.dir.as_ref().map(|d| d.join(file).is_file()).unwrap_or(true);
        let saved = st.files.get(file).cloned().filter(|_| on_disk);
        st.events.push(Event::Restore { file: file.to_string(), hit: saved.is_some() });
        Ok(saved.map(|text| {
            let tokens = sim_tokens(text.chars().count());
            st.resident = text;
            SlotIo { tokens, bytes: tokens as u64 * 80 * 1024, ms: 0.1 }
        }))
    }

    fn erase(&self) -> Result<()> {
        let mut st = self.state.lock().unwrap();
        st.resident.clear();
        st.events.push(Event::Erase);
        Ok(())
    }

    fn context_size(&self) -> usize {
        self.ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_tokens_beyond_the_resident_prefix_are_processed() {
        let e = SimEngine::new(8192, |_| String::new());
        let prefix = "x".repeat(4000); // 1000 tokens
        let cold = e.complete(&prefix, &Sampling::precise(1), &mut |_| {}).unwrap();
        assert_eq!((cold.prompt_tokens, cold.cached_tokens), (1000, 0));
        let warm = e.complete(&format!("{prefix}and a question"), &Sampling::precise(1), &mut |_| {}).unwrap();
        assert!(warm.prompt_tokens < 10 && warm.cached_tokens >= 990, "{warm:?}");
    }

    #[test]
    fn a_saved_state_can_be_restored_after_the_slot_is_wiped() {
        let e = SimEngine::new(8192, |_| String::new());
        let prefix = "y".repeat(4000);
        e.complete(&prefix, &Sampling::precise(1), &mut |_| {}).unwrap();
        e.save("p.bin").unwrap();
        e.erase().unwrap();
        assert!(e.restore("p.bin").unwrap().is_some());
        let c = e.complete(&format!("{prefix}q"), &Sampling::precise(1), &mut |_| {}).unwrap();
        assert!(c.prompt_tokens <= 1, "restored state means the part is not re-read: {c:?}");
    }

    #[test]
    fn a_lost_file_restores_as_none() {
        let e = SimEngine::new(8192, |_| String::new());
        e.complete("z", &Sampling::precise(1), &mut |_| {}).unwrap();
        e.save("a.bin").unwrap();
        e.lose_file("a.bin");
        assert!(e.restore("a.bin").unwrap().is_none());
        assert_eq!((e.restore_hits(), e.restore_misses()), (0, 1));
    }

    #[test]
    fn n_predict_cuts_a_long_reply_off_and_says_so() {
        let e = SimEngine::new(8192, |_| "word ".repeat(100));
        let c = e.complete("p", &Sampling::precise(10), &mut |_| {}).unwrap();
        assert!(c.truncated && c.text.chars().count() <= 40);
    }
}
