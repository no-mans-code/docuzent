//! The llama.cpp server client: streaming completions, token counting, and
//! the slot save/restore API that persists the model's real KV cache.
//!
//! Started as
//! `llama-server -m qwen3-14b.gguf -c 12288 -ngl 99 -fa on --cache-type-k q8_0 --cache-type-v q8_0 --slot-save-path /kv --parallel 1`
//! - one slot, so "the KV cache" is unambiguous, with the cache stored in
//! 8-bit (half the disk per saved part at no measurable cost).

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::{Completion, KvSlot, Llm, Sampling, SlotIo};

/// The slot everything uses.
const SLOT: usize = 0;

pub struct LlamaServer {
    base: String,
    agent: ureq::Agent,
    ctx: usize,
    model_path: String,
    /// The model's own chat template, as the server reports it (empty if it does not).
    chat_template: String,
}

/// A saved-state file name is joined to the server's slot directory, so it
/// must be one plain name.
fn valid_file_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 200 && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')) && !name.starts_with('.')
}

/// Extra tries at a slot action after the connection failed.
const SLOT_RETRIES: usize = 2;

impl LlamaServer {
    /// Connects to a running server and reads its context size.
    pub fn connect(base: &str) -> Result<Self> {
        let base = base.trim_end_matches('/').to_string();
        let agent = ureq::AgentBuilder::new().max_idle_connections(0).timeout_connect(Duration::from_secs(5)).timeout_read(Duration::from_secs(900)).build();
        let props: Value = agent.get(&format!("{base}/props")).call().with_context(|| format!("no llama.cpp server at {base} - is it running?"))?.into_json()?;
        let ctx = props["default_generation_settings"]["n_ctx"].as_u64().or_else(|| props["n_ctx"].as_u64()).context("the server did not report its context size")? as usize;
        let model_path = props["model_path"].as_str().unwrap_or("").to_string();
        let chat_template = props["chat_template"].as_str().unwrap_or("").to_string();
        Ok(Self { base, agent, ctx, model_path, chat_template })
    }

    /// The weights file the server loaded (as it reports it), so the product
    /// can refuse to run on anything but its one model.
    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// docuzent writes its prompts in ChatML (`<|im_start|>`), which Qwen, Hermes and many other models are trained
    /// on. A model with another template (Llama 3, Gemma, Mistral) would read them as noise here - run such a model
    /// through Ollama, which applies the model's own template. Unknown (no template reported) counts as ChatML.
    pub fn speaks_chatml(&self) -> bool {
        self.chat_template.is_empty() || self.chat_template.contains("<|im_start|>")
    }

    /// Whether the server answers and its model has finished loading.
    pub fn healthy(base: &str) -> bool {
        ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(3))
            .build()
            .get(&format!("{}/health", base.trim_end_matches('/')))
            .call()
            .ok()
            .and_then(|r| r.into_json::<Value>().ok())
            .is_some_and(|v| v["status"] == "ok")
    }

    /// Saving, restoring and erasing a slot can each be done twice with the same result, so a failure of the
    /// *connection* (not an error the server answered with) is tried again: the server has been seen to
    /// hang up in the middle of a restore ("Unexpected EOF") for no reason it logs.
    fn slot_action(&self, action: &str, body: Value) -> Result<Value, ureq::Error> {
        let mut attempt = 0;
        loop {
            match self.agent.post(&format!("{}/slots/{SLOT}?action={action}", self.base)).send_json(body.clone()) {
                Ok(resp) => return Ok(resp.into_json().unwrap_or(Value::Null)),
                Err(ureq::Error::Transport(_)) if attempt < SLOT_RETRIES => {
                    attempt += 1;
                    std::thread::sleep(Duration::from_millis(400 * attempt as u64));
                }
                Err(e) => return Err(e),
            }
        }
    }
}

fn error_text(e: ureq::Error) -> String {
    match e {
        ureq::Error::Status(code, resp) => format!("HTTP {code}: {}", resp.into_string().unwrap_or_default().chars().take(300).collect::<String>()),
        other => other.to_string(),
    }
}

impl Llm for LlamaServer {
    fn complete(&self, prompt: &str, s: &Sampling, on_token: &mut dyn FnMut(&str)) -> Result<Completion> {
        let body = json!({
            "prompt": prompt,
            "n_predict": s.n_predict,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "stop": s.stop,
            "stream": true,
            "cache_prompt": true,
            "id_slot": SLOT,
        });
        let resp = self.agent.post(&format!("{}/completion", self.base)).send_json(body).map_err(|e| anyhow::anyhow!("completion failed: {}", error_text(e)))?;

        let started = Instant::now();
        let mut out = Completion::default();
        for line in BufReader::new(resp.into_reader()).lines() {
            let line = line.context("the completion stream broke")?;
            let Some(data) = line.strip_prefix("data:") else { continue };
            let Ok(v) = serde_json::from_str::<Value>(data.trim()) else { continue };
            if let Some(err) = v.get("error") {
                bail!("the model server reported an error: {err}");
            }
            if let Some(piece) = v["content"].as_str() {
                if !piece.is_empty() {
                    out.text.push_str(piece);
                    on_token(piece);
                }
            }
            if v["stop"].as_bool() == Some(true) {
                let t = &v["timings"];
                out.prompt_tokens = t["prompt_n"].as_u64().or_else(|| v["tokens_evaluated"].as_u64()).unwrap_or(0) as usize;
                out.cached_tokens = t["cache_n"].as_u64().or_else(|| v["tokens_cached"].as_u64()).unwrap_or(0) as usize;
                out.prompt_ms = t["prompt_ms"].as_f64().unwrap_or(0.0);
                out.gen_tokens = t["predicted_n"].as_u64().unwrap_or(0) as usize;
                out.gen_ms = t["predicted_ms"].as_f64().unwrap_or_else(|| started.elapsed().as_secs_f64() * 1000.0);
                out.truncated = v["stop_type"] == "limit" || v["truncated"].as_bool() == Some(true);
            }
        }
        Ok(out)
    }

    fn can_think(&self) -> bool {
        self.chat_template.is_empty() || self.chat_template.contains("<think>")
    }

    fn count_tokens(&self, text: &str) -> Result<usize> {
        let v: Value = self.agent.post(&format!("{}/tokenize", self.base)).send_json(json!({ "content": text })).map_err(|e| anyhow::anyhow!("tokenize failed: {}", error_text(e)))?.into_json()?;
        Ok(v["tokens"].as_array().context("the server returned no token list")?.len())
    }
}

impl KvSlot for LlamaServer {
    fn save(&self, file: &str) -> Result<SlotIo> {
        if !valid_file_name(file) {
            bail!("`{file}` is not a valid KV file name");
        }
        let v = self.slot_action("save", json!({ "filename": file })).map_err(|e| anyhow::anyhow!("saving the KV cache failed: {}", error_text(e)))?;
        Ok(SlotIo { tokens: v["n_saved"].as_u64().unwrap_or(0) as usize, bytes: v["n_written"].as_u64().unwrap_or(0), ms: v["timings"]["save_ms"].as_f64().unwrap_or(0.0) })
    }

    fn restore(&self, file: &str) -> Result<Option<SlotIo>> {
        if !valid_file_name(file) {
            bail!("`{file}` is not a valid KV file name");
        }
        match self.slot_action("restore", json!({ "filename": file })) {
            Ok(v) => Ok(Some(SlotIo { tokens: v["n_restored"].as_u64().unwrap_or(0) as usize, bytes: v["n_read"].as_u64().unwrap_or(0), ms: v["timings"]["restore_ms"].as_f64().unwrap_or(0.0) })),
            // Missing (evicted) or written by different weights: not an error, the text is simply read again.
            Err(ureq::Error::Status(400..=404, _)) => Ok(None),
            Err(e) => bail!("restoring the KV cache failed: {}", error_text(e)),
        }
    }

    fn erase(&self) -> Result<()> {
        self.slot_action("erase", json!({})).map_err(|e| anyhow::anyhow!("erasing the KV cache failed: {}", error_text(e)))?;
        Ok(())
    }

    fn context_size(&self) -> usize {
        self.ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// A one-thread stand-in for the llama.cpp server: answers each
    /// request from `respond(method_and_path, body) -> (status, content_type, body)`
    /// and remembers what it was asked.
    struct Mock {
        base: String,
        seen: Arc<Mutex<Vec<(String, String)>>>,
    }

    fn mock(respond: impl Fn(&str, &str) -> (u16, String) + Send + 'static) -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { break };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let (head, body) = loop {
                    let n = s.read(&mut chunk).unwrap_or(0);
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        let head = String::from_utf8_lossy(&buf[..p]).to_string();
                        let len = head.to_lowercase().lines().find_map(|l| l.strip_prefix("content-length:").and_then(|v| v.trim().parse::<usize>().ok())).unwrap_or(0);
                        while buf.len() < p + 4 + len {
                            let n = s.read(&mut chunk).unwrap_or(0);
                            if n == 0 {
                                break;
                            }
                            buf.extend_from_slice(&chunk[..n]);
                        }
                        break (head, String::from_utf8_lossy(&buf[p + 4..]).to_string());
                    }
                    if n == 0 {
                        break (String::new(), String::new());
                    }
                };
                let line = head.lines().next().unwrap_or("").to_string();
                let key = line.split(' ').take(2).collect::<Vec<_>>().join(" ");
                seen2.lock().unwrap().push((key.clone(), body.clone()));
                let (status, payload) = respond(&key, &body);
                let _ = write!(s, "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}", payload.len());
            }
        });
        Mock { base, seen }
    }

    fn standard(key: &str, _body: &str) -> (u16, String) {
        match key {
            "GET /props" => (200, r#"{"default_generation_settings":{"n_ctx":12288},"model_path":"/models/qwen3-14b-q4_k_m.gguf"}"#.into()),
            "POST /completion" => (
                200,
                "data: {\"content\":\"Hel\",\"stop\":false}\n\ndata: {\"content\":\"lo\",\"stop\":false}\n\ndata: {\"content\":\"\",\"stop\":true,\"stop_type\":\"eos\",\"timings\":{\"prompt_n\":18,\"cache_n\":4150,\"prompt_ms\":30.5,\"predicted_n\":2,\"predicted_ms\":55.0}}\n\n".into(),
            ),
            "POST /tokenize" => (200, r#"{"tokens":[1,2,3,4,5]}"#.into()),
            "POST /slots/0?action=save" => (200, r#"{"n_saved":4168,"n_written":363000000,"timings":{"save_ms":120.0}}"#.into()),
            "POST /slots/0?action=restore" => (200, r#"{"n_restored":4168,"n_read":363000000,"timings":{"restore_ms":70.0}}"#.into()),
            "POST /slots/0?action=erase" => (200, r#"{"n_erased":1}"#.into()),
            _ => (404, r#"{"error":"nope"}"#.into()),
        }
    }

    /// Regression: "restoring the KV cache failed: Network Error: Unexpected EOF" in the middle of a
    /// question ended it. The server hung up on one request; the same request works a moment later.
    #[test]
    fn a_slot_action_the_server_hangs_up_on_is_tried_again() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let served = Arc::new(Mutex::new(0usize));
        let served2 = served.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { break };
                let mut chunk = [0u8; 8192];
                let n = s.read(&mut chunk).unwrap_or(0);
                let req = String::from_utf8_lossy(&chunk[..n]).to_string();
                let mut count = served2.lock().unwrap();
                *count += 1;
                if req.contains("/slots/0") && *count == 2 {
                    continue; // hang up without a word, on the first slot request
                }
                let payload = if req.starts_with("GET /props") { r#"{"default_generation_settings":{"n_ctx":12288},"model_path":"m.gguf"}"# } else { r#"{"n_restored":4168,"n_read":1,"timings":{"restore_ms":70.0}}"# };
                let _ = write!(s, "HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: {}
Connection: close

{payload}", payload.len());
            }
        });
        let s = LlamaServer::connect(&base).unwrap();
        assert!(s.restore("some.kv").unwrap().is_some(), "the second try succeeded");
        assert_eq!(*served.lock().unwrap(), 3, "props, the request that was hung up on, and the retry");
    }

    #[test]
    fn connect_reads_the_context_size() {
        let m = mock(standard);
        let s = LlamaServer::connect(&m.base).unwrap();
        assert_eq!((s.context_size(), s.model_path()), (12288, "/models/qwen3-14b-q4_k_m.gguf"));
    }

    #[test]
    fn a_completion_streams_fragments_and_reports_where_the_time_went() {
        let m = mock(standard);
        let s = LlamaServer::connect(&m.base).unwrap();
        let mut pieces = Vec::new();
        let c = s.complete("PROMPT", &Sampling::precise(50), &mut |t| pieces.push(t.to_string())).unwrap();
        assert_eq!(pieces, vec!["Hel", "lo"]);
        assert_eq!(c.text, "Hello");
        assert_eq!((c.prompt_tokens, c.cached_tokens, c.gen_tokens), (18, 4150, 2), "only the new tokens were processed; the rest came from the KV cache");
        assert!(!c.truncated);
        let sent: Value = serde_json::from_str(&m.seen.lock().unwrap().iter().find(|(k, _)| k == "POST /completion").unwrap().1).unwrap();
        assert_eq!((sent["cache_prompt"].as_bool(), sent["id_slot"].as_u64(), sent["stream"].as_bool()), (Some(true), Some(0), Some(true)), "prefix reuse and the single slot are always requested");
        assert_eq!(sent["temperature"].as_f64(), Some(0.0));
    }

    #[test]
    fn slot_save_and_restore_report_tokens_bytes_and_time() {
        let m = mock(standard);
        let s = LlamaServer::connect(&m.base).unwrap();
        let saved = s.save("abc.bin").unwrap();
        assert_eq!((saved.tokens, saved.bytes), (4168, 363_000_000));
        let restored = s.restore("abc.bin").unwrap().unwrap();
        assert_eq!((restored.tokens, restored.ms), (4168, 70.0));
        s.erase().unwrap();
        let sent = m.seen.lock().unwrap().clone();
        assert!(sent.iter().any(|(k, b)| k == "POST /slots/0?action=save" && b.contains("abc.bin")));
    }

    /// Regression: an evicted KV file must mean "read the text again", never an error.
    #[test]
    fn restoring_a_missing_file_is_none_not_an_error() {
        let m = mock(|key, b| if key == "POST /slots/0?action=restore" { (400, r#"{"error":"Failed to restore"}"#.into()) } else { standard(key, b) });
        let s = LlamaServer::connect(&m.base).unwrap();
        assert!(s.restore("gone.bin").unwrap().is_none());
    }

    #[test]
    fn file_names_cannot_escape_the_slot_directory() {
        let m = mock(standard);
        let s = LlamaServer::connect(&m.base).unwrap();
        for bad in ["../etc/passwd", "a/b.bin", "", ".hidden", "a b.bin"] {
            assert!(s.save(bad).is_err() && s.restore(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn tokens_are_counted_by_the_servers_tokenizer() {
        let m = mock(standard);
        assert_eq!(LlamaServer::connect(&m.base).unwrap().count_tokens("anything").unwrap(), 5);
    }

    #[test]
    fn a_server_that_is_not_there_says_so() {
        let err = LlamaServer::connect("http://127.0.0.1:1").err().unwrap().to_string();
        assert!(err.contains("no llama.cpp server"), "{err}");
    }
}
