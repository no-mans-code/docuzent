//! An Ollama server as the engine: any model it has pulled, on any machine it runs on.
//!
//! What it can and cannot do, honestly:
//!
//! * It **can** run any chat model - docuzent builds its prompts in ChatML, and they are turned back into chat
//!   messages here, so Ollama applies each model's own chat template (Llama, Gemma, Mistral, Qwen...).
//! * It **cannot** save or restore a model's KV state to a file: Ollama has no such API (its old `context` was
//!   only token ids, replayed by re-reading them). So with Ollama nothing is "memorised": every question re-reads
//!   the parts of the book it needs. Answers are the same; they are slower, and the slower the longer the book.
//!   [`KvSlot::persists`] says so, and the pool then skips saving altogether.
//! * Ollama keeps the last prompt's prefix in memory, so consecutive calls on the same part are still cheap.

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};

use crate::{Completion, KvSlot, Llm, Sampling, SlotIo};

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";

pub struct OllamaServer {
    base: String,
    agent: ureq::Agent,
    model: String,
    ctx: usize,
    /// The model can reason in a separate thinking channel (Ollama's `think` option).
    thinking: bool,
}

/// One model an Ollama server has, as the settings page lists it.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct OllamaModel {
    pub name: String,
    pub parameter_size: String,
    /// The most context the model supports, when Ollama says.
    pub context_length: Option<usize>,
    pub capabilities: Vec<String>,
    /// Whether it can be run on (it must write text, in a big enough window), and if not, why.
    pub usable: bool,
    pub reason: Option<String>,
}

fn agent(read_secs: u64) -> ureq::Agent {
    ureq::AgentBuilder::new().max_idle_connections(0).timeout_connect(Duration::from_secs(5)).timeout_read(Duration::from_secs(read_secs)).build()
}

fn base_of(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

/// The largest `*.context_length` in a model's `model_info` (keys are prefixed by architecture: `llama.`, `qwen3.`).
fn context_length_of(info: &Value) -> Option<usize> {
    info.as_object()?.iter().filter(|(k, _)| k.ends_with(".context_length")).filter_map(|(_, v)| v.as_u64()).max().map(|n| n as usize)
}

/// A prompt built in ChatML, back as chat messages - and whether its last, open assistant turn asked for thinking
/// (`Some(true)`: a bare `<|im_start|>assistant`), for no thinking (`Some(false)`: an empty `<think></think>`), or
/// left no open turn at all (`None`). Text that is not ChatML is one user message.
pub fn to_messages(prompt: &str) -> (Vec<(String, String)>, Option<bool>) {
    if !prompt.contains(IM_START) {
        return (vec![("user".into(), prompt.to_string())], None);
    }
    let mut out = Vec::new();
    let mut rest = prompt;
    let mut think = None;
    while let Some(start) = rest.find(IM_START) {
        let after = &rest[start + IM_START.len()..];
        let Some(nl) = after.find('\n') else { break };
        let role = after[..nl].trim().to_string();
        let body = &after[nl + 1..];
        match body.find(IM_END) {
            Some(end) => {
                out.push((role, body[..end].to_string()));
                rest = &body[end + IM_END.len()..];
            }
            None => {
                // The open turn the model is to write.
                let open = body.trim();
                think = Some(!(open.starts_with("<think>") && open.contains("</think>")));
                if !open.is_empty() && !open.starts_with("<think>") {
                    // A started answer (a prefill) - kept as the start of the assistant's turn.
                    out.push((role, open.to_string()));
                }
                break;
            }
        }
    }
    (out, think)
}

impl OllamaServer {
    /// Whether an Ollama server answers at `url`.
    pub fn reachable(url: &str) -> bool {
        agent(3).get(&format!("{}/api/version", base_of(url))).call().is_ok()
    }

    /// The models the server has, each marked usable or not (an embedding model cannot talk; a model with less
    /// context than `min_context` cannot hold a part of a book and the question about it).
    pub fn list(url: &str, min_context: usize) -> Result<Vec<OllamaModel>> {
        let base = base_of(url);
        let v: Value = agent(10).get(&format!("{base}/api/tags")).call().with_context(|| format!("no Ollama server at {base} - is it running, and reachable from here?"))?.into_json()?;
        let mut out = Vec::new();
        for m in v["models"].as_array().cloned().unwrap_or_default() {
            let name = m["name"].as_str().unwrap_or("").to_string();
            let capabilities: Vec<String> = m["capabilities"].as_array().map(|a| a.iter().filter_map(|c| c.as_str().map(str::to_string)).collect()).unwrap_or_default();
            let context_length = m["details"]["context_length"].as_u64().map(|n| n as usize);
            let reason = if !capabilities.is_empty() && !capabilities.iter().any(|c| c == "completion") {
                Some("cannot write text (an embedding model?)".to_string())
            } else if context_length.is_some_and(|c| c < min_context) {
                Some(format!("its context ({} tokens) is under the {min_context} a part of a book and its question need", context_length.unwrap_or(0)))
            } else {
                None
            };
            out.push(OllamaModel { name, parameter_size: m["details"]["parameter_size"].as_str().unwrap_or("").to_string(), context_length, capabilities, usable: reason.is_none(), reason });
        }
        out.sort_by(|a, b| b.usable.cmp(&a.usable).then(a.name.cmp(&b.name)));
        Ok(out)
    }

    /// Connects to `model` on the server at `url`, with a window of the model's own context length, capped at
    /// `max_context` (more context costs memory on the machine running Ollama).
    pub fn connect(url: &str, model: &str, max_context: usize) -> Result<Self> {
        let base = base_of(url);
        let a = agent(900);
        let show: Value = a
            .post(&format!("{base}/api/show"))
            .send_json(json!({ "model": model }))
            .map_err(|e| match e {
                ureq::Error::Status(404, _) => anyhow::anyhow!("Ollama at {base} has no model `{model}` - pull it first (ollama pull {model})"),
                ureq::Error::Status(code, r) => anyhow::anyhow!("Ollama at {base} refused `{model}`: HTTP {code} {}", r.into_string().unwrap_or_default()),
                other => anyhow::anyhow!("no Ollama server at {base} - is it running, and reachable from here? ({other})"),
            })?
            .into_json()?;
        let native = context_length_of(&show["model_info"]);
        let ctx = native.map_or(max_context, |n| n.min(max_context));
        let capabilities: Vec<&str> = show["capabilities"].as_array().map(|c| c.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
        Ok(Self { base, agent: a, model: model.to_string(), ctx, thinking: capabilities.contains(&"thinking") })
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn can_think(&self) -> bool {
        self.thinking
    }

    fn body(&self, prompt: &str, s: &Sampling) -> (Value, bool) {
        let (messages, think) = to_messages(prompt);
        let wants_thinking = think == Some(true) && self.thinking;
        let mut body = json!({
            "model": self.model,
            "messages": messages.iter().map(|(role, content)| json!({ "role": role, "content": content })).collect::<Vec<_>>(),
            "stream": true,
            "keep_alive": "30m",
            "options": { "num_ctx": self.ctx, "temperature": s.temperature, "top_p": s.top_p, "num_predict": s.n_predict, "stop": s.stop },
        });
        if self.thinking {
            body["think"] = json!(wants_thinking);
        }
        (body, wants_thinking)
    }
}

impl Llm for OllamaServer {
    fn complete(&self, prompt: &str, s: &Sampling, on_token: &mut dyn FnMut(&str)) -> Result<Completion> {
        let (body, wants_thinking) = self.body(prompt, s);
        let resp = self.agent.post(&format!("{}/api/chat", self.base)).send_json(body).map_err(|e| match e {
            ureq::Error::Status(code, r) => anyhow::anyhow!("Ollama refused the request: HTTP {code} {}", r.into_string().unwrap_or_default().chars().take(300).collect::<String>()),
            other => anyhow::anyhow!("Ollama at {} did not answer: {other}", self.base),
        })?;
        let started = Instant::now();
        let mut out = Completion::default();
        let mut in_think = false;
        let mut emit = |out: &mut Completion, piece: &str| {
            out.text.push_str(piece);
            on_token(piece);
        };
        for line in BufReader::new(resp.into_reader()).lines() {
            let line = line.context("the Ollama stream broke")?;
            if line.trim().is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(&line).with_context(|| format!("Ollama sent something that is not JSON: {}", line.chars().take(200).collect::<String>()))?;
            if let Some(err) = v.get("error") {
                bail!("Ollama reported an error: {err}");
            }
            // Reasoning is passed on - wrapped the way Qwen writes it - only when it was asked for; a model that
            // reasons anyway (it cannot be switched off) must not put its reasoning into a digit or a JSON reply.
            if let Some(t) = v["message"]["thinking"].as_str().filter(|t| !t.is_empty()) {
                if wants_thinking {
                    if !in_think {
                        emit(&mut out, "<think>\n");
                        in_think = true;
                    }
                    emit(&mut out, t);
                }
            }
            if let Some(c) = v["message"]["content"].as_str().filter(|c| !c.is_empty()) {
                if in_think {
                    emit(&mut out, "\n</think>\n\n");
                    in_think = false;
                }
                emit(&mut out, c);
            }
            if v["done"].as_bool() == Some(true) {
                out.prompt_tokens = v["prompt_eval_count"].as_u64().unwrap_or(0) as usize;
                out.prompt_ms = v["prompt_eval_duration"].as_f64().unwrap_or(0.0) / 1e6;
                out.gen_tokens = v["eval_count"].as_u64().unwrap_or(0) as usize;
                out.gen_ms = v["eval_duration"].as_f64().map(|n| n / 1e6).unwrap_or_else(|| started.elapsed().as_secs_f64() * 1000.0);
                out.truncated = v["done_reason"].as_str() == Some("length");
            }
        }
        Ok(out)
    }

    /// Ollama has no tokenizer endpoint. A conservative estimate (three characters a token; English runs nearer
    /// four) - it only sizes budgets, so erring high is safe.
    fn count_tokens(&self, text: &str) -> Result<usize> {
        Ok(text.chars().count().div_ceil(3))
    }
}

impl KvSlot for OllamaServer {
    fn save(&self, _file: &str) -> Result<SlotIo> {
        Ok(SlotIo::default())
    }

    fn restore(&self, _file: &str) -> Result<Option<SlotIo>> {
        Ok(None)
    }

    fn erase(&self) -> Result<()> {
        Ok(())
    }

    fn context_size(&self) -> usize {
        self.ctx
    }

    fn persists(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chatml;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    #[test]
    fn a_chatml_prompt_becomes_chat_messages_and_says_whether_to_think() {
        let p = format!("{}{}{}", chatml::system("You are the book."), chatml::user("Why?"), chatml::assistant_open());
        let (m, think) = to_messages(&p);
        assert_eq!(m, vec![("system".to_string(), "You are the book.".to_string()), ("user".to_string(), "Why?".to_string())]);
        assert_eq!(think, Some(false), "an empty think block means: answer directly");
        let p = format!("{}{}", chatml::user("How many?"), chatml::assistant_open_thinking());
        assert_eq!(to_messages(&p).1, Some(true));
        let p = format!("{}{}{}{}", chatml::user("Q"), chatml::assistant("A"), chatml::user("More?"), chatml::assistant_open());
        assert_eq!(to_messages(&p).0.iter().map(|(r, _)| r.as_str()).collect::<Vec<_>>(), vec!["user", "assistant", "user"]);
        assert_eq!(to_messages("plain text"), (vec![("user".to_string(), "plain text".to_string())], None));
    }

    fn serve(respond: impl Fn(&str, &str) -> String + Send + 'static) -> (String, Arc<Mutex<Vec<(String, String)>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { break };
                let mut buf = Vec::new();
                let mut chunk = [0u8; 8192];
                let (key, body) = loop {
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
                        let key = head.lines().next().unwrap_or("").split(' ').take(2).collect::<Vec<_>>().join(" ");
                        break (key, String::from_utf8_lossy(&buf[p + 4..]).to_string());
                    }
                    if n == 0 {
                        break (String::new(), String::new());
                    }
                };
                seen2.lock().unwrap().push((key.clone(), body.clone()));
                let payload = respond(&key, &body);
                let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}", payload.len());
            }
        });
        (base, seen)
    }

    fn standard(key: &str, _body: &str) -> String {
        match key {
            "POST /api/show" => r#"{"model_info":{"qwen3.context_length":40960},"capabilities":["completion","thinking"]}"#.into(),
            "POST /api/chat" => [
                r#"{"message":{"role":"assistant","content":"","thinking":"472 plus 10"},"done":false}"#,
                r#"{"message":{"role":"assistant","content":"Gryffindor"},"done":false}"#,
                r#"{"message":{"role":"assistant","content":" won."},"done":true,"done_reason":"stop","prompt_eval_count":120,"prompt_eval_duration":50000000,"eval_count":3,"eval_duration":90000000}"#,
            ]
            .join("\n"),
            "GET /api/tags" => r#"{"models":[
                {"name":"qwen3:14b","details":{"parameter_size":"14.8B","context_length":40960},"capabilities":["completion","thinking"]},
                {"name":"yi:6b","details":{"parameter_size":"6B","context_length":4096},"capabilities":["completion"]},
                {"name":"nomic-embed-text","details":{"parameter_size":"137M","context_length":2048},"capabilities":["embedding"]}]}"#
                .into(),
            _ => "{}".into(),
        }
    }

    #[test]
    fn the_window_is_the_models_own_context_capped_and_thinking_is_known() {
        let (base, _) = serve(standard);
        let o = OllamaServer::connect(&base, "qwen3:14b", 12288).unwrap();
        assert_eq!((o.context_size(), o.can_think(), o.persists()), (12288, true, false), "capped at what was allowed; nothing is saved");
        let o = OllamaServer::connect(&base, "qwen3:14b", 65536).unwrap();
        assert_eq!(o.context_size(), 40960, "never more than the model has");
    }

    #[test]
    fn a_completion_is_a_chat_with_the_window_and_the_reasoning_only_when_asked_for() {
        let (base, seen) = serve(standard);
        let o = OllamaServer::connect(&base, "qwen3:14b", 12288).unwrap();
        let mut streamed = String::new();
        let c = o.complete(&format!("{}{}", chatml::user("How many?"), chatml::assistant_open_thinking()), &Sampling::thinking(500), &mut |t| streamed.push_str(t)).unwrap();
        assert_eq!(c.text, "<think>\n472 plus 10\n</think>\n\nGryffindor won.", "reasoning arrives the way the rest of docuzent expects it");
        assert_eq!(streamed, c.text);
        assert_eq!((c.prompt_tokens, c.gen_tokens, c.truncated), (120, 3, false));
        let body: Value = serde_json::from_str(&seen.lock().unwrap().iter().rev().find(|(k, _)| k == "POST /api/chat").unwrap().1).unwrap();
        assert_eq!((body["think"].as_bool(), body["options"]["num_ctx"].as_u64(), body["model"].as_str()), (Some(true), Some(12288), Some("qwen3:14b")));
        // asked NOT to think: whatever reasoning comes is not passed on
        let c = o.complete(&chatml::ask("", "Score it."), &Sampling::precise(3), &mut |_| {}).unwrap();
        assert_eq!(c.text, "Gryffindor won.");
        let body: Value = serde_json::from_str(&seen.lock().unwrap().iter().rev().find(|(k, _)| k == "POST /api/chat").unwrap().1).unwrap();
        assert_eq!(body["think"].as_bool(), Some(false));
    }

    #[test]
    fn models_that_cannot_talk_or_hold_a_part_are_listed_as_unusable_with_the_reason() {
        let (base, _) = serve(standard);
        let models = OllamaServer::list(&base, 8192).unwrap();
        let get = |n: &str| models.iter().find(|m| m.name == n).unwrap().clone();
        assert!(get("qwen3:14b").usable);
        assert!(!get("yi:6b").usable && get("yi:6b").reason.unwrap().contains("4096"));
        assert!(!get("nomic-embed-text").usable);
        assert_eq!(models[0].name, "qwen3:14b", "usable ones first");
    }

    #[test]
    fn a_missing_server_or_model_says_what_to_do() {
        let e = OllamaServer::connect("http://127.0.0.1:1", "qwen3:14b", 12288).err().unwrap().to_string();
        assert!(e.contains("no Ollama server"), "{e}");
        assert!(!OllamaServer::reachable("http://127.0.0.1:1"));
    }
}
