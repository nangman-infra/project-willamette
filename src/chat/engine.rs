//! `ChatEngine` — stateful multi-turn dialogue runner.
//!
//! See [`super`] for the contract overview. Lifetime parameters: the
//! engine borrows a `ModelGraph<'a>` for its entire life; both `'a`
//! (the mmap-backed model bytes) and `'g` (the borrow of the graph)
//! must outlive the engine.

use crate::error::WillametteError;
use crate::model::cached_forward::forward_with_cache;
use crate::model::graph::ModelGraph;
use crate::model::kv_cache::KVCache;
use crate::model::lm_head::compute_logits_from_graph;
use crate::model::sampler::{Sampler, SamplingParams};
use crate::tokenizer::{PromptPart, Tokenizer};

/// Who said it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

/// `<|eot_id|>` — LLaMA-3 family "end of turn" id. The BitNet b1.58 2B
/// model inherits the same tokenizer (see `inspect.log`). Stop on it
/// in addition to the configured EOS.
const LLAMA3_EOT_ID: u32 = 128009;

pub struct ChatEngine<'g, 'a> {
    graph: &'g ModelGraph<'a>,
    tokenizer: Tokenizer,
    cache: KVCache,
    sampler: Sampler,
    history: Vec<ChatMessage>,
    /// Token position the NEXT prefill / generation step will write at.
    /// Equivalent to `cache.position() as u32` — we cache it so callers
    /// can introspect.
    next_pos: u32,
    /// Hard cap on cache + new tokens — must match the cache's
    /// `max_seq_len`.
    max_seq_len: usize,
    /// Optional system prompt; sits in front of the first user turn
    /// when present.
    system_prompt: Option<String>,
}

impl<'g, 'a> ChatEngine<'g, 'a> {
    /// Construct an engine. `max_seq_len` sizes the KV cache; choose
    /// it to comfortably exceed prompt + expected dialogue length
    /// (the engine errors out cleanly if the budget is exceeded).
    pub fn new(
        graph: &'g ModelGraph<'a>,
        tokenizer: Tokenizer,
        sampling: SamplingParams,
        max_seq_len: usize,
    ) -> Self {
        let n_layers = graph.layers.len();
        let kv_dim = graph.config.kv_dim as usize;
        Self {
            graph,
            tokenizer,
            cache: KVCache::new(n_layers, kv_dim, max_seq_len),
            sampler: Sampler::new(sampling),
            history: Vec::new(),
            next_pos: 0,
            max_seq_len,
            system_prompt: None,
        }
    }

    /// Replace the engine's sampling configuration (history reset
    /// included; the sampler's rolling history was tied to the old
    /// sampler).
    pub fn set_sampling(&mut self, sampling: SamplingParams) {
        self.sampler = Sampler::new(sampling);
        // Re-observe the existing history so repetition penalty still
        // sees what was already said.
        for msg in &self.history {
            // We can't recompute exact token ids cheaply; re-encoding
            // them here would be wrong because BPE merges may differ
            // from the in-stream ids the model actually emitted.
            // Leave the new sampler's history empty — repetition
            // penalty will only apply to tokens emitted from now on.
            let _ = msg;
        }
    }

    /// Set or clear the system prompt. Takes effect on the **next**
    /// turn (does not retroactively rewrite cache).
    pub fn set_system_prompt(&mut self, sys: Option<String>) {
        self.system_prompt = sys;
    }

    /// Clear conversation history and reset the KV cache. The next
    /// turn will be a fresh first turn (BOS prepended).
    pub fn reset(&mut self) {
        self.cache.reset();
        self.history.clear();
        self.next_pos = 0;
    }

    pub fn history(&self) -> &[ChatMessage] {
        &self.history
    }

    pub fn token_position(&self) -> u32 {
        self.next_pos
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Push one user message through the model and stream the
    /// assistant response. Returns the assembled response string
    /// (also appended to `self.history`).
    ///
    /// `tick` is called once per UTF-8-safe chunk of generated text.
    /// Use it to render to stdout, a TUI buffer, or anywhere else.
    pub fn send_user_message<F>(
        &mut self,
        user_text: &str,
        max_new_tokens: usize,
        mut tick: F,
    ) -> Result<String, WillametteError>
    where
        F: FnMut(&str),
    {
        // Stage 9-C — apply the real BitNet chat template:
        //
        //   {% for message in messages %}
        //     {% if loop.first %}{{ bos_token }}{% endif %}
        //     {% if message['role'] == 'user' %}
        //       Human: <content>\n\nBITNETAssistant: <eos_token>
        //     {% elif message['role'] == 'assistant' %}
        //       <content><eos_token>
        //     {% endif %}
        //   {% endfor %}
        //
        // Note the unusual position of `eos_token` — directly after
        // "BITNETAssistant: " (i.e. BEFORE the model writes its reply).
        // The model was trained on this; we must reproduce it byte-for-byte
        // via PromptPart::Special(eos_id) to avoid BPE-splitting the
        // 7-byte string "<|end_of_text|>" into 7 byte-level tokens.
        let bos = self.tokenizer.bos_id.ok_or_else(|| {
            WillametteError::UnsupportedTokenizer(
                "chat: bos_token_id is not set in tokenizer metadata".to_string(),
            )
        })?;
        // The GGUF metadata stores `tokenizer.ggml.eos_token_id = 128001`
        // (`<|end_of_text|>`), which is the document-level EOS used for
        // stopping generation. The Jinja template's `eos_token` variable,
        // however, refers to the *chat* turn boundary — in LLaMA-3 family
        // models (and BitNet b1.58 inherits the LLaMA-3 tokenizer), the
        // canonical turn EOS is `<|eot_id|>` (128009). Injecting 128001
        // before the assistant turn causes the model to predict BOS and
        // start a fresh document; 128009 keeps it in chat mode.
        // We use 128009 if the vocab is LLaMA-3 sized (>= 128010), else
        // fall back to the metadata-defined EOS.
        let chat_eos: u32 = if self.tokenizer.vocab_size() >= 128010 {
            LLAMA3_EOT_ID
        } else {
            self.tokenizer.eos_id.ok_or_else(|| {
                WillametteError::UnsupportedTokenizer(
                    "chat: eos_token_id is not set in tokenizer metadata".to_string(),
                )
            })?
        };

        // Materialise the text segments (lifetimes must outlive `parts`).
        let user_segment: String;
        let parts: Vec<PromptPart<'_>>;
        if self.history.is_empty() {
            // First turn — emit BOS, then "Human: ...BITNETAssistant: ",
            // then EOS. Optional system prompt is folded into the user
            // message because the template has no `system` role.
            user_segment = if let Some(sys) = &self.system_prompt {
                format!("Human: {}\n{}\n\nBITNETAssistant: ", sys, user_text)
            } else {
                format!("Human: {}\n\nBITNETAssistant: ", user_text)
            };
            parts = vec![
                PromptPart::Special(bos),
                PromptPart::Text(&user_segment),
                PromptPart::Special(chat_eos),
            ];
        } else {
            // Subsequent turn — close the previous assistant turn with
            // EOS, then a fresh "Human: ...BITNETAssistant: " segment,
            // then the EOS that precedes this turn's response.
            user_segment = format!("Human: {}\n\nBITNETAssistant: ", user_text);
            parts = vec![
                PromptPart::Special(chat_eos),
                PromptPart::Text(&user_segment),
                PromptPart::Special(chat_eos),
            ];
        }

        let prompt_tokens = self.tokenizer.encode_with_specials(&parts)?;
        if prompt_tokens.is_empty() {
            return Err(WillametteError::GgufParse(
                "chat: user fragment encoded to zero tokens".to_string(),
            ));
        }

        // Budget check up front: prompt + worst-case decode must fit.
        let need = self.cache.position() + prompt_tokens.len() + max_new_tokens;
        if need > self.max_seq_len {
            return Err(WillametteError::GgufParse(format!(
                "chat: context budget exceeded — cache.position={} + new prompt={} + max_new_tokens={} = {} > max_seq_len={}",
                self.cache.position(),
                prompt_tokens.len(),
                max_new_tokens,
                need,
                self.max_seq_len
            )));
        }

        // Prefill new user-side tokens into the cache.
        let mut last_hidden = Vec::new();
        for (i, &tid) in prompt_tokens.iter().enumerate() {
            let pos = self.next_pos + i as u32;
            last_hidden = forward_with_cache(self.graph, &mut self.cache, tid, pos)?;
            self.sampler.observe(tid);
        }
        self.next_pos += prompt_tokens.len() as u32;

        // Generate the assistant response with UTF-8-safe streaming.
        let mut response_text = String::new();
        let mut generated_count = 0usize;
        let mut pending_bytes: Vec<u8> = Vec::new();
        let mut emitted_up_to: usize = 0;

        for step in 0..max_new_tokens {
            let logits = compute_logits_from_graph(&last_hidden, self.graph)?;
            let next = self.sampler.sample(&logits)?;

            // Stop tokens.
            if Some(next) == self.tokenizer.eos_id || next == LLAMA3_EOT_ID {
                break;
            }

            // Stream the new bytes.
            if let Ok(more) = self.tokenizer.decode_to_bytes(&[next]) {
                pending_bytes.extend_from_slice(&more);
                let valid_end = match std::str::from_utf8(&pending_bytes) {
                    Ok(_) => pending_bytes.len(),
                    Err(e) => e.valid_up_to(),
                };
                if valid_end > emitted_up_to {
                    // SAFETY: bytes[..valid_end] is valid UTF-8 by
                    // `Utf8Error::valid_up_to`.
                    let chunk = unsafe {
                        std::str::from_utf8_unchecked(&pending_bytes[emitted_up_to..valid_end])
                    };
                    tick(chunk);
                    response_text.push_str(chunk);
                    emitted_up_to = valid_end;
                }
            }

            generated_count += 1;
            self.sampler.observe(next);

            // ALWAYS forward the just-emitted token into the cache —
            // unlike one-shot generation, chat needs the K/V of every
            // response token in the cache so the NEXT turn sees them.
            // The cost is one extra forward at the very last step, but
            // that's the price of continuity. (If we don't forward,
            // turn N+1 sees a phantom "the last response ended with
            // token T-1" view of history.)
            last_hidden =
                forward_with_cache(self.graph, &mut self.cache, next, self.next_pos)?;
            self.next_pos += 1;
            let _ = step;
        }

        // Flush any trailing incomplete UTF-8 suffix as U+FFFD so the
        // caller can see something was there (same convention as
        // `willamette run`).
        if emitted_up_to < pending_bytes.len() {
            tick("\u{FFFD}");
            response_text.push('\u{FFFD}');
        }

        // Persist the turn to history (raw user text + decoded
        // response — NOT the wrapped "Human: ..." prompt).
        self.history.push(ChatMessage {
            role: Role::User,
            content: user_text.to_string(),
        });
        self.history.push(ChatMessage {
            role: Role::Assistant,
            content: response_text.clone(),
        });
        let _ = generated_count;
        Ok(response_text)
    }
}
