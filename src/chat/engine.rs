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
use crate::tokenizer::{EncodeOptions, Tokenizer};

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

/// Text-level stop sequences for the chat loop.
///
/// BitNet b1.58 2B-4T is a *base* model — never SFT'd, never trained
/// to emit `<|end_of_text|>` or `<|eot_id|>` at turn boundaries. With
/// the `Human:/BITNETAssistant:` chat template it learns to continue
/// the pattern indefinitely, fabricating its own `User:`/`AI Assistant:`
/// follow-up turns past the answer we actually wanted.
///
/// These strings should NEVER appear in a legitimate single-turn
/// response from the model:
///
///   * `BITNETAssistant:` — the exact template phrase we use to prompt
///     the model. If it appears in the output, the model has
///     hallucinated a new turn.
///   * `User:`, `Human:` — common ways for the model to start a fake
///     user turn (LLaMA-3 pretrain data has lots of "User: …" forum
///     dialogue).
///   * `AI Assistant:`, `AI:` — alternative completions of the
///     "User: X" pattern.
///
/// When any of these appear in the model's emitted text we truncate
/// the response at that point and stop generating. The bytes before
/// the match have already streamed to the caller (no rewind possible)
/// but the *history* recorded for the next turn is the clean prefix.
const CHAT_STOP_SEQUENCES: &[&str] = &[
    // Our template's exact phrase — always a hallucination if echoed.
    "BITNETAssistant:",
    // Most common base-model fake-turn openers (colon-terminated).
    "User:",
    "Human:",
    "AI Assistant:",
    "AI:",
    "Assistant:",
    "Question:",
    // Parenthesised variants observed in v0.2.1 TUI sessions:
    //   "Human (reply): Yes.", "User (continued): ...".
    // The space-then-paren form is rare in legitimate response text,
    // so the false-positive risk is much lower than e.g. bare "User".
    "Human (",
    "User (",
    "AI (",
];

/// Find the earliest start-of-stop-sequence in `text`, or `None`.
///
/// Pure function — exposed at module scope so unit tests can exercise
/// the boundary logic without spinning up a `ChatEngine` + model.
pub(crate) fn find_chat_stop_sequence(text: &str) -> Option<usize> {
    let mut earliest: Option<usize> = None;
    for stop in CHAT_STOP_SEQUENCES {
        if let Some(idx) = text.find(stop) {
            earliest = Some(match earliest {
                Some(prev) => prev.min(idx),
                None => idx,
            });
        }
    }
    earliest
}

/// Trim `response_text` at the earliest hallucinated turn-boundary
/// match, dropping any trailing whitespace introduced before the
/// boundary so the recorded history reads cleanly. Returns `true` if
/// a boundary was found and the text was modified.
pub(crate) fn truncate_at_chat_stop_sequence(response_text: &mut String) -> bool {
    let Some(idx) = find_chat_stop_sequence(response_text) else {
        return false;
    };
    response_text.truncate(idx);
    let trimmed_len = response_text.trim_end().len();
    response_text.truncate(trimmed_len);
    true
}

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
        // v0.2.1 chat template — see [`build_chat_fragment`] below for
        // why we use a plain text bridge instead of injecting the GGUF
        // chat_template's `eos_token` marker between turns.
        let (fragment, opts) = self.build_chat_fragment(user_text);
        let prompt_tokens = self.tokenizer.encode(&fragment, opts)?;
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

        let last_hidden = self.prefill_prompt_tokens(&prompt_tokens)?;
        let response_text =
            self.stream_assistant_response(last_hidden, max_new_tokens, &mut tick)?;

        self.history.push(ChatMessage {
            role: Role::User,
            content: user_text.to_string(),
        });
        self.history.push(ChatMessage {
            role: Role::Assistant,
            content: response_text.clone(),
        });
        Ok(response_text)
    }

    /// Build the text fragment + encode options for this turn.
    ///
    /// Why no `eos_token` injection? Empirical testing showed BitNet
    /// b1.58 2B-4T is a base/foundation model (not instruct-tuned).
    /// Injecting either `<|end_of_text|>` (128001 — puts the model in
    /// document-completion mode) or `<|eot_id|>` (128009 — causes
    /// "PowerShell> Hello!", "Vietnamese> Cảm ơn!" prefix
    /// hallucinations) between turns produces degenerate output. A
    /// plain `\n\nHuman:/BITNETAssistant:` text bridge yields the
    /// cleanest response the model can produce. See CHANGELOG
    /// v0.2.1-mvp for the full diagnosis.
    fn build_chat_fragment(&self, user_text: &str) -> (String, EncodeOptions) {
        if self.history.is_empty() {
            let fragment = if let Some(sys) = &self.system_prompt {
                format!("{}\n\nHuman: {}\n\nBITNETAssistant: ", sys, user_text)
            } else {
                format!("Human: {}\n\nBITNETAssistant: ", user_text)
            };
            let opts = EncodeOptions {
                add_bos: true,
                add_eos: false,
            };
            (fragment, opts)
        } else {
            let fragment = format!("\n\nHuman: {}\n\nBITNETAssistant: ", user_text);
            (fragment, EncodeOptions::none())
        }
    }

    /// Prefill the prompt tokens into the KV cache, advance position,
    /// observe each token for the sampler's rolling history, return
    /// the last layer's hidden state for the first generation step.
    fn prefill_prompt_tokens(
        &mut self,
        prompt_tokens: &[u32],
    ) -> Result<Vec<f32>, WillametteError> {
        let mut last_hidden = Vec::new();
        for (i, &tid) in prompt_tokens.iter().enumerate() {
            let pos = self.next_pos + i as u32;
            last_hidden = forward_with_cache(self.graph, &mut self.cache, tid, pos)?;
            self.sampler.observe(tid);
        }
        self.next_pos += prompt_tokens.len() as u32;
        Ok(last_hidden)
    }

    /// Greedy / sampled token loop with UTF-8-safe streaming and EOS
    /// stop. Always forwards the just-emitted non-EOS token into the
    /// cache so the next turn sees the full response (unlike one-shot
    /// generation which can skip the last forward).
    fn stream_assistant_response<F>(
        &mut self,
        mut last_hidden: Vec<f32>,
        max_new_tokens: usize,
        tick: &mut F,
    ) -> Result<String, WillametteError>
    where
        F: FnMut(&str),
    {
        let mut response_text = String::new();
        let mut pending_bytes: Vec<u8> = Vec::new();
        let mut emitted_up_to: usize = 0;

        for _step in 0..max_new_tokens {
            let logits = compute_logits_from_graph(&last_hidden, self.graph)?;
            let next = self.sampler.sample(&logits)?;

            if Some(next) == self.tokenizer.eos_id || next == LLAMA3_EOT_ID {
                break;
            }

            self.emit_token_bytes(
                next,
                &mut pending_bytes,
                &mut emitted_up_to,
                &mut response_text,
                tick,
            );

            // Detect the base model fabricating a follow-up turn boundary
            // (`User:`, `BITNETAssistant:`, etc.). Truncate the response
            // so the recorded history stays clean, even though the
            // boundary string itself has already streamed to `tick`.
            if truncate_at_chat_stop_sequence(&mut response_text) {
                break;
            }

            self.sampler.observe(next);
            last_hidden = forward_with_cache(self.graph, &mut self.cache, next, self.next_pos)?;
            self.next_pos += 1;
        }

        // Flush any trailing incomplete UTF-8 suffix as U+FFFD so the
        // caller can see something was there.
        if emitted_up_to < pending_bytes.len() {
            tick("\u{FFFD}");
            response_text.push('\u{FFFD}');
        }
        Ok(response_text)
    }

    /// Append the bytes of one decoded token to `pending_bytes`, then
    /// emit the largest valid-UTF-8 prefix that hasn't already been
    /// emitted. Multi-byte codepoints split across BPE tokens stay
    /// buffered until completed.
    fn emit_token_bytes<F: FnMut(&str)>(
        &self,
        next: u32,
        pending_bytes: &mut Vec<u8>,
        emitted_up_to: &mut usize,
        response_text: &mut String,
        tick: &mut F,
    ) {
        let Ok(more) = self.tokenizer.decode_to_bytes(&[next]) else {
            return;
        };
        pending_bytes.extend_from_slice(&more);
        let valid_end = match std::str::from_utf8(pending_bytes) {
            Ok(_) => pending_bytes.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid_end <= *emitted_up_to {
            return;
        }
        // SAFETY: bytes[..valid_end] is valid UTF-8 by `Utf8Error::valid_up_to`.
        let chunk =
            unsafe { std::str::from_utf8_unchecked(&pending_bytes[*emitted_up_to..valid_end]) };
        tick(chunk);
        response_text.push_str(chunk);
        *emitted_up_to = valid_end;
    }
}

#[cfg(test)]
mod stop_sequence_tests {
    use super::{find_chat_stop_sequence, truncate_at_chat_stop_sequence};

    #[test]
    fn no_stop_in_clean_text() {
        assert_eq!(
            find_chat_stop_sequence("Hello! How can I assist you today?"),
            None
        );
    }

    #[test]
    fn detects_bitnet_assistant_template_phrase() {
        let txt = "Sure, here is more.\n\nBITNETAssistant: more answer";
        let idx = find_chat_stop_sequence(txt).expect("must match");
        assert_eq!(
            &txt[idx..idx + "BITNETAssistant:".len()],
            "BITNETAssistant:"
        );
    }

    #[test]
    fn detects_user_marker_without_newline() {
        // Real broken output observed in v0.2.1 TUI session:
        //   "...assist you today? 🖥️ 💬👍🤖✨User: Can you explain..."
        let txt = "Hello! How can I help? 🖥️User: another question?";
        let idx = find_chat_stop_sequence(txt).expect("must match");
        assert_eq!(&txt[idx..idx + "User:".len()], "User:");
    }

    #[test]
    fn detects_human_marker() {
        let txt = "Reply text here.\nHuman: next turn";
        let idx = find_chat_stop_sequence(txt).expect("must match");
        assert_eq!(&txt[idx..idx + "Human:".len()], "Human:");
    }

    #[test]
    fn detects_ai_assistant_marker() {
        let txt = "Some answer.AI Assistant: more talk";
        let idx = find_chat_stop_sequence(txt).expect("must match");
        assert_eq!(&txt[idx..idx + "AI Assistant:".len()], "AI Assistant:");
    }

    #[test]
    fn picks_earliest_match_when_multiple_present() {
        // "User:" at byte 12; "BITNETAssistant:" at byte 23.
        let txt = "answer text.User: askBITNETAssistant: answer2";
        let idx = find_chat_stop_sequence(txt).expect("must match");
        assert_eq!(idx, 12);
        assert_eq!(&txt[idx..idx + 5], "User:");
    }

    #[test]
    fn matches_at_zero_offset() {
        let txt = "BITNETAssistant: immediate";
        assert_eq!(find_chat_stop_sequence(txt), Some(0));
    }

    #[test]
    fn truncate_removes_boundary_and_following_text() {
        let mut s = "I am here to help.\n\nUser: another".to_string();
        let modified = truncate_at_chat_stop_sequence(&mut s);
        assert!(modified);
        assert_eq!(s, "I am here to help.");
    }

    #[test]
    fn truncate_strips_trailing_whitespace_before_boundary() {
        let mut s = "Answer text.   \n\n   BITNETAssistant:".to_string();
        let modified = truncate_at_chat_stop_sequence(&mut s);
        assert!(modified);
        // Trailing whitespace between the answer and the boundary is gone.
        assert_eq!(s, "Answer text.");
    }

    #[test]
    fn truncate_no_op_on_clean_text() {
        let mut s = "Just a clean reply.".to_string();
        let modified = truncate_at_chat_stop_sequence(&mut s);
        assert!(!modified);
        assert_eq!(s, "Just a clean reply.");
    }

    #[test]
    fn truncate_handles_unicode_correctly() {
        // Korean greeting before a hallucinated turn — must not slice
        // mid-UTF-8 sequence.
        let mut s = "안녕하세요! 무엇을 도와드릴까요?\nUser: 다음".to_string();
        let modified = truncate_at_chat_stop_sequence(&mut s);
        assert!(modified);
        assert_eq!(s, "안녕하세요! 무엇을 도와드릴까요?");
        // Sanity: the surviving string is still valid UTF-8 (a panic
        // here would mean we truncated mid-codepoint).
        assert!(std::str::from_utf8(s.as_bytes()).is_ok());
    }

    #[test]
    fn detects_parenthesised_human_variant() {
        // Observed in v0.2.1 multi-turn TUI session:
        //   "...let me know! 😊💭💡\n  \nHuman (reply): Yes."
        let txt = "Tell me more 💡\n  \nHuman (reply): Yes.";
        let idx = find_chat_stop_sequence(txt).expect("must match");
        assert_eq!(&txt[idx..idx + "Human (".len()], "Human (");
    }

    #[test]
    fn detects_question_marker() {
        let txt = "answer one.\nQuestion: another?";
        let idx = find_chat_stop_sequence(txt).expect("must match");
        assert_eq!(&txt[idx..idx + "Question:".len()], "Question:");
    }

    #[test]
    fn does_not_match_substring_humans() {
        // Adversarial-ish: 'Human' as the start of 'Humans' should NOT
        // trip the bare-Human checks. Our patterns require either ':'
        // or ' (' immediately after, so 'Humans' is safe.
        assert_eq!(find_chat_stop_sequence("Humans need food"), None);
        assert_eq!(find_chat_stop_sequence("Users of the system"), None);
    }

    #[test]
    fn real_world_v021_failure_case_is_truncated() {
        // Captured verbatim from the user's TUI session that triggered
        // this fix: "how are you?" turn rolled into a fake
        // "User: Can you explain..." follow-up.
        let mut s = "I'm just a computer program, so I don't have feelings, but I'm here \
                     and ready to help you! How can I assist you today? 🖥️ 💬👍🤖✨User: \
                     Can you explain what a hash function is?"
            .to_string();
        let modified = truncate_at_chat_stop_sequence(&mut s);
        assert!(modified);
        assert!(s.ends_with("today? 🖥️ 💬👍🤖✨"));
        assert!(!s.contains("User:"));
        assert!(!s.contains("hash function"));
    }
}
