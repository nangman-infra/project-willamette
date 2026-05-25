//! `ChatEngine` — stateful multi-turn dialogue runner.
//!
//! See [`super`] for the contract overview. Lifetime parameters: the
//! engine borrows a `ModelGraph<'a>` for its entire life; both `'a`
//! (the mmap-backed model bytes) and `'g` (the borrow of the graph)
//! must outlive the engine.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::error::WillametteError;
use crate::model::cached_forward::{forward_with_cache, forward_with_cache_progress};
use crate::model::graph::ModelGraph;
use crate::model::kv_cache::KVCache;
use crate::model::lm_head::compute_logits_from_graph;
use crate::model::sampler::{Sampler, SamplingParams};
use crate::tokenizer::{EncodeOptions, Tokenizer};

/// Shared atomic state between the engine (worker thread) and the
/// TUI (UI thread). High-frequency updates (layer-by-layer, token-by-
/// token) flow through these atomics rather than the `mpsc` channel
/// so the UI can poll at its own redraw cadence without flooding.
#[derive(Debug, Default)]
pub struct WorkerProgress {
    /// 0..n_layers while inside a forward, or `u32::MAX` when idle.
    pub current_layer: AtomicU32,
    /// Tokens emitted in the *current* turn (resets per turn).
    pub tokens_emitted: AtomicU32,
    /// Cap for the current turn (= max_new_tokens). 0 when idle.
    pub tokens_cap: AtomicU32,
    /// UNIX epoch nanoseconds when the current turn started. 0 = idle.
    pub turn_start_nanos: AtomicU64,
    /// Most-recently-computed KV cache size in bytes.
    pub kv_cache_bytes: AtomicU64,
    /// Cancel flag — set by UI when the user presses Esc mid-turn.
    /// The engine checks before each new token and exits cleanly.
    pub cancel_requested: AtomicBool,
}

impl WorkerProgress {
    pub fn new() -> Self {
        Self {
            current_layer: AtomicU32::new(u32::MAX),
            tokens_emitted: AtomicU32::new(0),
            tokens_cap: AtomicU32::new(0),
            turn_start_nanos: AtomicU64::new(0),
            kv_cache_bytes: AtomicU64::new(0),
            cancel_requested: AtomicBool::new(false),
        }
    }

    /// Reset for the start of a new turn.
    pub fn begin_turn(&self, cap: u32) {
        self.current_layer.store(u32::MAX, Ordering::Relaxed);
        self.tokens_emitted.store(0, Ordering::Relaxed);
        self.tokens_cap.store(cap, Ordering::Relaxed);
        self.cancel_requested.store(false, Ordering::Relaxed);
        let now_nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        self.turn_start_nanos.store(now_nanos, Ordering::Relaxed);
    }

    /// Mark the worker as idle (turn finished or aborted).
    pub fn end_turn(&self) {
        self.current_layer.store(u32::MAX, Ordering::Relaxed);
        self.turn_start_nanos.store(0, Ordering::Relaxed);
    }
}

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

/// Byte length of the longest entry in [`CHAT_STOP_SEQUENCES`].
/// Used as the size of the look-ahead buffer in
/// [`ChatEngine::stream_assistant_response`] so a stop sequence is
/// detected and truncated *before* its bytes ever reach the caller's
/// `tick` callback. We round up from 16 ("BITNETAssistant:".len()) to
/// 24 to give a safety margin for any future-added longer sequence.
pub(crate) const CHAT_STOP_LOOKAHEAD_BYTES: usize = 24;

/// Predicate for emoji / pictograph Unicode codepoints.
///
/// Strips most pictographs without touching CJK letters, math
/// symbols, currency, etc. Ranges covered (some intentionally
/// over-cover related symbol blocks):
///
/// * `U+1F300..U+1F9FF` — Misc Symbols and Pictographs, Emoticons,
///   Transport & Map, Geometric Shapes Extended, Supplemental
///   Symbols, Pictographs Extended-A. Includes skin-tone modifiers.
/// * `U+1FA00..U+1FAFF` — Symbols and Pictographs Extended-A/B.
/// * `U+1F1E6..U+1F1FF` — Regional Indicator Symbols (flag halves).
/// * `U+2600..U+27BF` — Misc Symbols + Dingbats (✨, ✅, ⚡, …).
/// * `U+200D` — Zero-width joiner (binds compound emoji).
/// * `U+FE00..U+FE0F` — Variation selectors (text vs emoji form).
pub(crate) fn is_emoji_char(c: char) -> bool {
    let cp = c as u32;
    matches!(
        cp,
        // Misc Symbols & Pictographs through Supplemental Symbols
        // — covers Emoticons U+1F600.. AND skin tones U+1F3FB..U+1F3FF
        // which sit inside this range.
        0x1F300..=0x1F9FF
            // Symbols and Pictographs Extended-A/B
            | 0x1FA00..=0x1FAFF
            // Regional Indicator Symbols (flag halves)
            | 0x1F1E6..=0x1F1FF
            // Misc Symbols + Dingbats (✨ ✅ ⚡ etc.)
            | 0x2600..=0x27BF
            // Zero-width joiner
            | 0x200D
            // Variation selectors (text-vs-emoji presentation flips)
            | 0xFE00..=0xFE0F
    )
}

/// Return a copy of `text` with every emoji/pictograph character
/// removed. Leaves CJK, Latin, control characters untouched.
pub(crate) fn strip_emoji_chars(text: &str) -> String {
    text.chars().filter(|c| !is_emoji_char(*c)).collect()
}

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
    /// Optional shared progress + cancel state. Stdio chat leaves
    /// this as `None`; the TUI installs an `Arc<WorkerProgress>` so
    /// it can show layer-by-layer + tok/s and trigger mid-turn
    /// cancellation.
    progress: Option<Arc<WorkerProgress>>,
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
            progress: None,
        }
    }

    /// Install a shared progress / cancel state. Used by the TUI to
    /// observe layer-wise progress and request mid-turn cancellation.
    pub fn set_worker_progress(&mut self, progress: Arc<WorkerProgress>) {
        self.progress = Some(progress);
    }

    /// Cheap accessors for the dashboard.
    pub fn config_n_layers(&self) -> u32 {
        self.graph.config.block_count
    }
    pub fn config_n_embd(&self) -> u32 {
        self.graph.config.embedding_length
    }
    pub fn config_vocab_size(&self) -> u32 {
        self.graph.config.vocab_size
    }
    pub fn config_architecture(&self) -> &str {
        &self.graph.config.architecture
    }
    pub fn config_kv_dim(&self) -> u32 {
        self.graph.config.kv_dim
    }
    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }
    /// Estimate KV cache memory in bytes given the current cache
    /// position. Two f32 tensors per layer per token (K and V),
    /// each of length kv_dim.
    pub fn estimate_kv_cache_bytes(&self) -> u64 {
        let layers = self.graph.layers.len() as u64;
        let kv_dim = self.graph.config.kv_dim as u64;
        let pos = self.next_pos as u64;
        layers * kv_dim * pos * 4 * 2
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

        if let Some(p) = &self.progress {
            p.begin_turn(max_new_tokens as u32);
        }

        let last_hidden = self.prefill_prompt_tokens(&prompt_tokens)?;
        let result = self.stream_assistant_response(last_hidden, max_new_tokens, &mut tick);

        if let Some(p) = &self.progress {
            p.end_turn();
        }

        let response_text = result?;

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

    /// Forward a single token while updating the optional
    /// `WorkerProgress.current_layer` after each transformer layer
    /// completes. Wraps the bare `forward_with_cache` for the
    /// no-progress case and `forward_with_cache_progress` for the
    /// observed case.
    fn forward_with_optional_progress(
        &mut self,
        token: u32,
        position: u32,
    ) -> Result<Vec<f32>, WillametteError> {
        if let Some(progress) = self.progress.clone() {
            forward_with_cache_progress(self.graph, &mut self.cache, token, position, |layer_idx| {
                progress.current_layer.store(layer_idx, Ordering::Relaxed);
            })
        } else {
            forward_with_cache(self.graph, &mut self.cache, token, position)
        }
    }

    /// Greedy / sampled token loop with three guarantees:
    ///
    /// 1. **UTF-8 safety** — multi-byte codepoints split across BPE
    ///    tokens stay buffered until complete.
    /// 2. **No emoji clutter** — pictograph Unicode codepoints are
    ///    filtered out of both the live stream and the recorded
    ///    history. The model still emits the underlying tokens (so
    ///    the KV cache stays in sync with what the model thinks it
    ///    said) but the *visible* output is text-only.
    /// 3. **No turn-boundary leak** — every emit is delayed by
    ///    [`CHAT_STOP_LOOKAHEAD_BYTES`] so we can scan for a
    ///    hallucinated `User:` / `BITNETAssistant:` / etc. *before*
    ///    those bytes reach the caller's `tick` callback. When a
    ///    stop sequence is detected we truncate the response and
    ///    discard the still-buffered tail — neither the screen nor
    ///    the history shows the leak.
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
        let mut tick_flushed_up_to: usize = 0;
        let turn_start = Instant::now();

        for _step in 0..max_new_tokens {
            // Cancellation check — set by UI on Esc.
            if let Some(p) = &self.progress {
                if p.cancel_requested.load(Ordering::Relaxed) {
                    break;
                }
            }

            let logits = compute_logits_from_graph(&last_hidden, self.graph)?;
            let next = self.sampler.sample(&logits)?;

            if Some(next) == self.tokenizer.eos_id || next == LLAMA3_EOT_ID {
                break;
            }

            self.append_token_bytes(
                next,
                &mut pending_bytes,
                &mut emitted_up_to,
                &mut response_text,
            );

            // Live tok/s accounting.
            if let Some(p) = &self.progress {
                let emitted = p.tokens_emitted.fetch_add(1, Ordering::Relaxed) + 1;
                let _ = emitted;
                let _ = turn_start; // turn_start_nanos already set by begin_turn
            }

            // Stop-sequence check on the WHOLE accumulated response,
            // not just the not-yet-tick'd tail. If the model has
            // started fabricating a turn boundary, both the visible
            // (already-tick'd) bytes and the still-buffered tail are
            // truncated; nothing further reaches the caller.
            if truncate_at_chat_stop_sequence(&mut response_text) {
                // The bytes we already emitted to `tick` are unrecoverable,
                // but anything past `tick_flushed_up_to` is still ours to
                // discard. Just stop — history is now clean.
                break;
            }

            // Tick everything that's safely past the look-ahead window.
            let safe_end = response_text
                .len()
                .saturating_sub(CHAT_STOP_LOOKAHEAD_BYTES);
            let safe_end = floor_char_boundary(&response_text, safe_end);
            if safe_end > tick_flushed_up_to {
                let chunk = &response_text[tick_flushed_up_to..safe_end];
                tick(chunk);
                tick_flushed_up_to = safe_end;
            }

            self.sampler.observe(next);
            last_hidden = self.forward_with_optional_progress(next, self.next_pos)?;
            self.next_pos += 1;

            // Update KV cache size estimate for the dashboard.
            if let Some(p) = &self.progress {
                p.kv_cache_bytes
                    .store(self.estimate_kv_cache_bytes(), Ordering::Relaxed);
            }
        }

        // End-of-generation: no more tokens are coming, so the
        // look-ahead buffer can't possibly grow into a stop sequence.
        // Flush whatever's left.
        if tick_flushed_up_to < response_text.len() {
            tick(&response_text[tick_flushed_up_to..]);
        }
        // Trailing incomplete UTF-8 suffix (Korean / emoji / CJK split
        // mid-codepoint) shows as U+FFFD so the user knows something
        // was cut off.
        if emitted_up_to < pending_bytes.len() {
            tick("\u{FFFD}");
            response_text.push('\u{FFFD}');
        }
        Ok(response_text)
    }

    /// Append the bytes of one decoded token to `pending_bytes`, then
    /// append the emoji-stripped valid-UTF-8 prefix into
    /// `response_text`. Does NOT call `tick`; the caller does that
    /// after the look-ahead window slides forward.
    fn append_token_bytes(
        &self,
        next: u32,
        pending_bytes: &mut Vec<u8>,
        emitted_up_to: &mut usize,
        response_text: &mut String,
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
        // Filter emoji codepoints. The model still emitted the
        // underlying tokens (cache stays in sync); we just don't
        // show or remember the visual clutter.
        response_text.push_str(&strip_emoji_chars(chunk));
        *emitted_up_to = valid_end;
    }
}

/// Snap `idx` down to the nearest UTF-8 character boundary in `s`.
/// Equivalent to the unstable `str::floor_char_boundary` — provided
/// inline so we don't need nightly.
fn floor_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

#[cfg(test)]
mod stop_sequence_tests {
    use super::{
        find_chat_stop_sequence, floor_char_boundary, is_emoji_char, strip_emoji_chars,
        truncate_at_chat_stop_sequence,
    };

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

    // ── emoji filter tests ──

    #[test]
    fn is_emoji_char_basic_pictographs() {
        // 😊 U+1F60A (Emoticons block)
        assert!(is_emoji_char('\u{1F60A}'));
        // 👍 U+1F44D
        assert!(is_emoji_char('\u{1F44D}'));
        // 🤖 U+1F916
        assert!(is_emoji_char('\u{1F916}'));
        // ✨ U+2728 (Dingbats)
        assert!(is_emoji_char('\u{2728}'));
        // 🏻 U+1F3FB (skin tone modifier)
        assert!(is_emoji_char('\u{1F3FB}'));
        // 🇰 U+1F1F0 (regional indicator)
        assert!(is_emoji_char('\u{1F1F0}'));
    }

    #[test]
    fn is_emoji_char_leaves_text_alone() {
        // ASCII
        assert!(!is_emoji_char('a'));
        assert!(!is_emoji_char('!'));
        // CJK
        assert!(!is_emoji_char('한'));
        assert!(!is_emoji_char('国'));
        assert!(!is_emoji_char('日'));
        // Math/currency
        assert!(!is_emoji_char('∑'));
        assert!(!is_emoji_char('€'));
    }

    #[test]
    fn strip_emoji_removes_trailing_clutter() {
        let raw = "Hello! How can I assist you today? 😊👍🏻💬📚✨";
        assert_eq!(
            strip_emoji_chars(raw),
            "Hello! How can I assist you today? "
        );
    }

    #[test]
    fn strip_emoji_preserves_korean() {
        let raw = "안녕하세요 👋 친구!";
        assert_eq!(strip_emoji_chars(raw), "안녕하세요  친구!");
    }

    #[test]
    fn strip_emoji_handles_zwj_sequences() {
        // 👨‍💻 = U+1F468 + U+200D + U+1F4BB ("man technologist")
        let raw = "Hi 👨\u{200D}💻 there";
        assert_eq!(strip_emoji_chars(raw), "Hi  there");
    }

    // ── floor_char_boundary tests ──

    #[test]
    fn floor_char_boundary_on_ascii_is_identity() {
        let s = "hello";
        assert_eq!(floor_char_boundary(s, 3), 3);
        assert_eq!(floor_char_boundary(s, 0), 0);
        assert_eq!(floor_char_boundary(s, 5), 5);
    }

    #[test]
    fn floor_char_boundary_snaps_back_inside_codepoint() {
        // "한" = U+D55C = 0xED 0x95 0x9C (3 bytes)
        let s = "한국";
        // Byte 1 is inside the first '한' codepoint — must snap to 0.
        assert_eq!(floor_char_boundary(s, 1), 0);
        assert_eq!(floor_char_boundary(s, 2), 0);
        // Byte 3 is the start of '국' — already a boundary.
        assert_eq!(floor_char_boundary(s, 3), 3);
        assert_eq!(floor_char_boundary(s, 4), 3);
        assert_eq!(floor_char_boundary(s, 5), 3);
        assert_eq!(floor_char_boundary(s, 6), 6);
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
