//! Chat engine — multi-turn dialogue over a single loaded model.
//!
//! Stage 9-A scope: the runtime engine only. A simple stdin/stdout
//! harness exists in `src/main.rs::cmd_chat` to drive it from the CLI;
//! a richer ratatui TUI ships in Stage 9-E and reuses the same
//! [`ChatEngine`] API.
//!
//! What this module guarantees:
//!
//! * Model + tokenizer + ModelGraph are loaded **exactly once** per
//!   `ChatEngine` instance.
//! * The per-layer [`KVCache`](crate::model::kv_cache::KVCache) is
//!   reused across turns — only the new user-message tokens (plus the
//!   model's generated response tokens) get prefilled into the cache
//!   per turn, not the whole transcript.
//! * Output streams to the caller via a `FnMut(&str)` tick, in UTF-8
//!   safe chunks that respect multi-byte character boundaries.
//! * Generation stops on `tokenizer.eos_id`, on `<|eot_id|>`
//!   (128009 for LLaMA-3 family), or after `max_new_tokens`.
//!
//! What this module does **not** do at Stage 9-A:
//!
//! * Apply the precise BitNet chat template with `<|end_of_text|>`
//!   injected between turns — that needs Stage 9-B
//!   (`Tokenizer::encode_with_specials`) and Stage 9-C (template
//!   wiring). Stage 9-A uses a simpler `Human:/BITNETAssistant:`
//!   bridge that the model handles well in practice.
//! * Slash-command parsing — that lives in the harness (Stage 9-D).

pub mod engine;
pub mod tui;

pub use engine::{ChatEngine, ChatMessage, Role};
pub use tui::run_tui;
