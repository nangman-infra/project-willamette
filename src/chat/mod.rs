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
//! * Generation stops on `tokenizer.eos_id`, ChatML `<|im_end|>`, on
//!   `<|eot_id|>` (128009 for LLaMA-3 family), or after `max_new_tokens`.
//!
//! BitNet uses its empirically validated `Human:/BITNETAssistant:` bridge;
//! compatible classic-Llama instruct models use incremental ChatML markers.
//! Slash-command parsing lives in the CLI/TUI harness.

pub(crate) mod dashboard;
pub mod engine;
pub(crate) mod input_editor;
pub(crate) mod markdown;
pub(crate) mod sysmon;
pub mod tui;

pub use engine::{ChatEngine, ChatMessage, Role};
pub use tui::{run_tui, run_tui_with_model_info};
