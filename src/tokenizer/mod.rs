//! GGUF-backed byte-level BPE tokenizer.
//!
//! ## Stage 2 scope
//!
//! Implements only `tokenizer.ggml.model = "gpt2"` (LLaMA 3 family byte-level
//! BPE). All vocabulary, merges, and special-token IDs come from the model's
//! GGUF metadata — there is no hand-written vocabulary, no ASCII fallback,
//! and no unknown-token degradation path.
//!
//! Any other tokenizer model (e.g. `"llama"` SentencePiece, `"bert"`,
//! `"rwkv"`) results in [`WillametteError::UnsupportedTokenizer`].
//!
//! ## Roundtrip guarantee
//!
//! For any UTF-8 input `s`, `decode(encode(s, no_special_tokens)) == s` byte
//! for byte. This is verified by [`crate::tokenizer`] integration tests
//! against the real `ggml-model-i2_s.gguf` (see
//! `tests/tokenizer_roundtrip.rs`).
//!
//! ## What this module does NOT do
//!
//! * Recognize literal special-token strings (e.g. `<|begin_of_text|>` in
//!   input text) as the BOS token id during encoding. Use the `add_bos`
//!   encode option to inject BOS.
//! * Re-implement Meta's chat template. The Jinja template is exposed via
//!   `tokenizer.chat_template` metadata; consumers can render it externally.
//! * Apply Unicode normalization (NFC/NFD). LLaMA 3 BPE is byte-level so
//!   normalization is not part of the algorithm.

use std::collections::HashMap;

use crate::error::WillametteError;
use crate::gguf::reader::GgufValue;

mod bpe;
mod byte_unicode;
mod pretokenize;

use bpe::Bpe;
use byte_unicode::ByteUnicode;

#[derive(Debug, Clone, Copy)]
pub struct EncodeOptions {
    pub add_bos: bool,
    pub add_eos: bool,
}

impl EncodeOptions {
    pub fn none() -> Self {
        Self {
            add_bos: false,
            add_eos: false,
        }
    }
}

pub struct Tokenizer {
    byte_unicode: ByteUnicode,
    bpe: Bpe,
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub pad_id: Option<u32>,
    pub default_add_bos: bool,
    pub default_add_eos: bool,
    pub model_type: String,
}

impl Tokenizer {
    /// Build a tokenizer purely from a GGUF metadata map.
    ///
    /// This is the **only** public constructor. There is no `Default`, no
    /// `new()`, no synthetic-vocab path. If the metadata does not describe a
    /// supported tokenizer, returns [`WillametteError::UnsupportedTokenizer`].
    pub fn from_gguf_metadata(meta: &HashMap<String, GgufValue>) -> Result<Self, WillametteError> {
        let model_type = required_str(meta, "tokenizer.ggml.model")?.to_string();
        if model_type != "gpt2" {
            return Err(WillametteError::UnsupportedTokenizer(format!(
                "Stage 2 only supports tokenizer.ggml.model = \"gpt2\" \
                 (byte-level BPE, LLaMA 3 family); got \"{}\"",
                model_type
            )));
        }

        let id_to_token = load_string_array(meta, "tokenizer.ggml.tokens")?;
        if id_to_token.is_empty() {
            return Err(WillametteError::UnsupportedTokenizer(
                "tokenizer.ggml.tokens is empty".to_string(),
            ));
        }

        let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(id_to_token.len());
        for (i, tok) in id_to_token.iter().enumerate() {
            // If duplicates exist, the latest id wins. That's intentional —
            // upstream sometimes has the special-token aliasing convention.
            token_to_id.insert(tok.clone(), i as u32);
        }

        let merges_raw = load_string_array(meta, "tokenizer.ggml.merges")?;
        let mut merge_ranks: HashMap<(String, String), u32> =
            HashMap::with_capacity(merges_raw.len());
        for (rank, merge) in merges_raw.iter().enumerate() {
            // Each merge is "PART_A PART_B" — exactly one ASCII space separator.
            let mut parts = merge.splitn(2, ' ');
            let a = parts.next().ok_or_else(|| {
                WillametteError::UnsupportedTokenizer(format!(
                    "merges[{}] malformed (empty): {:?}",
                    rank, merge
                ))
            })?;
            let b = parts.next().ok_or_else(|| {
                WillametteError::UnsupportedTokenizer(format!(
                    "merges[{}] malformed (no second part): {:?}",
                    rank, merge
                ))
            })?;
            merge_ranks.insert((a.to_string(), b.to_string()), rank as u32);
        }

        let byte_unicode = ByteUnicode::new();

        // Every base byte must be present as its own token. This guarantees
        // that BPE encoding never produces an OOV symbol.
        for b in 0u8..=255 {
            let s = byte_unicode.encode_byte(b).to_string();
            if !token_to_id.contains_key(&s) {
                return Err(WillametteError::UnsupportedTokenizer(format!(
                    "vocab is missing base byte token for 0x{:02X} (\"{}\") — \
                     not a complete byte-level BPE vocabulary",
                    b, s
                )));
            }
        }

        let bos_id = u32_or_none(meta, "tokenizer.ggml.bos_token_id");
        let eos_id = u32_or_none(meta, "tokenizer.ggml.eos_token_id");
        let pad_id = u32_or_none(meta, "tokenizer.ggml.padding_token_id");

        let default_add_bos = bool_or_default(meta, "tokenizer.ggml.add_bos_token", false);
        let default_add_eos = bool_or_default(meta, "tokenizer.ggml.add_eos_token", false);

        Ok(Self {
            byte_unicode,
            bpe: Bpe::new(merge_ranks),
            id_to_token,
            token_to_id,
            bos_id,
            eos_id,
            pad_id,
            default_add_bos,
            default_add_eos,
            model_type,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn default_encode_options(&self) -> EncodeOptions {
        EncodeOptions {
            add_bos: self.default_add_bos,
            add_eos: self.default_add_eos,
        }
    }

    /// Encode `text` into token IDs.
    ///
    /// * Pre-tokenizes via the GPT-2 default regex (see
    ///   [`pretokenize`]). This is lossless: chunks concatenate to the input.
    /// * For each chunk, maps every byte through the byte→unicode table, then
    ///   applies BPE merges greedily by rank.
    /// * Optionally prepends `bos_id` and/or appends `eos_id` per
    ///   [`EncodeOptions`].
    pub fn encode(&self, text: &str, options: EncodeOptions) -> Result<Vec<u32>, WillametteError> {
        let mut ids = Vec::new();

        if options.add_bos {
            let bos = self.bos_id.ok_or_else(|| {
                WillametteError::UnsupportedTokenizer(
                    "add_bos requested but bos_token_id is not set in metadata".to_string(),
                )
            })?;
            ids.push(bos);
        }

        for chunk in pretokenize::gpt2_pretokenize(text) {
            let symbols: Vec<String> = chunk
                .as_bytes()
                .iter()
                .map(|&b| self.byte_unicode.encode_byte(b).to_string())
                .collect();

            let merged = self.bpe.encode_word(symbols);
            for tok in &merged {
                match self.token_to_id.get(tok) {
                    Some(&id) => ids.push(id),
                    None => {
                        return Err(WillametteError::UnsupportedTokenizer(format!(
                            "BPE produced symbol {:?} not in vocab \
                             (chunk: {:?}). This should be impossible with a \
                             complete byte-level vocab.",
                            tok, chunk
                        )));
                    }
                }
            }
        }

        if options.add_eos {
            let eos = self.eos_id.ok_or_else(|| {
                WillametteError::UnsupportedTokenizer(
                    "add_eos requested but eos_token_id is not set in metadata".to_string(),
                )
            })?;
            ids.push(eos);
        }

        Ok(ids)
    }

    /// Decode token IDs to a raw byte stream (no UTF-8 validation).
    ///
    /// Useful when generation may have stopped in the middle of a
    /// multi-byte UTF-8 character — the raw bytes are always
    /// recoverable. Callers wanting a `String` should use
    /// [`Tokenizer::decode`] (strict) or
    /// [`Tokenizer::decode_lossy`] (replaces invalid suffix with U+FFFD).
    pub fn decode_to_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, WillametteError> {
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 2);
        for &id in ids {
            let token_str = self.id_to_token.get(id as usize).ok_or_else(|| {
                WillametteError::UnsupportedTokenizer(format!(
                    "token id {} out of vocab range (size = {})",
                    id,
                    self.id_to_token.len()
                ))
            })?;
            for c in token_str.chars() {
                let b = self.byte_unicode.decode_char(c).ok_or_else(|| {
                    WillametteError::UnsupportedTokenizer(format!(
                        "token {:?} (id {}) contains char '{}' (U+{:04X}) \
                         with no byte-unicode inverse",
                        token_str, id, c, c as u32
                    ))
                })?;
                bytes.push(b);
            }
        }
        Ok(bytes)
    }

    /// Decode token IDs to text, replacing any trailing incomplete
    /// UTF-8 byte sequence with `U+FFFD` (replacement character). This
    /// is the right choice when generation may have been truncated
    /// mid-character (e.g. `max_new_tokens` reached during a 3-byte
    /// Korean codepoint).
    ///
    /// Internal multi-byte sequences that are well-formed are
    /// preserved exactly. Only an incomplete suffix is replaced.
    pub fn decode_lossy(&self, ids: &[u32]) -> Result<String, WillametteError> {
        let bytes = self.decode_to_bytes(ids)?;
        match std::str::from_utf8(&bytes) {
            Ok(_) => Ok(unsafe { String::from_utf8_unchecked(bytes) }),
            Err(e) => {
                let valid_end = e.valid_up_to();
                // SAFETY: bytes[..valid_end] is the maximal valid UTF-8 prefix
                // by definition of `Utf8Error::valid_up_to`.
                let head =
                    unsafe { std::str::from_utf8_unchecked(&bytes[..valid_end]).to_string() };
                if valid_end < bytes.len() {
                    Ok(format!("{}\u{FFFD}", head))
                } else {
                    Ok(head)
                }
            }
        }
    }

    /// Decode token IDs back to UTF-8 text. Strict: fails if the
    /// concatenated bytes are not valid UTF-8 (e.g. generation stopped
    /// mid-multi-byte-character). Use [`Tokenizer::decode_lossy`] for
    /// generation streams that may be truncated.
    ///
    /// Special tokens (e.g. BOS) decode to their literal display string
    /// (e.g. `"<|begin_of_text|>"`); the caller chooses whether to keep them.
    pub fn decode(&self, ids: &[u32]) -> Result<String, WillametteError> {
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 2);
        for &id in ids {
            let token_str = self.id_to_token.get(id as usize).ok_or_else(|| {
                WillametteError::UnsupportedTokenizer(format!(
                    "token id {} out of vocab range (size = {})",
                    id,
                    self.id_to_token.len()
                ))
            })?;
            for c in token_str.chars() {
                let b = self.byte_unicode.decode_char(c).ok_or_else(|| {
                    WillametteError::UnsupportedTokenizer(format!(
                        "token {:?} (id {}) contains char '{}' (U+{:04X}) \
                         with no byte-unicode inverse",
                        token_str, id, c, c as u32
                    ))
                })?;
                bytes.push(b);
            }
        }
        String::from_utf8(bytes).map_err(|e| {
            WillametteError::UnsupportedTokenizer(format!(
                "decoded bytes are not valid UTF-8: {}",
                e
            ))
        })
    }

    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }
}

// ── small helpers ──

fn required_str<'a>(
    meta: &'a HashMap<String, GgufValue>,
    key: &str,
) -> Result<&'a str, WillametteError> {
    meta.get(key).and_then(|v| v.as_str()).ok_or_else(|| {
        WillametteError::UnsupportedTokenizer(format!(
            "missing or non-string metadata key: {}",
            key
        ))
    })
}

fn load_string_array(
    meta: &HashMap<String, GgufValue>,
    key: &str,
) -> Result<Vec<String>, WillametteError> {
    let value = meta.get(key).ok_or_else(|| {
        WillametteError::UnsupportedTokenizer(format!("missing metadata key: {}", key))
    })?;
    match value {
        GgufValue::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for (i, item) in arr.iter().enumerate() {
                match item {
                    GgufValue::Str(s) => out.push(s.clone()),
                    other => {
                        return Err(WillametteError::UnsupportedTokenizer(format!(
                            "{}[{}] is not a string (got {:?})",
                            key, i, other
                        )));
                    }
                }
            }
            Ok(out)
        }
        other => Err(WillametteError::UnsupportedTokenizer(format!(
            "{} is not an array (got {:?})",
            key, other
        ))),
    }
}

fn u32_or_none(meta: &HashMap<String, GgufValue>, key: &str) -> Option<u32> {
    meta.get(key).and_then(|v| v.as_u64()).map(|v| v as u32)
}

fn bool_or_default(meta: &HashMap<String, GgufValue>, key: &str, default: bool) -> bool {
    match meta.get(key) {
        Some(GgufValue::Bool(b)) => *b,
        _ => default,
    }
}
