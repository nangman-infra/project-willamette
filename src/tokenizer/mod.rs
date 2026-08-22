//! GGUF-backed GPT-2 byte BPE and Llama SentencePiece BPE tokenizers.
//!
//! ## Stage 2 scope
//!
//! Supports `tokenizer.ggml.model = "gpt2"` and classic Llama
//! SentencePiece BPE. All vocabulary data and special-token IDs come from
//! GGUF metadata; no model vocabulary is compiled into the runtime.
//!
//! ## Roundtrip guarantee
//!
//! Complete GPT-2 byte vocabularies round-trip any UTF-8 input byte for byte.
//! Classic SentencePiece removes its configured dummy prefix in [`Tokenizer::decode`],
//! while raw piece decoding keeps generated leading spaces. SmolLM intentionally
//! omits a source-pinned set of byte tokens. The BitNet roundtrip contract is
//! verified against the real `ggml-model-i2_s.gguf` in `tests/tokenizer_roundtrip.rs`.
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
use pretokenize::Gpt2PreTokenizer;

const SMOLLM_ALLOWED_MISSING_BYTES: &[u8] = &[
    0x04, 0x06, 0x13, 0x14, 0x16, 0x1d, 0xc0, 0xc1, 0xf1, 0xf2, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa,
    0xfb, 0xfc, 0xfd, 0xfe, 0xff,
];

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

/// A piece of a prompt for [`Tokenizer::encode_with_specials`].
///
/// `Text` segments go through the selected tokenizer backend. `Special`
/// segments are inserted as a single token id verbatim — bypassing
/// BPE entirely. This matters for chat templates that need to inject
/// `<|end_of_text|>` (128001), `<|eot_id|>` (128009), or other
/// special markers in the middle of an otherwise textual prompt:
/// going through BPE would split `<|end_of_text|>` into 7 byte-level
/// tokens instead of the 1 id the model trained on.
#[derive(Debug, Clone, Copy)]
pub enum PromptPart<'a> {
    Text(&'a str),
    Special(u32),
}

pub struct Tokenizer {
    backend: TokenizerBackend,
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, u32>,
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub pad_id: Option<u32>,
    pub default_add_bos: bool,
    pub default_add_eos: bool,
    pub model_type: String,
}

enum TokenizerBackend {
    Gpt2 {
        byte_unicode: Box<ByteUnicode>,
        bpe: Box<Bpe>,
        pretokenizer: Gpt2PreTokenizer,
    },
    SentencePiece {
        scores: Vec<f32>,
        token_types: Vec<u32>,
        unknown_id: u32,
        add_space_prefix: bool,
    },
}

impl Tokenizer {
    /// Build a tokenizer purely from a GGUF metadata map.
    ///
    /// This is the **only** public constructor. There is no `Default`, no
    /// `new()`, no synthetic-vocab path. If the metadata does not describe a
    /// supported tokenizer, returns [`WillametteError::UnsupportedTokenizer`].
    pub fn from_gguf_metadata(meta: &HashMap<String, GgufValue>) -> Result<Self, WillametteError> {
        let model_type = required_str(meta, "tokenizer.ggml.model")?.to_string();
        if !matches!(model_type.as_str(), "gpt2" | "llama") {
            return Err(WillametteError::UnsupportedTokenizer(format!(
                "unsupported tokenizer.ggml.model {model_type:?}; supported: gpt2, llama"
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

        let backend = match model_type.as_str() {
            "gpt2" => load_gpt2_backend(meta, &token_to_id)?,
            "llama" => load_sentencepiece_backend(meta, &id_to_token, &token_to_id)?,
            other => {
                return Err(WillametteError::UnsupportedTokenizer(format!(
                    "unsupported tokenizer.ggml.model {other:?}; supported: gpt2, llama"
                )))
            }
        };

        let bos_id = checked_special_id(meta, "tokenizer.ggml.bos_token_id", id_to_token.len())?;
        let eos_id = checked_special_id(meta, "tokenizer.ggml.eos_token_id", id_to_token.len())?;
        let pad_id =
            checked_special_id(meta, "tokenizer.ggml.padding_token_id", id_to_token.len())?;

        let default_add_bos =
            bool_or_default(meta, "tokenizer.ggml.add_bos_token", model_type == "llama")?;
        let default_add_eos = bool_or_default(meta, "tokenizer.ggml.add_eos_token", false)?;

        Ok(Self {
            backend,
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

        self.encode_text_into(text, &mut ids)?;

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

    /// Encode a sequence of [`PromptPart`]s — text gets full byte-level
    /// BPE; specials are emitted as a single token id, verbatim.
    ///
    /// Does **not** auto-add BOS or EOS — caller is responsible (the
    /// whole point of this API is precise control over what ends up in
    /// the prompt). To prepend BOS, start the slice with
    /// `PromptPart::Special(tokenizer.bos_id.unwrap())`.
    ///
    /// Returns [`WillametteError::UnsupportedTokenizer`] if a special id
    /// is outside `0..vocab_size`.
    pub fn encode_with_specials(
        &self,
        parts: &[PromptPart<'_>],
    ) -> Result<Vec<u32>, WillametteError> {
        if matches!(&self.backend, TokenizerBackend::Gpt2 { .. }) {
            let mut ids = Vec::new();
            for part in parts {
                match *part {
                    PromptPart::Text(text) => self.encode_text_into(text, &mut ids)?,
                    PromptPart::Special(id) => self.push_checked_special(id, &mut ids)?,
                }
            }
            return Ok(ids);
        }

        let mut ids: Vec<u32> = Vec::new();
        let mut pending_text = String::new();
        for part in parts {
            match *part {
                PromptPart::Text(s) => pending_text.push_str(s),
                PromptPart::Special(id) => {
                    if !pending_text.is_empty() {
                        self.encode_text_into(&pending_text, &mut ids)?;
                        pending_text.clear();
                    }
                    self.push_checked_special(id, &mut ids)?;
                }
            }
        }
        if !pending_text.is_empty() {
            self.encode_text_into(&pending_text, &mut ids)?;
        }
        Ok(ids)
    }

    /// Encode one ChatML user turn followed by an assistant generation prompt.
    ///
    /// The marker ids are resolved from the vocabulary rather than hard-coded,
    /// and are inserted verbatim so byte-level BPE cannot split them.
    pub fn encode_chatml_user_turn(
        &self,
        prompt: &str,
    ) -> Result<(Vec<u32>, u32), WillametteError> {
        self.encode_chatml_turn(None, prompt)
    }

    /// Encode an optional ChatML system turn, one user turn, and the assistant
    /// generation prompt.
    pub fn encode_chatml_turn(
        &self,
        system: Option<&str>,
        prompt: &str,
    ) -> Result<(Vec<u32>, u32), WillametteError> {
        let (start_id, end_id) = self.chatml_marker_ids()?;
        let mut parts = Vec::with_capacity(if system.is_some() { 12 } else { 7 });
        if let Some(system) = system {
            parts.extend_from_slice(&[
                PromptPart::Special(start_id),
                PromptPart::Text("system\n"),
                PromptPart::Text(system),
                PromptPart::Special(end_id),
                PromptPart::Text("\n"),
            ]);
        }
        parts.extend_from_slice(&[
            PromptPart::Special(start_id),
            PromptPart::Text("user\n"),
            PromptPart::Text(prompt),
            PromptPart::Special(end_id),
            PromptPart::Text("\n"),
            PromptPart::Special(start_id),
            PromptPart::Text("assistant\n"),
        ]);
        let ids = self.encode_with_specials(&parts)?;
        Ok((ids, end_id))
    }

    /// Encode the incremental fragment for a ChatML turn after an assistant
    /// response whose opening marker and generated content are already cached.
    pub fn encode_chatml_follow_up(&self, prompt: &str) -> Result<Vec<u32>, WillametteError> {
        let (start_id, end_id) = self.chatml_marker_ids()?;
        self.encode_with_specials(&[
            PromptPart::Special(end_id),
            PromptPart::Text("\n"),
            PromptPart::Special(start_id),
            PromptPart::Text("user\n"),
            PromptPart::Text(prompt),
            PromptPart::Special(end_id),
            PromptPart::Text("\n"),
            PromptPart::Special(start_id),
            PromptPart::Text("assistant\n"),
        ])
    }

    pub(crate) fn chatml_marker_ids(&self) -> Result<(u32, u32), WillametteError> {
        let start_id = self.token_id("<|im_start|>").ok_or_else(|| {
            WillametteError::UnsupportedTokenizer(
                "ChatML requested but <|im_start|> is missing from the vocabulary".to_string(),
            )
        })?;
        let end_id = self.token_id("<|im_end|>").ok_or_else(|| {
            WillametteError::UnsupportedTokenizer(
                "ChatML requested but <|im_end|> is missing from the vocabulary".to_string(),
            )
        })?;
        Ok((start_id, end_id))
    }

    fn push_checked_special(&self, id: u32, ids: &mut Vec<u32>) -> Result<(), WillametteError> {
        if (id as usize) >= self.vocab_size() {
            return Err(WillametteError::UnsupportedTokenizer(format!(
                "encode_with_specials: special id {} out of vocab range (0..{})",
                id,
                self.vocab_size()
            )));
        }
        ids.push(id);
        Ok(())
    }

    fn encode_text_into(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), WillametteError> {
        match &self.backend {
            TokenizerBackend::Gpt2 {
                byte_unicode,
                bpe,
                pretokenizer,
            } => encode_gpt2(
                text,
                byte_unicode,
                bpe,
                *pretokenizer,
                &self.token_to_id,
                ids,
            ),
            TokenizerBackend::SentencePiece {
                scores,
                unknown_id,
                add_space_prefix,
                ..
            } => encode_sentencepiece(
                text,
                scores,
                *unknown_id,
                *add_space_prefix,
                &self.token_to_id,
                ids,
            ),
        }
    }

    /// Decode token IDs to a raw byte stream (no UTF-8 validation).
    ///
    /// Useful when generation may have stopped in the middle of a
    /// multi-byte UTF-8 character — the raw bytes are always
    /// recoverable. Callers wanting a `String` should use
    /// [`Tokenizer::decode`] (strict) or
    /// [`Tokenizer::decode_lossy`] (replaces invalid suffix with U+FFFD).
    pub fn decode_to_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, WillametteError> {
        if let TokenizerBackend::SentencePiece { token_types, .. } = &self.backend {
            return decode_sentencepiece(ids, &self.id_to_token, token_types);
        }
        let TokenizerBackend::Gpt2 { byte_unicode, .. } = &self.backend else {
            unreachable!()
        };
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
                let b = byte_unicode.decode_char(c).ok_or_else(|| {
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

    /// Decode a generated token span to text, preserving SentencePiece's
    /// piece-level leading-space marker and replacing any trailing incomplete
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

    /// Decode a complete token sequence back to UTF-8 text. Strict: fails if the
    /// concatenated bytes are not valid UTF-8 (e.g. generation stopped
    /// mid-multi-byte-character). Use [`Tokenizer::decode_lossy`] for
    /// generation streams that may be truncated.
    ///
    /// For SentencePiece this removes one synthetic leading space when
    /// `add_space_prefix` is enabled, making `decode(encode(text))` match `text`.
    /// Use [`Tokenizer::decode_to_bytes`] or [`Tokenizer::decode_lossy`] for a
    /// generated token span whose leading space must be preserved.
    ///
    /// GPT-2 special tokens decode to their literal display strings. SentencePiece
    /// control tokens such as BOS and EOS emit no bytes.
    pub fn decode(&self, ids: &[u32]) -> Result<String, WillametteError> {
        let mut bytes = self.decode_to_bytes(ids)?;
        if matches!(
            &self.backend,
            TokenizerBackend::SentencePiece {
                add_space_prefix: true,
                ..
            }
        ) && bytes.first() == Some(&b' ')
        {
            bytes.remove(0);
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

fn load_gpt2_backend(
    meta: &HashMap<String, GgufValue>,
    token_to_id: &HashMap<String, u32>,
) -> Result<TokenizerBackend, WillametteError> {
    let pretokenizer = match meta.get("tokenizer.ggml.pre") {
        None => Gpt2PreTokenizer::Default,
        Some(GgufValue::Str(value)) if value == "default" => Gpt2PreTokenizer::Default,
        Some(GgufValue::Str(value)) if value == "smollm" => Gpt2PreTokenizer::SmolLm,
        Some(GgufValue::Str(value)) => {
            return Err(WillametteError::UnsupportedTokenizer(format!(
                "unsupported tokenizer.ggml.pre {value:?} for gpt2; supported: default, smollm"
            )))
        }
        Some(_) => {
            return Err(WillametteError::UnsupportedTokenizer(
                "tokenizer.ggml.pre is not a string".to_string(),
            ))
        }
    };
    let merges_raw = load_string_array(meta, "tokenizer.ggml.merges")?;
    let mut merge_ranks = HashMap::with_capacity(merges_raw.len());
    for (rank, merge) in merges_raw.iter().enumerate() {
        let (left, right) = merge.split_once(' ').ok_or_else(|| {
            WillametteError::UnsupportedTokenizer(format!("merges[{rank}] malformed: {merge:?}"))
        })?;
        merge_ranks.insert((left.to_string(), right.to_string()), rank as u32);
    }
    let byte_unicode = ByteUnicode::new();
    let mut missing_bytes = Vec::new();
    for byte in 0u8..=255 {
        let token = byte_unicode.encode_byte(byte).to_string();
        if !token_to_id.contains_key(&token) {
            missing_bytes.push(byte);
        }
    }
    let forbidden_missing = missing_bytes
        .iter()
        .copied()
        .filter(|&byte| {
            !matches!(pretokenizer, Gpt2PreTokenizer::SmolLm) || !smollm_may_omit_byte(byte)
        })
        .collect::<Vec<_>>();
    if !forbidden_missing.is_empty() {
        return Err(WillametteError::UnsupportedTokenizer(format!(
            "vocab is missing unsupported base byte tokens {forbidden_missing:?}; all missing: {missing_bytes:?}"
        )));
    }
    Ok(TokenizerBackend::Gpt2 {
        byte_unicode: Box::new(byte_unicode),
        bpe: Box::new(Bpe::new(merge_ranks)),
        pretokenizer,
    })
}

fn smollm_may_omit_byte(byte: u8) -> bool {
    SMOLLM_ALLOWED_MISSING_BYTES.contains(&byte)
}

fn load_sentencepiece_backend(
    meta: &HashMap<String, GgufValue>,
    tokens: &[String],
    token_to_id: &HashMap<String, u32>,
) -> Result<TokenizerBackend, WillametteError> {
    let vocab_size = tokens.len();
    let scores = load_f32_array(meta, "tokenizer.ggml.scores")?;
    let token_types = load_u32_array(meta, "tokenizer.ggml.token_type")?;
    if scores.len() != vocab_size || token_types.len() != vocab_size {
        return Err(WillametteError::UnsupportedTokenizer(format!(
            "SentencePiece token/score/type lengths differ: {vocab_size}/{}/{}",
            scores.len(),
            token_types.len()
        )));
    }
    if let Some(index) = scores.iter().position(|score| !score.is_finite()) {
        return Err(WillametteError::UnsupportedTokenizer(format!(
            "SentencePiece token score {index} is not finite"
        )));
    }
    let unknown_id =
        checked_special_id(meta, "tokenizer.ggml.unknown_token_id", vocab_size)?.unwrap_or(0);
    for (id, (&token_type, token)) in token_types.iter().zip(tokens).enumerate() {
        if !(1..=6).contains(&token_type) {
            return Err(WillametteError::UnsupportedTokenizer(format!(
                "SentencePiece token {id} has unsupported type {token_type}"
            )));
        }
        if token_type == 6 && parse_byte_piece(token).is_none() {
            return Err(WillametteError::UnsupportedTokenizer(format!(
                "malformed SentencePiece byte token {token:?} at id {id}"
            )));
        }
    }
    for byte in 0u8..=255 {
        let piece = format!("<0x{byte:02X}>");
        let Some(&id) = token_to_id.get(&piece) else {
            return Err(WillametteError::UnsupportedTokenizer(format!(
                "SentencePiece vocabulary is missing byte fallback {piece}"
            )));
        };
        if token_types[id as usize] != 6 {
            return Err(WillametteError::UnsupportedTokenizer(format!(
                "SentencePiece byte fallback {piece} has type {}, expected 6",
                token_types[id as usize]
            )));
        }
    }
    let add_space_prefix = optional_bool_or_default(meta, "tokenizer.ggml.add_space_prefix", true)?;
    Ok(TokenizerBackend::SentencePiece {
        scores,
        token_types,
        unknown_id,
        add_space_prefix,
    })
}

fn encode_gpt2(
    text: &str,
    byte_unicode: &ByteUnicode,
    bpe: &Bpe,
    pretokenizer: Gpt2PreTokenizer,
    token_to_id: &HashMap<String, u32>,
    ids: &mut Vec<u32>,
) -> Result<(), WillametteError> {
    for chunk in pretokenize::pretokenize(text, pretokenizer) {
        let symbols = chunk
            .as_bytes()
            .iter()
            .map(|&byte| byte_unicode.encode_byte(byte).to_string())
            .collect();
        for token in bpe.encode_word(symbols) {
            if let Some(&id) = token_to_id.get(&token) {
                ids.push(id);
                continue;
            }
            let tolerated_smollm_omission = matches!(pretokenizer, Gpt2PreTokenizer::SmolLm)
                && token
                    .chars()
                    .next()
                    .filter(|_| token.chars().count() == 1)
                    .and_then(|symbol| byte_unicode.decode_char(symbol))
                    .is_some_and(smollm_may_omit_byte);
            if !tolerated_smollm_omission {
                return Err(WillametteError::UnsupportedTokenizer(format!(
                    "BPE produced symbol {token:?} not in vocabulary"
                )));
            }
        }
    }
    Ok(())
}

fn encode_sentencepiece(
    text: &str,
    scores: &[f32],
    unknown_id: u32,
    add_space_prefix: bool,
    token_to_id: &HashMap<String, u32>,
    ids: &mut Vec<u32>,
) -> Result<(), WillametteError> {
    if text.is_empty() {
        return Ok(());
    }
    let normalized = if add_space_prefix {
        format!(" {text}")
    } else {
        text.to_string()
    }
    .replace(' ', "▁");
    let mut symbols = normalized
        .chars()
        .map(|c| Some(c.to_string()))
        .collect::<Vec<_>>();

    while let Some((left, right)) = best_sentencepiece_merge(&symbols, scores, token_to_id) {
        let right_symbol = symbols[right].take().unwrap();
        symbols[left].as_mut().unwrap().push_str(&right_symbol);
    }

    for symbol in symbols.into_iter().flatten() {
        emit_sentencepiece_symbol(&symbol, unknown_id, token_to_id, ids);
    }
    Ok(())
}

fn best_sentencepiece_merge(
    symbols: &[Option<String>],
    scores: &[f32],
    token_to_id: &HashMap<String, u32>,
) -> Option<(usize, usize)> {
    let live = symbols
        .iter()
        .enumerate()
        .filter_map(|(index, symbol)| symbol.as_ref().map(|symbol| (index, symbol)))
        .collect::<Vec<_>>();
    let mut best: Option<(usize, usize, f32)> = None;
    for pair in live.windows(2) {
        let merged = format!("{}{}", pair[0].1, pair[1].1);
        let Some(&token) = token_to_id.get(&merged) else {
            continue;
        };
        let candidate = (pair[0].0, pair[1].0, scores[token as usize]);
        if best.is_none_or(|current| {
            candidate.2 > current.2 || (candidate.2 == current.2 && candidate.0 < current.0)
        }) {
            best = Some(candidate);
        }
    }
    best.map(|(left, right, _)| (left, right))
}

fn emit_sentencepiece_symbol(
    symbol: &str,
    unknown_id: u32,
    token_to_id: &HashMap<String, u32>,
    ids: &mut Vec<u32>,
) {
    if let Some(&token) = token_to_id.get(symbol) {
        ids.push(token);
        return;
    }
    for byte in symbol.as_bytes() {
        let piece = format!("<0x{byte:02X}>");
        ids.push(token_to_id.get(&piece).copied().unwrap_or(unknown_id));
    }
}

fn decode_sentencepiece(
    ids: &[u32],
    tokens: &[String],
    token_types: &[u32],
) -> Result<Vec<u8>, WillametteError> {
    let mut output = Vec::new();
    for &id in ids {
        let token = tokens.get(id as usize).ok_or_else(|| {
            WillametteError::UnsupportedTokenizer(format!(
                "token id {id} out of vocab range (size = {})",
                tokens.len()
            ))
        })?;
        match token_types[id as usize] {
            3 => {}
            6 => {
                output.push(parse_byte_piece(token).ok_or_else(|| {
                    WillametteError::UnsupportedTokenizer(format!(
                        "malformed SentencePiece byte token {token:?}"
                    ))
                })?);
            }
            _ => output.extend_from_slice(token.replace('▁', " ").as_bytes()),
        }
    }
    Ok(output)
}

fn parse_byte_piece(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    (hex.len() == 2)
        .then(|| u8::from_str_radix(hex, 16).ok())
        .flatten()
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

fn load_f32_array(
    meta: &HashMap<String, GgufValue>,
    key: &str,
) -> Result<Vec<f32>, WillametteError> {
    let Some(GgufValue::Array(values)) = meta.get(key) else {
        return Err(WillametteError::UnsupportedTokenizer(format!(
            "missing or non-array metadata key: {key}"
        )));
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value.as_f32().ok_or_else(|| {
                WillametteError::UnsupportedTokenizer(format!("{key}[{index}] is not a float"))
            })
        })
        .collect()
}

fn load_u32_array(
    meta: &HashMap<String, GgufValue>,
    key: &str,
) -> Result<Vec<u32>, WillametteError> {
    let Some(GgufValue::Array(values)) = meta.get(key) else {
        return Err(WillametteError::UnsupportedTokenizer(format!(
            "missing or non-array metadata key: {key}"
        )));
    };
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    WillametteError::UnsupportedTokenizer(format!("{key}[{index}] is not a u32"))
                })
        })
        .collect()
}

fn checked_special_id(
    meta: &HashMap<String, GgufValue>,
    key: &str,
    vocab_size: usize,
) -> Result<Option<u32>, WillametteError> {
    let Some(value) = meta.get(key) else {
        return Ok(None);
    };
    let raw = value.as_u64().ok_or_else(|| {
        WillametteError::UnsupportedTokenizer(format!("{key} is not an unsigned integer"))
    })?;
    if raw == u32::MAX as u64 {
        return Ok(None);
    }
    let id = u32::try_from(raw).map_err(|_| {
        WillametteError::UnsupportedTokenizer(format!("{key} value {raw} does not fit in u32"))
    })?;
    if id as usize >= vocab_size {
        return Err(WillametteError::UnsupportedTokenizer(format!(
            "{key} id {id} is outside vocabulary size {vocab_size}"
        )));
    }
    Ok(Some(id))
}

fn bool_or_default(
    meta: &HashMap<String, GgufValue>,
    key: &str,
    default: bool,
) -> Result<bool, WillametteError> {
    match meta.get(key) {
        Some(GgufValue::Bool(value)) => Ok(*value),
        Some(_) => Err(WillametteError::UnsupportedTokenizer(format!(
            "{key} is not a boolean"
        ))),
        None => Ok(default),
    }
}

fn optional_bool_or_default(
    meta: &HashMap<String, GgufValue>,
    key: &str,
    default: bool,
) -> Result<bool, WillametteError> {
    match meta.get(key) {
        Some(GgufValue::Bool(value)) => Ok(*value),
        Some(_) => Err(WillametteError::UnsupportedTokenizer(format!(
            "{key} is not a boolean"
        ))),
        None => Ok(default),
    }
}
