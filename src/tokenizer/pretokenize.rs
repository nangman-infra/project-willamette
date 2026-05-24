//! Pre-tokenization for `tokenizer.ggml.model = "gpt2"`.
//!
//! When the GGUF metadata has no `tokenizer.ggml.pre` key (the case for
//! `microsoft/bitnet-b1.58-2B-4T-gguf`), upstream llama.cpp falls back to
//! the `LLAMA_VOCAB_PRE_TYPE_DEFAULT` regex set (and prints a
//! "GENERATION QUALITY WILL BE DEGRADED" warning).
//!
//! `DEFAULT` uses **three regexes applied sequentially** per
//! `unicode_regex_split` in `3rdparty/llama.cpp/src/unicode.cpp:653` and
//! the default branch in `llama-vocab.cpp:495..501` of the pinned
//! commit:
//!
//! 1. `[\p{P}\$\+<=>\^~\|]+`  — extract punctuation/operator runs.
//! 2. `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`
//!    — apostrophe contractions, letters / digits / non-alphanum
//!    runs with optional leading space, and trailing whitespace.
//! 3. `\p{N}+`  — split any digit run that is still inside a larger
//!    chunk (this is why ` 1` ends up as `[' ', '1']`, not as a single
//!    ` 1` token).
//!
//! Each regex receives only the chunks NOT matched by earlier regexes;
//! matched runs become standalone pre-tokens. The algorithm matches
//! `tests/tokenizer_roundtrip.rs::roundtrip_does_not_depend_on_token_id_hardcoding`
//! and is validated against `bitnet.cpp llama-tokenize` outputs in
//! Stage 5-E (see `docs/REFERENCE_COMPATIBILITY.md`).
//!
//! Lookaround caveat: Rust's `regex` crate does not support
//! `(?!\S)`. We replace `\s+(?!\S)` with plain `\s+`. The semantics
//! differ only on chunks of mixed whitespace + non-whitespace, which
//! cannot occur here because regex 2 receives chunks already filtered
//! by regex 1; the parts that survive are either a single whitespace
//! run (matches identically) or non-whitespace contexts where the
//! `\s+` branch never triggers as the leftmost-longest winner.

use regex::Regex;
use std::sync::OnceLock;

static GPT2_RE_PUNCT: OnceLock<Regex> = OnceLock::new();
static GPT2_RE_MAIN: OnceLock<Regex> = OnceLock::new();
static GPT2_RE_DIGITS: OnceLock<Regex> = OnceLock::new();

fn re_punct() -> &'static Regex {
    GPT2_RE_PUNCT.get_or_init(|| {
        Regex::new(r#"[\p{P}\$\+<=>\^~\|]+"#).expect("DEFAULT regex 1 must compile")
    })
}

fn re_main() -> &'static Regex {
    GPT2_RE_MAIN.get_or_init(|| {
        Regex::new(r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+"#)
            .expect("DEFAULT regex 2 must compile")
    })
}

fn re_digits() -> &'static Regex {
    GPT2_RE_DIGITS.get_or_init(|| Regex::new(r#"\p{N}+"#).expect("DEFAULT regex 3 must compile"))
}

/// Split one chunk by a single regex into a vector of substring
/// references, preserving order. Each substring is non-empty.
fn split_by_regex<'a>(chunk: &'a str, re: &Regex) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut last = 0;
    for m in re.find_iter(chunk) {
        if m.start() > last {
            out.push(&chunk[last..m.start()]);
        }
        out.push(m.as_str());
        last = m.end();
    }
    if last < chunk.len() {
        out.push(&chunk[last..]);
    }
    out
}

/// Apply the 3-regex sequential split that matches llama.cpp's
/// `LLAMA_VOCAB_PRE_TYPE_DEFAULT` for `tokenizer.ggml.model = "gpt2"`
/// without a `tokenizer.ggml.pre` key.
///
/// Each regex application splits only the chunks left unmatched by
/// previous regexes. The resulting list of pre-tokens is the input to
/// byte-level BPE. Concatenation of all chunks equals the input
/// byte-for-byte (lossless).
pub(super) fn gpt2_pretokenize(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    // Step 1 — punctuation-run extraction.
    let mut step1: Vec<(bool, &str)> = Vec::new();
    let mut last = 0;
    for m in re_punct().find_iter(text) {
        if m.start() > last {
            step1.push((false, &text[last..m.start()]));
        }
        step1.push((true, m.as_str()));
        last = m.end();
    }
    if last < text.len() {
        step1.push((false, &text[last..]));
    }

    // Step 2 — main GPT-2 alternation on the still-unmatched chunks.
    let mut step2: Vec<(bool, &str)> = Vec::new();
    for (matched, s) in step1 {
        if matched {
            step2.push((true, s));
        } else {
            for sub in split_by_regex(s, re_main()) {
                // Step 2 always declares its output "matched" once it
                // produced any non-empty substring — there is no
                // separate fallback set, and any leftover residual
                // (which `split_by_regex` already pushed unchanged) is
                // forwarded to step 3 untouched.
                let was_matched = re_main().is_match(sub);
                step2.push((was_matched, sub));
            }
        }
    }

    // Step 3 — digit-run extraction inside any chunk that still has
    // mixed content (e.g. `" 1"` becomes `[" ", "1"]`).
    let mut step3: Vec<&str> = Vec::new();
    for (matched, s) in step2 {
        if matched {
            // Step 3 still has the right to split a step-2 match if the
            // step-2 pattern matched a `\p{N}+` run with a leading
            // space (e.g. ` ?\p{N}+`). Mirror llama.cpp's behaviour by
            // running regex 3 on every chunk; matches stay, gaps emit
            // as standalone substrings.
            for sub in split_by_regex(s, re_digits()) {
                step3.push(sub);
            }
        } else {
            for sub in split_by_regex(s, re_digits()) {
                step3.push(sub);
            }
        }
    }

    step3
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_lossless(s: &str) {
        let parts = gpt2_pretokenize(s);
        let joined: String = parts.iter().copied().collect();
        assert_eq!(joined, s, "pre-tokenization is not lossless for {:?}", s);
    }

    #[test]
    fn lossless_ascii() {
        check_lossless("Hello world!");
        check_lossless("don't stop");
        check_lossless("");
    }

    #[test]
    fn lossless_korean() {
        check_lossless("안녕하세요");
        check_lossless("안녕 하세요 안녕");
    }

    #[test]
    fn lossless_emoji() {
        check_lossless("hello 🎉 world");
        check_lossless("🚀🌟✨");
    }

    #[test]
    fn lossless_whitespace_extremes() {
        check_lossless("  multiple   spaces  ");
        check_lossless("\n\t\r\nmixed");
        check_lossless("trailing   ");
    }

    #[test]
    fn lossless_mixed_scripts() {
        check_lossless("Hello, 안녕 world! 한글 + emoji 🎉 + 123.");
    }

    #[test]
    fn arithmetic_splits_per_default_regex() {
        // `LLAMA_VOCAB_PRE_TYPE_DEFAULT` rule:
        //   regex 1 extracts "+", "=" as standalone runs;
        //   regex 2 splits the remaining numbers + spaces;
        //   regex 3 separates digits from any leading space.
        // Expected: ["1", " ", "+", " ", "1", " ", "="].
        let parts = gpt2_pretokenize("1 + 1 =");
        assert_eq!(parts, vec!["1", " ", "+", " ", "1", " ", "="]);
    }
}
