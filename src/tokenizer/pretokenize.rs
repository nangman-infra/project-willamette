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
static SMOLLM_RE_MAIN: OnceLock<Regex> = OnceLock::new();
static GPT2_RE_DIGITS: OnceLock<Regex> = OnceLock::new();
static SMOLLM_RE_DIGIT: OnceLock<Regex> = OnceLock::new();
static QWEN2_RE_LETTER: OnceLock<Regex> = OnceLock::new();
static QWEN2_RE_NUMBER: OnceLock<Regex> = OnceLock::new();
static QWEN2_RE_WHITESPACE: OnceLock<Regex> = OnceLock::new();

#[derive(Debug, Clone, Copy)]
pub(super) enum Gpt2PreTokenizer {
    Default,
    SmolLm,
    Qwen2,
}

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

fn re_smollm_main() -> &'static Regex {
    SMOLLM_RE_MAIN.get_or_init(|| {
        // Omitting `\s+(?!\S)` is intentional. Trailing whitespace is emitted
        // as the unmatched gap, while internal runs leave their final space
        // available to the optional-space word branch.
        Regex::new(r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+"#)
            .expect("SmolLM main regex must compile")
    })
}

fn re_digits() -> &'static Regex {
    GPT2_RE_DIGITS.get_or_init(|| Regex::new(r#"\p{N}+"#).expect("DEFAULT regex 3 must compile"))
}

fn re_individual_digit() -> &'static Regex {
    SMOLLM_RE_DIGIT.get_or_init(|| Regex::new(r#"\p{N}"#).expect("SmolLM digit regex must compile"))
}

fn re_qwen2_letter() -> &'static Regex {
    QWEN2_RE_LETTER
        .get_or_init(|| Regex::new(r#"^\p{L}$"#).expect("Qwen2 letter regex must compile"))
}

fn re_qwen2_number() -> &'static Regex {
    QWEN2_RE_NUMBER
        .get_or_init(|| Regex::new(r#"^\p{N}$"#).expect("Qwen2 number regex must compile"))
}

fn re_qwen2_whitespace() -> &'static Regex {
    QWEN2_RE_WHITESPACE
        .get_or_init(|| Regex::new(r#"^\s$"#).expect("Qwen2 whitespace regex must compile"))
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

/// SmolLM applies `\p{N}` before the GPT-2 main expression, making every
/// decimal digit its own pre-token. Source: `LLAMA_VOCAB_PRE_TYPE_SMOLLM` in
/// pinned llama.cpp `704485942ab54bbbbf1f241b3550ffba35f5f37e`.
fn smollm_pretokenize(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut digits_extracted = Vec::new();
    let mut last = 0;
    for matched in re_individual_digit().find_iter(text) {
        if matched.start() > last {
            digits_extracted.push((false, &text[last..matched.start()]));
        }
        digits_extracted.push((true, matched.as_str()));
        last = matched.end();
    }
    if last < text.len() {
        digits_extracted.push((false, &text[last..]));
    }

    let mut output = Vec::new();
    for (matched, chunk) in digits_extracted {
        if matched {
            output.push(chunk);
        } else {
            output.extend(smollm_split_main(chunk));
        }
    }
    output
}

fn smollm_split_main(chunk: &str) -> Vec<&str> {
    let mut output = Vec::new();
    let mut last = 0;
    for matched in re_smollm_main().find_iter(chunk) {
        if matched.start() > last {
            split_internal_whitespace(
                &chunk[last..matched.start()],
                matched.as_str().starts_with(' '),
                &mut output,
            );
        }
        output.push(matched.as_str());
        last = matched.end();
    }
    if last < chunk.len() {
        // This corresponds to llama.cpp's trailing `\s+(?!\S)` branch.
        output.push(&chunk[last..]);
    }
    output
}

fn split_internal_whitespace<'a>(
    gap: &'a str,
    next_match_has_space: bool,
    output: &mut Vec<&'a str>,
) {
    if !gap.chars().all(char::is_whitespace) {
        output.push(gap);
        return;
    }
    if next_match_has_space {
        output.push(gap);
        return;
    }
    let mut index = 0;
    while index < gap.len() {
        if gap.as_bytes()[index] == b' ' {
            let start = index;
            let current = gap[index..].chars().next().unwrap();
            index += current.len_utf8();
            while index < gap.len() && gap[index..].starts_with(current) {
                index += current.len_utf8();
            }
            output.push(&gap[start..index]);
        } else {
            let width = gap[index..].chars().next().unwrap().len_utf8();
            output.push(&gap[index..index + width]);
            index += width;
        }
    }
}

#[derive(Clone, Copy)]
struct Qwen2Char {
    start: usize,
    end: usize,
    value: char,
    letter: bool,
    number: bool,
    whitespace: bool,
}

/// Qwen2's ordered regex, implemented directly to preserve the
/// `\s+(?!\S)` behavior unsupported by Rust's regex crate. This follows
/// llama.cpp's `unicode_regex_split_custom_qwen2` branch order.
fn qwen2_pretokenize(text: &str) -> Vec<&str> {
    let chars = text
        .char_indices()
        .map(|(start, value)| {
            let end = start + value.len_utf8();
            let slice = &text[start..end];
            Qwen2Char {
                start,
                end,
                value,
                letter: re_qwen2_letter().is_match(slice),
                number: re_qwen2_number().is_match(slice),
                whitespace: re_qwen2_whitespace().is_match(slice),
            }
        })
        .collect::<Vec<_>>();
    let mut output = Vec::new();
    let mut pos = 0;

    while pos < chars.len() {
        let start = chars[pos].start;

        if chars[pos].value == '\'' {
            let contraction_len = qwen2_contraction_len(&chars[pos..]);
            if contraction_len > 0 {
                pos += contraction_len;
                output.push(&text[start..chars[pos - 1].end]);
                continue;
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if chars[pos].value != '\r'
            && chars[pos].value != '\n'
            && !chars[pos].number
            && (chars[pos].letter || chars.get(pos + 1).is_some_and(|c| c.letter))
        {
            pos += 1;
            while chars.get(pos).is_some_and(|c| c.letter) {
                pos += 1;
            }
            output.push(&text[start..chars[pos - 1].end]);
            continue;
        }

        // \p{N}
        if chars[pos].number {
            output.push(&text[start..chars[pos].end]);
            pos += 1;
            continue;
        }

        //  ?[^\s\p{L}\p{N}]+[\r\n]*
        let symbol_pos = pos + usize::from(chars[pos].value == ' ');
        if chars
            .get(symbol_pos)
            .is_some_and(|c| !c.whitespace && !c.letter && !c.number)
        {
            pos = symbol_pos + 1;
            while chars
                .get(pos)
                .is_some_and(|c| !c.whitespace && !c.letter && !c.number)
            {
                pos += 1;
            }
            while chars
                .get(pos)
                .is_some_and(|c| matches!(c.value, '\r' | '\n'))
            {
                pos += 1;
            }
            output.push(&text[start..chars[pos - 1].end]);
            continue;
        }

        let whitespace_start = pos;
        let mut last_newline_end = None;
        while chars.get(pos).is_some_and(|c| c.whitespace) {
            if matches!(chars[pos].value, '\r' | '\n') {
                last_newline_end = Some(pos + 1);
            }
            pos += 1;
        }
        if let Some(end) = last_newline_end {
            pos = end;
        } else if pos - whitespace_start > 1 && pos < chars.len() {
            pos -= 1;
        } else if pos == whitespace_start {
            pos += 1;
        }
        output.push(&text[start..chars[pos - 1].end]);
    }

    output
}

fn qwen2_contraction_len(chars: &[Qwen2Char]) -> usize {
    let lower = |index: usize| chars.get(index).map(|c| c.value.to_ascii_lowercase());
    match (lower(1), lower(2)) {
        (Some('s' | 't' | 'm' | 'd'), _) => 2,
        (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => 3,
        _ => 0,
    }
}

pub(super) fn pretokenize(text: &str, kind: Gpt2PreTokenizer) -> Vec<&str> {
    match kind {
        Gpt2PreTokenizer::Default => gpt2_pretokenize(text),
        Gpt2PreTokenizer::SmolLm => smollm_pretokenize(text),
        Gpt2PreTokenizer::Qwen2 => qwen2_pretokenize(text),
    }
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

    #[test]
    fn smollm_splits_digits_individually() {
        let parts = smollm_pretokenize("84 cats");
        assert_eq!(parts, ["8", "4", " cats"]);
        assert_eq!(parts.concat(), "84 cats");
    }

    #[test]
    fn smollm_preserves_internal_space_for_the_following_word() {
        assert_eq!(smollm_pretokenize("a  b"), ["a", " ", " b"]);
        assert_eq!(smollm_pretokenize("a   b"), ["a", "  ", " b"]);
        assert_eq!(smollm_pretokenize("a  "), ["a", "  "]);
        assert_eq!(smollm_pretokenize("a\t\tb"), ["a", "\t", "\t", "b"]);
        assert_eq!(smollm_pretokenize("a\n\nb"), ["a", "\n", "\n", "b"]);
        assert_eq!(smollm_pretokenize("a \tb"), ["a", " ", "\t", "b"]);
        assert_eq!(smollm_pretokenize("a \t\t b"), ["a", " \t\t", " b"]);
        assert_eq!(smollm_pretokenize("a\n\t\n b"), ["a", "\n\t\n", " b"]);
    }

    #[test]
    fn qwen2_matches_llama_cpp_boundaries() {
        assert_eq!(
            qwen2_pretokenize("Hello123 WORLD'S!\n  next"),
            ["Hello", "1", "2", "3", " WORLD", "'S", "!\n", " ", " next"]
        );
        assert_eq!(qwen2_pretokenize("a  b"), ["a", " ", " b"]);
        assert_eq!(qwen2_pretokenize("a \t\n  b"), ["a", " \t\n", " ", " b"]);
    }

    #[test]
    fn qwen2_is_lossless_for_mixed_unicode() {
        for text in [
            "",
            "Qwen2.5: hello, world! 12345",
            "안녕하세요 世界 🎉",
            "don't I'M we'll",
            "line one\r\n\tline two  ",
            "e\u{301} १२٣",
        ] {
            assert_eq!(qwen2_pretokenize(text).concat(), text, "input: {text:?}");
        }
    }
}
