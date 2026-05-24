#![allow(clippy::needless_range_loop)]
// The byte → unicode table builds two parallel 256-element arrays
// indexed by byte value; explicit `for b in 0..256` matches the
// GPT-2 algorithm in `bytes_to_unicode()` 1:1 and is the form audited
// against upstream.

//! GPT-2 byte → Unicode reversible mapping.
//!
//! GPT-2 / LLaMA 3 byte-level BPE encodes a raw byte by mapping it to one of
//! 256 printable Unicode code points. Bytes that are already in a "safe"
//! printable range map to themselves; the rest are remapped to the U+0100…
//! block. The inverse mapping is bijective and lets us decode a sequence of
//! BPE token strings back to the exact original byte stream — which is what
//! makes the algorithm UTF-8 lossless even for tokens that span code-point
//! boundaries (e.g. mid-Korean syllable).
//!
//! Source of the spec: OpenAI GPT-2 `bytes_to_unicode()` in
//! <https://github.com/openai/gpt-2/blob/master/src/encoder.py>.

use std::collections::HashMap;

pub(super) struct ByteUnicode {
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
}

impl ByteUnicode {
    pub(super) fn new() -> Self {
        let mut printable = [false; 256];
        for b in (b'!' as usize)..=(b'~' as usize) {
            printable[b] = true;
        }
        for b in 0xA1usize..=0xACusize {
            printable[b] = true;
        }
        for b in 0xAEusize..=0xFFusize {
            printable[b] = true;
        }

        let mut byte_to_char = ['\0'; 256];

        for b in 0usize..256 {
            if printable[b] {
                byte_to_char[b] = char::from_u32(b as u32).expect("valid printable code point");
            }
        }

        let mut next_code: u32 = 256;
        for b in 0usize..256 {
            if !printable[b] {
                byte_to_char[b] = char::from_u32(next_code).expect("U+0100..U+0143 are valid");
                next_code += 1;
            }
        }

        let mut char_to_byte: HashMap<char, u8> = HashMap::with_capacity(256);
        for b in 0usize..256 {
            char_to_byte.insert(byte_to_char[b], b as u8);
        }

        Self {
            byte_to_char,
            char_to_byte,
        }
    }

    pub(super) fn encode_byte(&self, b: u8) -> char {
        self.byte_to_char[b as usize]
    }

    pub(super) fn decode_char(&self, c: char) -> Option<u8> {
        self.char_to_byte.get(&c).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_byte_roundtrips() {
        let bu = ByteUnicode::new();
        for b in 0u8..=255 {
            let c = bu.encode_byte(b);
            assert_eq!(
                bu.decode_char(c),
                Some(b),
                "byte 0x{:02X} -> '{}' did not roundtrip",
                b,
                c
            );
        }
    }

    #[test]
    fn all_codepoints_unique() {
        let bu = ByteUnicode::new();
        let mut seen = std::collections::HashSet::new();
        for b in 0u8..=255 {
            assert!(
                seen.insert(bu.encode_byte(b)),
                "duplicate code point for byte 0x{:02X}",
                b
            );
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn known_gpt2_mappings() {
        let bu = ByteUnicode::new();
        // Space (0x20) is the 33rd non-printable byte (after 0x00..0x1F), so
        // it should map to U+0100 + 32 = U+0120 ("Ġ").
        assert_eq!(bu.encode_byte(0x20), 'Ġ');
        // Newline (0x0A) is the 11th non-printable → U+010A ("Ċ").
        assert_eq!(bu.encode_byte(0x0A), 'Ċ');
        // Tab (0x09) is the 10th non-printable → U+0109 ("ĉ").
        assert_eq!(bu.encode_byte(0x09), 'ĉ');
        // Printable ASCII maps to itself.
        assert_eq!(bu.encode_byte(b'A'), 'A');
        assert_eq!(bu.encode_byte(b'!'), '!');
        assert_eq!(bu.encode_byte(b'~'), '~');
    }
}
