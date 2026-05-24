//! GPT-2 byte-pair encoding (BPE) merge algorithm.
//!
//! Given a word represented as a sequence of single-character symbols (in our
//! pipeline, each symbol is one [`super::byte_unicode::ByteUnicode`]-mapped
//! character of one source byte), repeatedly find the pair with the lowest
//! merge rank and combine it, until no further merges apply.
//!
//! Complexity is O(n²) in the number of symbols per word. For pre-tokenized
//! chunks this is small and acceptable; large-input throughput optimization
//! belongs in a later pass, not Stage 2.

use std::collections::HashMap;

pub(super) struct Bpe {
    merge_ranks: HashMap<(String, String), u32>,
}

impl Bpe {
    pub(super) fn new(merge_ranks: HashMap<(String, String), u32>) -> Self {
        Self { merge_ranks }
    }

    pub(super) fn encode_word(&self, mut symbols: Vec<String>) -> Vec<String> {
        if symbols.len() < 2 {
            return symbols;
        }

        loop {
            let mut best_rank = u32::MAX;
            let mut best_idx: Option<usize> = None;

            for i in 0..symbols.len() - 1 {
                let key = (symbols[i].clone(), symbols[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&key) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = Some(i);
                    }
                }
            }

            let Some(idx) = best_idx else {
                break;
            };

            let merged = {
                let mut s = String::with_capacity(symbols[idx].len() + symbols[idx + 1].len());
                s.push_str(&symbols[idx]);
                s.push_str(&symbols[idx + 1]);
                s
            };
            symbols[idx] = merged;
            symbols.remove(idx + 1);

            if symbols.len() < 2 {
                break;
            }
        }

        symbols
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(merges: &[(&str, &str)]) -> Bpe {
        let map = merges
            .iter()
            .enumerate()
            .map(|(rank, (a, b))| ((a.to_string(), b.to_string()), rank as u32))
            .collect();
        Bpe::new(map)
    }

    fn syms(strs: &[&str]) -> Vec<String> {
        strs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn empty_returns_empty() {
        let bpe = build(&[]);
        assert!(bpe.encode_word(vec![]).is_empty());
    }

    #[test]
    fn single_symbol_passes_through() {
        let bpe = build(&[]);
        assert_eq!(bpe.encode_word(syms(&["x"])), vec!["x".to_string()]);
    }

    #[test]
    fn no_merges_keeps_symbols() {
        let bpe = build(&[]);
        assert_eq!(
            bpe.encode_word(syms(&["a", "b", "c"])),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn single_merge_applies() {
        let bpe = build(&[("a", "b")]);
        assert_eq!(
            bpe.encode_word(syms(&["a", "b", "c"])),
            vec!["ab".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn lowest_rank_pair_wins() {
        // ("b", "c") has rank 0 (highest priority), so it merges first;
        // afterwards no further merge applies.
        let bpe = build(&[("b", "c"), ("a", "b")]);
        assert_eq!(
            bpe.encode_word(syms(&["a", "b", "c"])),
            vec!["a".to_string(), "bc".to_string()]
        );
    }

    #[test]
    fn cascading_merges() {
        let bpe = build(&[("a", "b"), ("ab", "c")]);
        assert_eq!(
            bpe.encode_word(syms(&["a", "b", "c"])),
            vec!["abc".to_string()]
        );
    }

    #[test]
    fn merge_inside_run() {
        let bpe = build(&[("a", "a")]);
        // "aaaa": pair (a,a) rank 0. Merges leftmost first.
        // Step 1: ["aa", "a", "a"]. (a,a) appears at idx 1.
        // Step 2: ["aa", "aa"]. (aa, aa) has no merge rank, stop.
        assert_eq!(
            bpe.encode_word(syms(&["a", "a", "a", "a"])),
            vec!["aa".to_string(), "aa".to_string()]
        );
    }
}
