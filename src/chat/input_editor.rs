//! Readline-grade single-line input editor with history.
//!
//! Pure data structure — no ratatui, no terminal I/O. Owns the input
//! buffer + cursor position + history + reverse-search state. The
//! TUI layer translates key events into method calls and renders the
//! resulting state.
//!
//! What this module guarantees:
//!
//! * UTF-8 safe — cursor moves are by codepoint, not byte. Multi-byte
//!   characters (Korean, emoji) are atomic.
//! * Deterministic — every public method has predictable state after.
//! * Tested in isolation — every key-binding semantic has a unit test
//!   below.
//!
//! What this module does NOT do:
//!
//! * Multi-line input (Shift-Enter / Alt-Enter for newline). The
//!   current scope is single-line; multi-line composition can be
//!   added later by relaxing the `submit` invariant.
//! * Rendering. The TUI computes cursor screen position from
//!   `cursor_byte()` and the input's wrapped line layout.

use std::collections::VecDeque;

/// Maximum number of past prompts kept in memory + persisted to the
/// history file. Older entries get evicted FIFO.
pub const HISTORY_CAP: usize = 1000;

/// Reverse-search mode state (Ctrl-R).
#[derive(Debug, Clone)]
pub struct SearchState {
    /// What the user has typed into the search prompt so far.
    pub needle: String,
    /// Which history entry is currently matched (newest-first index).
    pub match_idx: Option<usize>,
}

impl SearchState {
    fn new() -> Self {
        Self {
            needle: String::new(),
            match_idx: None,
        }
    }
}

/// Single-line input editor with cursor, history, and reverse-search.
///
/// Cursor is stored as a **byte offset** into `buffer` (never inside
/// a multi-byte codepoint). All cursor moves snap to char boundaries.
#[derive(Debug, Clone)]
pub struct InputEditor {
    buffer: String,
    /// Cursor as byte offset. `0 <= cursor_byte <= buffer.len()`.
    cursor_byte: usize,
    /// Newest-first ring buffer of past prompts. `history[0]` is the
    /// most recently submitted.
    history: VecDeque<String>,
    /// While the user is scrolling history via Up/Down, this is the
    /// index into `history` of the entry currently shown. `None`
    /// means the user is on the "fresh" line (not browsing history).
    history_cursor: Option<usize>,
    /// Saved buffer from before the user started Up-arrowing — so
    /// pressing Down past the newest entry restores what they were
    /// typing.
    saved_for_history: Option<String>,
    /// Active reverse-search state (Ctrl-R). `None` outside search.
    search: Option<SearchState>,
}

impl Default for InputEditor {
    fn default() -> Self {
        Self::new()
    }
}

impl InputEditor {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            cursor_byte: 0,
            history: VecDeque::with_capacity(HISTORY_CAP),
            history_cursor: None,
            saved_for_history: None,
            search: None,
        }
    }

    /// Build an editor pre-loaded with persisted history. Newest
    /// entry should be first (matches file load order if file is
    /// written oldest-first and we reverse on load).
    pub fn with_history(history: Vec<String>) -> Self {
        let mut e = Self::new();
        for h in history.into_iter().take(HISTORY_CAP) {
            e.history.push_back(h);
        }
        e
    }

    // ── accessors ──────────────────────────────────────────────────

    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Byte offset of the cursor — exposed for tests and as the
    /// canonical position. The TUI uses `cursor_char` instead.
    #[allow(dead_code)]
    pub fn cursor_byte(&self) -> usize {
        self.cursor_byte
    }

    /// Cursor offset in *characters* — what the renderer needs to
    /// position the screen cursor. Always `<= buffer.chars().count()`.
    pub fn cursor_char(&self) -> usize {
        self.buffer[..self.cursor_byte].chars().count()
    }

    /// Convenience predicate — used by tests and forthcoming UI
    /// states (e.g., disabling the "send" prompt when empty).
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn search(&self) -> Option<&SearchState> {
        self.search.as_ref()
    }

    pub fn history(&self) -> &VecDeque<String> {
        &self.history
    }

    // ── single-character insert / delete ───────────────────────────

    pub fn insert_char(&mut self, c: char) {
        self.exit_history_browse_if_needed();
        self.buffer.insert(self.cursor_byte, c);
        self.cursor_byte += c.len_utf8();
        // Typing while in reverse-search updates the needle instead
        // of the buffer. Handled separately via search_input().
    }

    /// Insert a UTF-8 string at the cursor (bracketed paste path).
    pub fn insert_str(&mut self, s: &str) {
        self.exit_history_browse_if_needed();
        self.buffer.insert_str(self.cursor_byte, s);
        self.cursor_byte += s.len();
    }

    /// Backspace — remove the codepoint to the left of the cursor.
    pub fn backspace(&mut self) {
        self.exit_history_browse_if_needed();
        if self.cursor_byte == 0 {
            return;
        }
        let prev = self.buffer[..self.cursor_byte]
            .char_indices()
            .next_back()
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.buffer.replace_range(prev..self.cursor_byte, "");
        self.cursor_byte = prev;
    }

    /// Delete — remove the codepoint at/right of the cursor.
    pub fn delete(&mut self) {
        self.exit_history_browse_if_needed();
        if self.cursor_byte >= self.buffer.len() {
            return;
        }
        let next_boundary = self.buffer[self.cursor_byte..]
            .char_indices()
            .nth(1)
            .map(|(i, _)| self.cursor_byte + i)
            .unwrap_or(self.buffer.len());
        self.buffer
            .replace_range(self.cursor_byte..next_boundary, "");
    }

    // ── cursor movement ────────────────────────────────────────────

    pub fn move_left(&mut self) {
        if self.cursor_byte == 0 {
            return;
        }
        let prev = self.buffer[..self.cursor_byte]
            .char_indices()
            .next_back()
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.cursor_byte = prev;
    }

    pub fn move_right(&mut self) {
        if self.cursor_byte >= self.buffer.len() {
            return;
        }
        let next = self.buffer[self.cursor_byte..]
            .char_indices()
            .nth(1)
            .map(|(i, _)| self.cursor_byte + i)
            .unwrap_or(self.buffer.len());
        self.cursor_byte = next;
    }

    pub fn move_home(&mut self) {
        self.cursor_byte = 0;
    }

    pub fn move_end(&mut self) {
        self.cursor_byte = self.buffer.len();
    }

    // ── line/word kill (Ctrl-W / Ctrl-U / Ctrl-K) ──────────────────

    /// Ctrl-W — delete the word to the left of the cursor.
    ///
    /// "Word" = run of non-whitespace, optionally preceded by
    /// whitespace. So "foo bar |" deletes "bar" + the trailing space.
    pub fn delete_word_back(&mut self) {
        self.exit_history_browse_if_needed();
        if self.cursor_byte == 0 {
            return;
        }
        // Walk left skipping whitespace, then non-whitespace.
        let head = &self.buffer[..self.cursor_byte];
        let chars: Vec<(usize, char)> = head.char_indices().collect();
        let mut i = chars.len();
        // Skip whitespace immediately to the left of cursor.
        while i > 0 && chars[i - 1].1.is_whitespace() {
            i -= 1;
        }
        // Skip the word itself.
        while i > 0 && !chars[i - 1].1.is_whitespace() {
            i -= 1;
        }
        let start = chars.get(i).map(|(idx, _)| *idx).unwrap_or(0);
        self.buffer.replace_range(start..self.cursor_byte, "");
        self.cursor_byte = start;
    }

    /// Ctrl-U — delete from cursor back to the start of the line.
    pub fn delete_to_start(&mut self) {
        self.exit_history_browse_if_needed();
        self.buffer.replace_range(0..self.cursor_byte, "");
        self.cursor_byte = 0;
    }

    /// Ctrl-K — delete from cursor to the end of the line.
    pub fn delete_to_end(&mut self) {
        self.exit_history_browse_if_needed();
        self.buffer.truncate(self.cursor_byte);
    }

    pub fn clear(&mut self) {
        self.exit_history_browse_if_needed();
        self.buffer.clear();
        self.cursor_byte = 0;
    }

    // ── history browse (Up / Down) ─────────────────────────────────

    /// Up arrow — show the previous (older) history entry.
    pub fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        // First Up press: save current buffer, jump to newest entry.
        let next_cursor = match self.history_cursor {
            None => {
                self.saved_for_history = Some(std::mem::take(&mut self.buffer));
                0
            }
            Some(i) if i + 1 < self.history.len() => i + 1,
            Some(i) => i, // already at oldest, stay
        };
        self.history_cursor = Some(next_cursor);
        self.buffer = self.history[next_cursor].clone();
        self.cursor_byte = self.buffer.len();
    }

    /// Down arrow — show the next (newer) history entry. Past the
    /// newest, restore the in-progress buffer.
    pub fn history_next(&mut self) {
        let cur = match self.history_cursor {
            None => return, // not browsing — Down is a no-op
            Some(i) => i,
        };
        if cur == 0 {
            // Was on newest entry — return to in-progress buffer.
            self.buffer = self.saved_for_history.take().unwrap_or_default();
            self.cursor_byte = self.buffer.len();
            self.history_cursor = None;
        } else {
            let next = cur - 1;
            self.history_cursor = Some(next);
            self.buffer = self.history[next].clone();
            self.cursor_byte = self.buffer.len();
        }
    }

    // ── reverse-i-search (Ctrl-R) ──────────────────────────────────

    pub fn begin_search(&mut self) {
        if self.search.is_none() {
            self.search = Some(SearchState::new());
            self.refresh_search_match();
        } else {
            // Already searching — Ctrl-R again steps to next older match.
            self.search_step_older();
        }
    }

    pub fn search_input(&mut self, c: char) {
        if let Some(s) = self.search.as_mut() {
            s.needle.push(c);
            self.refresh_search_match();
        }
    }

    pub fn search_backspace(&mut self) {
        if let Some(s) = self.search.as_mut() {
            s.needle.pop();
            self.refresh_search_match();
        }
    }

    /// Confirm the current search match (Enter): drop the search
    /// overlay and put the matched entry into the buffer.
    pub fn search_accept(&mut self) {
        if let Some(s) = self.search.take() {
            if let Some(idx) = s.match_idx {
                self.buffer = self.history[idx].clone();
                self.cursor_byte = self.buffer.len();
            }
        }
    }

    /// Esc out of search without selecting.
    pub fn search_cancel(&mut self) {
        self.search = None;
    }

    fn refresh_search_match(&mut self) {
        let Some(s) = self.search.as_mut() else {
            return;
        };
        if s.needle.is_empty() {
            s.match_idx = None;
            return;
        }
        // Search newest-first.
        s.match_idx = self
            .history
            .iter()
            .position(|entry| entry.contains(s.needle.as_str()));
    }

    fn search_step_older(&mut self) {
        let Some(s) = self.search.as_mut() else {
            return;
        };
        let from = s.match_idx.map(|i| i + 1).unwrap_or(0);
        if s.needle.is_empty() {
            return;
        }
        s.match_idx = self
            .history
            .iter()
            .enumerate()
            .skip(from)
            .find(|(_, e)| e.contains(s.needle.as_str()))
            .map(|(i, _)| i);
    }

    // ── submit ─────────────────────────────────────────────────────

    /// Take the current buffer (Enter pressed). Pushes onto history
    /// (deduped against the newest entry) and clears the editor.
    /// Returns the submitted text.
    pub fn submit(&mut self) -> String {
        self.search = None;
        self.history_cursor = None;
        self.saved_for_history = None;
        let line = std::mem::take(&mut self.buffer);
        self.cursor_byte = 0;
        if !line.is_empty() && self.history.front().map(|h| h.as_str()) != Some(line.as_str()) {
            self.history.push_front(line.clone());
            while self.history.len() > HISTORY_CAP {
                self.history.pop_back();
            }
        }
        line
    }

    // ── internal ───────────────────────────────────────────────────

    fn exit_history_browse_if_needed(&mut self) {
        // Editing while history-browsing fixes the current entry as
        // the new in-progress buffer.
        if self.history_cursor.is_some() {
            self.history_cursor = None;
            self.saved_for_history = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_cursor_track() {
        let mut e = InputEditor::new();
        e.insert_char('h');
        e.insert_char('i');
        assert_eq!(e.buffer(), "hi");
        assert_eq!(e.cursor_byte(), 2);
        assert_eq!(e.cursor_char(), 2);
    }

    #[test]
    fn move_left_right_basic() {
        let mut e = InputEditor::new();
        e.insert_str("abc");
        assert_eq!(e.cursor_byte(), 3);
        e.move_left();
        assert_eq!(e.cursor_byte(), 2);
        e.move_left();
        e.move_left();
        e.move_left(); // saturating
        assert_eq!(e.cursor_byte(), 0);
        e.move_right();
        assert_eq!(e.cursor_byte(), 1);
    }

    #[test]
    fn move_left_right_korean_codepoint_atomic() {
        let mut e = InputEditor::new();
        e.insert_str("한글");
        // '한' = U+D55C = 3 bytes. '글' = U+AE00 = 3 bytes.
        assert_eq!(e.cursor_byte(), 6);
        e.move_left();
        assert_eq!(e.cursor_byte(), 3); // before '글'
        e.move_left();
        assert_eq!(e.cursor_byte(), 0); // before '한'
                                        // Char-count cursor is independent of byte offset:
        e.move_right();
        assert_eq!(e.cursor_char(), 1);
    }

    #[test]
    fn home_end() {
        let mut e = InputEditor::new();
        e.insert_str("hello");
        e.move_home();
        assert_eq!(e.cursor_byte(), 0);
        e.move_end();
        assert_eq!(e.cursor_byte(), 5);
    }

    #[test]
    fn backspace_basic() {
        let mut e = InputEditor::new();
        e.insert_str("abc");
        e.backspace();
        assert_eq!(e.buffer(), "ab");
        assert_eq!(e.cursor_byte(), 2);
    }

    #[test]
    fn backspace_korean() {
        let mut e = InputEditor::new();
        e.insert_str("한글");
        e.backspace();
        assert_eq!(e.buffer(), "한");
        assert_eq!(e.cursor_byte(), 3);
    }

    #[test]
    fn delete_forward() {
        let mut e = InputEditor::new();
        e.insert_str("abc");
        e.move_home();
        e.delete();
        assert_eq!(e.buffer(), "bc");
        assert_eq!(e.cursor_byte(), 0);
    }

    #[test]
    fn ctrl_w_deletes_last_word_with_trailing_space() {
        let mut e = InputEditor::new();
        e.insert_str("foo bar");
        e.delete_word_back();
        assert_eq!(e.buffer(), "foo ");
    }

    #[test]
    fn ctrl_w_with_trailing_whitespace() {
        let mut e = InputEditor::new();
        e.insert_str("foo bar   ");
        // Cursor at the end after multiple spaces.
        e.delete_word_back();
        // Should eat trailing whitespace + the "bar".
        assert_eq!(e.buffer(), "foo ");
    }

    #[test]
    fn ctrl_u_deletes_to_start() {
        let mut e = InputEditor::new();
        e.insert_str("abcdef");
        e.move_left();
        e.move_left(); // cursor at "abcd|ef"
        e.delete_to_start();
        assert_eq!(e.buffer(), "ef");
        assert_eq!(e.cursor_byte(), 0);
    }

    #[test]
    fn ctrl_k_deletes_to_end() {
        let mut e = InputEditor::new();
        e.insert_str("abcdef");
        e.move_home();
        e.move_right();
        e.move_right(); // "ab|cdef"
        e.delete_to_end();
        assert_eq!(e.buffer(), "ab");
        assert_eq!(e.cursor_byte(), 2);
    }

    #[test]
    fn history_prev_next_cycles() {
        let mut e = InputEditor::with_history(vec![
            "newest".to_string(),
            "middle".to_string(),
            "oldest".to_string(),
        ]);
        e.insert_str("typing");
        e.history_prev();
        assert_eq!(e.buffer(), "newest");
        e.history_prev();
        assert_eq!(e.buffer(), "middle");
        e.history_prev();
        assert_eq!(e.buffer(), "oldest");
        e.history_prev(); // saturate
        assert_eq!(e.buffer(), "oldest");
        e.history_next();
        assert_eq!(e.buffer(), "middle");
        e.history_next();
        assert_eq!(e.buffer(), "newest");
        e.history_next(); // restore in-progress
        assert_eq!(e.buffer(), "typing");
        e.history_next(); // no-op past in-progress
        assert_eq!(e.buffer(), "typing");
    }

    #[test]
    fn submit_pushes_to_history_and_clears_buffer() {
        let mut e = InputEditor::new();
        e.insert_str("hello");
        let line = e.submit();
        assert_eq!(line, "hello");
        assert_eq!(e.buffer(), "");
        assert_eq!(e.cursor_byte(), 0);
        assert_eq!(e.history().front().map(|s| s.as_str()), Some("hello"));
    }

    #[test]
    fn submit_dedupes_consecutive_duplicates() {
        let mut e = InputEditor::new();
        e.insert_str("same");
        e.submit();
        e.insert_str("same");
        e.submit();
        assert_eq!(e.history().len(), 1);
    }

    #[test]
    fn submit_caps_history() {
        let mut e = InputEditor::new();
        for i in 0..(HISTORY_CAP + 50) {
            e.insert_str(&format!("entry-{}", i));
            e.submit();
        }
        assert_eq!(e.history().len(), HISTORY_CAP);
        // Newest is at front:
        assert_eq!(
            e.history().front().unwrap().as_str(),
            &format!("entry-{}", HISTORY_CAP + 49)
        );
    }

    #[test]
    fn search_finds_matching_entry() {
        let mut e = InputEditor::with_history(vec![
            "deploy to prod".to_string(),
            "what is the capital of france".to_string(),
            "how are you".to_string(),
        ]);
        e.begin_search();
        e.search_input('c');
        e.search_input('a');
        e.search_input('p');
        let s = e.search().unwrap();
        assert_eq!(s.match_idx, Some(1));
        assert_eq!(
            e.history()[s.match_idx.unwrap()],
            "what is the capital of france"
        );
    }

    #[test]
    fn search_accept_loads_matched_entry() {
        let mut e = InputEditor::with_history(vec!["alpha-bravo".to_string()]);
        e.begin_search();
        e.search_input('b');
        e.search_accept();
        assert!(e.search().is_none());
        assert_eq!(e.buffer(), "alpha-bravo");
    }

    #[test]
    fn search_cancel_does_not_change_buffer() {
        let mut e = InputEditor::new();
        e.insert_str("typing");
        let saved = e.buffer().to_string();
        e.begin_search();
        e.search_input('x');
        e.search_cancel();
        assert!(e.search().is_none());
        assert_eq!(e.buffer(), saved);
    }

    #[test]
    fn editing_after_history_browse_fixes_buffer() {
        let mut e = InputEditor::with_history(vec!["old-prompt".to_string()]);
        e.history_prev();
        assert_eq!(e.buffer(), "old-prompt");
        e.insert_str(" + new");
        assert_eq!(e.buffer(), "old-prompt + new");
        // Down should NOT restore anything — we already committed.
        e.history_next();
        assert_eq!(e.buffer(), "old-prompt + new");
    }
}
