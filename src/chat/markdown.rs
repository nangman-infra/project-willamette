//! Minimal markdown renderer for chat output in the ratatui TUI.
//!
//! The base BitNet model writes responses in light-markdown form
//! (`**bold**`, `1.`, `-`, `` `code` ``, occasional `#` headings).
//! Rendering them as raw text makes the TUI look like a code dump.
//! This module converts each line into a styled
//! [`ratatui::text::Line`] so headings appear bold + coloured, bullets
//! get a `•` glyph, numbered items keep their counter coloured, and
//! inline `**bold**` / `` `code` `` show up properly.
//!
//! Intentionally NOT a full CommonMark parser. The model never emits
//! HTML, link references, or table syntax; pulling in
//! `pulldown-cmark` for this would be over-engineering. The
//! `parse_line` function is a one-pass scanner over the characters.
//!
//! What's covered:
//! * `# foo`, `## foo`, `### foo` — bold + cyan headings (any of the
//!   three levels uses the same style; chat doesn't need real heading
//!   hierarchy).
//! * `- foo` or `* foo` — bullet, prefixed with `•` in yellow.
//! * `123. foo` — numbered list, counter in yellow.
//! * `**bold**` — bold inline (across the rest of the line).
//! * `` `code` `` — yellow inline code.
//! * Leading indent (spaces / tabs) is preserved before applying
//!   list / heading detection on the trimmed part.
//!
//! What's NOT covered (renders as plain text):
//! * `*italic*` — too easy to confuse with `**bold**` half-matches in
//!   streaming output, where a single `*` may arrive before its
//!   closing pair.
//! * Code fences (```` ``` ````): the fence line itself renders as
//!   text; the content inside is treated as ordinary lines (inline
//!   markdown still applies).
//! * Block quotes (`> foo`).
//! * Hyperlinks `[text](url)`.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

/// Convert `text` into a series of styled lines suitable for
/// [`ratatui::widgets::Paragraph`]. Splits on `\n`; each input line
/// becomes one [`Line`].
pub fn render_markdown_lines(text: &str) -> Vec<Line<'static>> {
    text.split('\n').map(parse_line).collect()
}

fn parse_line(line: &str) -> Line<'static> {
    // Preserve leading whitespace as a raw Span so nested-list indent
    // survives.
    let trimmed = line.trim_start();
    let indent: String = line.chars().take_while(|c| c.is_whitespace()).collect();

    // ── headings ──
    for prefix in &["### ", "## ", "# "] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return heading_line(&indent, rest);
        }
    }

    // ── bullet list ──
    if let Some(rest) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
    {
        let mut spans = vec![
            Span::raw(indent),
            Span::styled("• ", Style::default().fg(Color::Yellow)),
        ];
        spans.extend(parse_inline(rest));
        return Line::from(spans);
    }

    // ── numbered list ── ("N. " or "NN. " ...)
    if let Some(dot_pos) = trimmed.find(". ") {
        let counter = &trimmed[..dot_pos];
        if !counter.is_empty() && counter.chars().all(|c| c.is_ascii_digit()) {
            let mut spans = vec![
                Span::raw(indent),
                Span::styled(format!("{}. ", counter), Style::default().fg(Color::Yellow)),
            ];
            spans.extend(parse_inline(&trimmed[dot_pos + 2..]));
            return Line::from(spans);
        }
    }

    // ── default: inline only ──
    let mut spans = vec![Span::raw(indent)];
    spans.extend(parse_inline(trimmed));
    Line::from(spans)
}

fn heading_line(indent: &str, body: &str) -> Line<'static> {
    Line::from(vec![
        Span::raw(indent.to_string()),
        Span::styled(
            body.to_string(),
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(Color::Cyan),
        ),
    ])
}

/// Tokenise inline syntax (`**bold**`, `` `code` ``) into styled spans.
fn parse_inline(text: &str) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut buf = String::new();
    let mut i = 0;

    while i < chars.len() {
        // **bold**
        if chars[i] == '*' && chars.get(i + 1) == Some(&'*') {
            if let Some(close) = find_double_star(&chars, i + 2) {
                flush_buf(&mut buf, &mut spans);
                let body: String = chars[i + 2..close].iter().collect();
                spans.push(Span::styled(
                    body,
                    Style::default().add_modifier(Modifier::BOLD),
                ));
                i = close + 2;
                continue;
            }
        }
        // `code`
        if chars[i] == '`' {
            if let Some(rel) = chars[i + 1..].iter().position(|&c| c == '`') {
                flush_buf(&mut buf, &mut spans);
                let body: String = chars[i + 1..i + 1 + rel].iter().collect();
                spans.push(Span::styled(body, Style::default().fg(Color::Yellow)));
                i = i + 1 + rel + 1;
                continue;
            }
        }
        buf.push(chars[i]);
        i += 1;
    }
    flush_buf(&mut buf, &mut spans);
    spans
}

fn flush_buf(buf: &mut String, spans: &mut Vec<Span<'static>>) {
    if !buf.is_empty() {
        spans.push(Span::raw(std::mem::take(buf)));
    }
}

fn find_double_star(chars: &[char], from: usize) -> Option<usize> {
    let mut i = from;
    while i + 1 < chars.len() {
        if chars[i] == '*' && chars[i + 1] == '*' {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::style::Modifier;

    #[test]
    fn plain_text_passes_through_with_indent_span() {
        let lines = render_markdown_lines("hello world");
        assert_eq!(lines.len(), 1);
        let spans = &lines[0].spans;
        // Two spans: empty-indent + "hello world"
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].content, "");
        assert_eq!(spans[1].content, "hello world");
    }

    #[test]
    fn double_star_renders_bold() {
        let lines = render_markdown_lines("a **bold** b");
        let spans = &lines[0].spans;
        // [indent, "a ", "bold", " b"]
        assert_eq!(spans.len(), 4);
        assert_eq!(spans[2].content, "bold");
        assert!(
            spans[2].style.add_modifier.contains(Modifier::BOLD),
            "the bold span must carry the BOLD modifier"
        );
        // surrounding spans must NOT be bold
        assert!(!spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert!(!spans[3].style.add_modifier.contains(Modifier::BOLD));
    }

    #[test]
    fn unclosed_double_star_stays_literal() {
        let lines = render_markdown_lines("a **only one");
        let spans = &lines[0].spans;
        // Should NOT be bolded; the `**` stays as literal text.
        let joined: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(joined, "a **only one");
        assert!(!spans
            .iter()
            .any(|s| s.style.add_modifier.contains(Modifier::BOLD)));
    }

    #[test]
    fn backtick_renders_inline_code() {
        let lines = render_markdown_lines("call `foo()` here");
        let spans = &lines[0].spans;
        // [indent, "call ", "foo()", " here"]
        assert_eq!(spans.len(), 4);
        assert_eq!(spans[2].content, "foo()");
        assert_eq!(spans[2].style.fg, Some(Color::Yellow));
    }

    #[test]
    fn bullet_list_gets_diamond_glyph() {
        let lines = render_markdown_lines("- first item");
        let spans = &lines[0].spans;
        // [indent, "• ", inline-spans-of-"first item"]
        assert_eq!(spans[1].content, "• ");
        assert_eq!(spans[1].style.fg, Some(Color::Yellow));
    }

    #[test]
    fn bullet_with_indent_preserves_indent() {
        let lines = render_markdown_lines("    - nested");
        let spans = &lines[0].spans;
        assert_eq!(spans[0].content, "    ");
        assert_eq!(spans[1].content, "• ");
    }

    #[test]
    fn star_bullet_also_recognized() {
        let lines = render_markdown_lines("* alt-bullet");
        let spans = &lines[0].spans;
        assert_eq!(spans[1].content, "• ");
    }

    #[test]
    fn numbered_list_keeps_counter_coloured() {
        let lines = render_markdown_lines("3. third item");
        let spans = &lines[0].spans;
        // [indent, "3. ", inline...]
        assert_eq!(spans[1].content, "3. ");
        assert_eq!(spans[1].style.fg, Some(Color::Yellow));
    }

    #[test]
    fn multidigit_numbered_list_works() {
        let lines = render_markdown_lines("12. twelfth");
        let spans = &lines[0].spans;
        assert_eq!(spans[1].content, "12. ");
    }

    #[test]
    fn period_in_normal_text_is_not_a_list_item() {
        // Adversarial: "v0.2.3 is out" should NOT be parsed as a
        // numbered list ("v0" → not all digits → falls through).
        let lines = render_markdown_lines("v0.2.3 is out");
        let spans = &lines[0].spans;
        let joined: String = spans.iter().map(|s| s.content.as_ref()).collect();
        assert_eq!(joined, "v0.2.3 is out");
    }

    #[test]
    fn h1_heading() {
        let lines = render_markdown_lines("# Hello");
        let spans = &lines[0].spans;
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[1].content, "Hello");
        assert!(spans[1].style.add_modifier.contains(Modifier::BOLD));
        assert_eq!(spans[1].style.fg, Some(Color::Cyan));
    }

    #[test]
    fn h2_and_h3_use_same_style() {
        let h2 = render_markdown_lines("## h2");
        let h3 = render_markdown_lines("### h3");
        assert_eq!(h2[0].spans[1].style, h3[0].spans[1].style);
    }

    #[test]
    fn multiple_lines_render_separately() {
        let lines = render_markdown_lines("line one\nline two");
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn realistic_korean_history_response_renders() {
        // Lifted (and shortened) from the actual TUI output the user
        // showed: numbered list with **bold** headings inside each
        // item, plus nested bullets. Written as one explicit-`\n`
        // line because `\` line continuation in a Rust string literal
        // eats the leading whitespace on the next physical line,
        // which would drop the nested bullet indent we want to test.
        let raw = "Korean history:\n1. **Ancient Korea**\n   - The Korean Peninsula was inhabited by various cultures.\n2. **Joseon Dynasty (1392–1910)**\n   - Founded by Yi Sun-sin.";
        let lines = render_markdown_lines(raw);
        assert_eq!(lines.len(), 5);
        // First line: "Korean history:" (no list/heading)
        assert!(!lines[0].spans.iter().any(|s| s.content == "1. "));
        // Second line: numbered "1. "
        assert!(lines[1].spans.iter().any(|s| s.content == "1. "));
        // Inside line 2 there's a bold "Ancient Korea"
        assert!(
            lines[1]
                .spans
                .iter()
                .any(|s| s.content == "Ancient Korea"
                    && s.style.add_modifier.contains(Modifier::BOLD))
        );
        // Third line: nested bullet
        let l3 = &lines[2].spans;
        assert_eq!(l3[0].content, "   "); // indent before "- "
        assert_eq!(l3[1].content, "• ");
    }
}
