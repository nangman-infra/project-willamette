//! Stage 9-E — ratatui-based chat TUI over [`super::ChatEngine`].
//!
//! Layout (rows top → bottom):
//!
//! ```text
//! ┌─ Project Willamette — BitNet b1.58 chat ────── ctx 23/2048 ─┐
//! │ USR: hello                                                   │
//! │ BOT: Hi! How can I help you today?                           │
//! │                                                              │
//! ├──────────────────────────────────────────────────────────────┤
//! │ status: idle  ·  /help for commands  ·  Ctrl-C to quit       │
//! ├──────────────────────────────────────────────────────────────┤
//! │ > _                                                           │
//! └──────────────────────────────────────────────────────────────┘
//! ```
//!
//! Concurrency model: a single worker thread owns the `ChatEngine`
//! and the model graph borrow. The main UI thread pushes user
//! messages and slash commands into a `Sender<UserCmd>`; the worker
//! emits `TokenEvent`s back as the model streams its response.
//!
//! Why a worker thread? `engine.send_user_message` is blocking — if
//! we ran it inline the TUI would freeze for the whole generation.
//! With the worker we can keep redrawing on every token chunk that
//! arrives, plus poll keyboard events for `Esc`/Ctrl-C.

use std::io::Stdout;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;

use super::ChatEngine;

/// What the UI sends to the worker.
enum UserCmd {
    Send(String),
    Reset,
    SetSys(Option<String>),
    Shutdown,
}

/// What the worker sends back to the UI.
enum TokenEvent {
    /// New chunk of text from the assistant's current turn.
    Chunk(String),
    /// Turn finished cleanly. Total time + chars/s.
    Done { secs: f64, chars: usize },
    /// Generation errored.
    Failed(String),
    /// Engine state changed (e.g. after /reset). Carries fresh ctx pos.
    StateChanged { token_position: u32 },
}

#[derive(Clone)]
struct DisplayMsg {
    role: Role,
    content: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    User,
    Bot,
    System,
}

impl Role {
    fn label(self) -> &'static str {
        match self {
            Role::User => "USR",
            Role::Bot => "BOT",
            Role::System => "SYS",
        }
    }
    fn color(self) -> Color {
        match self {
            Role::User => Color::Cyan,
            Role::Bot => Color::Green,
            Role::System => Color::Magenta,
        }
    }
}

struct UiState {
    history: Vec<DisplayMsg>,
    /// In-flight assistant response (not yet pushed to history).
    streaming: String,
    /// Input line buffer.
    input: String,
    /// Current status text shown in the status bar.
    status: String,
    /// True while the worker is mid-generation.
    generating: bool,
    /// Last-known KV cache position (updated as events arrive).
    token_position: u32,
    /// Cap reported in the header.
    max_seq_len: usize,
    /// Sticky generation timing (for the status bar after Done).
    last_turn_secs: Option<f64>,
    last_turn_chars: Option<usize>,
    /// Scroll offset for the history pane (lines from bottom).
    scroll_back: u16,
    /// Show transient hint until cleared.
    transient: Option<(String, Instant)>,
}

impl UiState {
    fn new(max_seq_len: usize) -> Self {
        Self {
            history: Vec::new(),
            streaming: String::new(),
            input: String::new(),
            status: "idle · type, then Enter".to_string(),
            generating: false,
            token_position: 0,
            max_seq_len,
            last_turn_secs: None,
            last_turn_chars: None,
            scroll_back: 0,
            transient: None,
        }
    }

    fn flash(&mut self, msg: impl Into<String>) {
        self.transient = Some((msg.into(), Instant::now()));
    }
}

/// Entry point. Loads the model, builds a [`ChatEngine`], and runs
/// the TUI to completion.
#[allow(clippy::too_many_arguments)]
pub fn run_tui(mut engine: ChatEngine<'_, '_>, max_new_tokens: usize) -> Result<()> {
    let max_seq = engine.max_seq_len();

    // Channels.
    let (cmd_tx, cmd_rx) = mpsc::channel::<UserCmd>();
    let (evt_tx, evt_rx) = mpsc::channel::<TokenEvent>();

    // Terminal setup.
    enable_raw_mode().map_err(|e| anyhow::anyhow!("enable raw mode: {}", e))?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)
        .map_err(|e| anyhow::anyhow!("enter alt screen: {}", e))?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal =
        Terminal::new(backend).map_err(|e| anyhow::anyhow!("ratatui terminal: {}", e))?;

    let mut ui = UiState::new(max_seq);
    ui.token_position = engine.token_position();

    let run_result = thread::scope(|s| -> Result<()> {
        // Worker thread.
        let worker_evt_tx = evt_tx.clone();
        s.spawn(move || {
            worker_loop(&mut engine, max_new_tokens, cmd_rx, worker_evt_tx);
        });

        // Main UI loop.
        ui_loop(&mut terminal, &mut ui, cmd_tx, evt_rx)
    });

    // Terminal teardown — always run, even on error.
    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();

    run_result
}

fn worker_loop<'g, 'a>(
    engine: &mut ChatEngine<'g, 'a>,
    max_new_tokens: usize,
    cmd_rx: Receiver<UserCmd>,
    evt_tx: Sender<TokenEvent>,
) {
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            UserCmd::Send(text) => {
                let start = Instant::now();
                let tx = evt_tx.clone();
                let result = engine.send_user_message(&text, max_new_tokens, |chunk| {
                    let _ = tx.send(TokenEvent::Chunk(chunk.to_string()));
                });
                let secs = start.elapsed().as_secs_f64();
                let _ = evt_tx.send(TokenEvent::StateChanged {
                    token_position: engine.token_position(),
                });
                match result {
                    Ok(response) => {
                        let _ = evt_tx.send(TokenEvent::Done {
                            secs,
                            chars: response.chars().count(),
                        });
                    }
                    Err(e) => {
                        let _ = evt_tx.send(TokenEvent::Failed(e.to_string()));
                    }
                }
            }
            UserCmd::Reset => {
                engine.reset();
                let _ = evt_tx.send(TokenEvent::StateChanged {
                    token_position: engine.token_position(),
                });
            }
            UserCmd::SetSys(s) => {
                engine.set_system_prompt(s);
            }
            UserCmd::Shutdown => break,
        }
    }
}

fn ui_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    ui: &mut UiState,
    cmd_tx: Sender<UserCmd>,
    evt_rx: Receiver<TokenEvent>,
) -> Result<()> {
    let tick_period = Duration::from_millis(33); // ~30 fps redraws
    let mut last_tick = Instant::now();

    loop {
        terminal
            .draw(|f| render(f, ui))
            .map_err(|e| anyhow::anyhow!("draw: {}", e))?;

        // Drain any pending TokenEvents (non-blocking).
        loop {
            match evt_rx.try_recv() {
                Ok(TokenEvent::Chunk(c)) => {
                    ui.streaming.push_str(&c);
                }
                Ok(TokenEvent::Done { secs, chars }) => {
                    let resp = std::mem::take(&mut ui.streaming);
                    ui.history.push(DisplayMsg {
                        role: Role::Bot,
                        content: resp,
                    });
                    ui.generating = false;
                    ui.last_turn_secs = Some(secs);
                    ui.last_turn_chars = Some(chars);
                    let cps = if secs > 0.0 { chars as f64 / secs } else { 0.0 };
                    ui.status = format!("idle · last turn {:.1}s ({:.1} chars/s)", secs, cps);
                }
                Ok(TokenEvent::Failed(msg)) => {
                    let partial = std::mem::take(&mut ui.streaming);
                    if !partial.is_empty() {
                        ui.history.push(DisplayMsg {
                            role: Role::Bot,
                            content: partial,
                        });
                    }
                    ui.generating = false;
                    ui.status = format!("error: {}", msg);
                }
                Ok(TokenEvent::StateChanged { token_position }) => {
                    ui.token_position = token_position;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => return Ok(()),
            }
        }

        // Clear transient after 2.5s.
        if let Some((_, when)) = ui.transient {
            if when.elapsed() > Duration::from_millis(2500) {
                ui.transient = None;
            }
        }

        // Poll input with a timeout that yields to redraws even when
        // nothing's happening.
        let timeout = tick_period.saturating_sub(last_tick.elapsed());
        if event::poll(timeout).map_err(|e| anyhow::anyhow!("poll: {}", e))? {
            if let Event::Key(key) = event::read().map_err(|e| anyhow::anyhow!("read: {}", e))? {
                if handle_key(ui, key, &cmd_tx)? {
                    let _ = cmd_tx.send(UserCmd::Shutdown);
                    return Ok(());
                }
            }
        }
        if last_tick.elapsed() >= tick_period {
            last_tick = Instant::now();
        }
    }
}

/// Returns `true` to request quit.
fn handle_key(ui: &mut UiState, key: KeyEvent, cmd_tx: &Sender<UserCmd>) -> Result<bool> {
    // Global: Ctrl-C quits regardless of mode.
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return Ok(true);
    }

    if ui.generating {
        // While generating, only Esc and Ctrl-C react; everything else
        // queues into the (read-only) input buffer-ish behavior — we
        // just ignore typing so the user can't desync.
        if matches!(key.code, KeyCode::Esc) {
            // Stage 9-E doesn't implement mid-turn cancel — generation
            // runs to EOS or max_new_tokens. Flash a hint so the user
            // knows their Esc was seen.
            ui.flash("(Esc ignored mid-turn — Stage 9-E doesn't support cancel yet)");
        }
        return Ok(false);
    }

    match key.code {
        KeyCode::Esc => return Ok(true),
        KeyCode::Enter => {
            let line = ui.input.trim().to_string();
            ui.input.clear();
            if line.is_empty() {
                return Ok(false);
            }
            if let Some(rest) = line.strip_prefix('/') {
                return handle_slash(ui, rest, cmd_tx);
            }
            // Real user message.
            ui.history.push(DisplayMsg {
                role: Role::User,
                content: line.clone(),
            });
            ui.streaming.clear();
            ui.generating = true;
            ui.status = "generating…".to_string();
            ui.scroll_back = 0;
            cmd_tx
                .send(UserCmd::Send(line))
                .map_err(|e| anyhow::anyhow!("send: {}", e))?;
        }
        KeyCode::Backspace => {
            ui.input.pop();
        }
        KeyCode::Char(c) => {
            ui.input.push(c);
        }
        KeyCode::PageUp => {
            ui.scroll_back = ui.scroll_back.saturating_add(5);
        }
        KeyCode::PageDown => {
            ui.scroll_back = ui.scroll_back.saturating_sub(5);
        }
        _ => {}
    }
    Ok(false)
}

fn handle_slash(ui: &mut UiState, cmd: &str, cmd_tx: &Sender<UserCmd>) -> Result<bool> {
    let (head, rest) = cmd
        .split_once(char::is_whitespace)
        .map(|(h, r)| (h, r.trim()))
        .unwrap_or((cmd, ""));
    match head {
        "quit" | "exit" | "q" => Ok(true),
        "help" | "?" => {
            ui.history.push(DisplayMsg {
                role: Role::System,
                content:
                    "Commands: /help /reset /sys <text|off> /clear /quit  ·  PgUp/PgDn scrolls"
                        .to_string(),
            });
            Ok(false)
        }
        "reset" => {
            cmd_tx
                .send(UserCmd::Reset)
                .map_err(|e| anyhow::anyhow!("reset: {}", e))?;
            ui.history.push(DisplayMsg {
                role: Role::System,
                content: "[history cleared; KV cache reset]".to_string(),
            });
            Ok(false)
        }
        "clear" => {
            ui.history.clear();
            Ok(false)
        }
        "sys" => {
            if rest.is_empty() {
                ui.flash("usage: /sys <text>  or  /sys off");
                return Ok(false);
            }
            let payload = if rest == "off" {
                None
            } else {
                Some(rest.to_string())
            };
            cmd_tx
                .send(UserCmd::SetSys(payload.clone()))
                .map_err(|e| anyhow::anyhow!("sys: {}", e))?;
            ui.history.push(DisplayMsg {
                role: Role::System,
                content: if payload.is_some() {
                    format!(
                        "[system prompt set ({} chars) — takes effect after next /reset]",
                        rest.len()
                    )
                } else {
                    "[system prompt cleared]".to_string()
                },
            });
            Ok(false)
        }
        other => {
            ui.flash(format!("unknown command: /{} (try /help)", other));
            Ok(false)
        }
    }
}

fn render(f: &mut ratatui::Frame, ui: &UiState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(1),    // history
            Constraint::Length(1), // status
            Constraint::Length(3), // input
        ])
        .split(f.area());

    // ── History area ──
    let mut lines: Vec<Line> = Vec::new();
    for msg in &ui.history {
        let role_span = Span::styled(
            format!(" {} ", msg.role.label()),
            Style::default()
                .fg(Color::Black)
                .bg(msg.role.color())
                .add_modifier(Modifier::BOLD),
        );
        let body_span = Span::raw(format!(" {}", msg.content));
        lines.push(Line::from(vec![role_span, body_span]));
        lines.push(Line::from(""));
    }
    if !ui.streaming.is_empty() || ui.generating {
        let role_span = Span::styled(
            " BOT ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Green)
                .add_modifier(Modifier::BOLD),
        );
        let body_span = Span::raw(format!(" {}", ui.streaming));
        let cursor = Span::styled(" ▌", Style::default().fg(Color::Green));
        lines.push(Line::from(vec![role_span, body_span, cursor]));
    }

    let pct = if ui.max_seq_len > 0 {
        100.0 * ui.token_position as f64 / ui.max_seq_len as f64
    } else {
        0.0
    };
    let title = format!(
        " Project Willamette — BitNet b1.58 chat   ·   ctx {}/{}  ({:.0}%) ",
        ui.token_position, ui.max_seq_len, pct
    );
    let history_para = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title(title)
                .title_style(Style::default().add_modifier(Modifier::BOLD)),
        )
        .wrap(Wrap { trim: false })
        .scroll((ui.scroll_back, 0));
    f.render_widget(history_para, chunks[0]);

    // ── Status bar ──
    let status_text = if let Some((msg, _)) = &ui.transient {
        format!("  {}", msg)
    } else {
        format!(
            "  {}   ·   /help · Ctrl-C exit · PgUp/PgDn scroll",
            ui.status
        )
    };
    let status_color = if ui.generating {
        Color::Yellow
    } else {
        Color::Gray
    };
    let status = Paragraph::new(Line::from(Span::styled(
        status_text,
        Style::default().fg(status_color),
    )));
    f.render_widget(status, chunks[1]);

    // ── Input area ──
    let prompt_prefix = if ui.generating { "  (busy) " } else { "  > " };
    let input_line = format!("{}{}_", prompt_prefix, ui.input);
    let input_para = Paragraph::new(input_line).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" input ")
            .title_style(Style::default().add_modifier(Modifier::BOLD)),
    );
    f.render_widget(input_para, chunks[2]);
}
