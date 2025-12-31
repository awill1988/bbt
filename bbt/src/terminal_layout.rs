//! Terminal layout with fixed header and log region using ratatui.

use crossterm::{
    cursor::{Hide, Show},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode},
};
use indicatif::HumanBytes;
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::Line,
    widgets::{Gauge, Paragraph, Wrap},
    Frame, Terminal,
};
use std::collections::VecDeque;
use std::io::{self, IsTerminal};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const STATUS_LINES: u16 = 5;
const BANNER_LINES: [&str; 5] = [
    "\u{250c}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2510}",
    "\u{2502}\u{250f}\u{2513} \u{257b}\u{250f}\u{2501}\u{2578}   \u{250f}\u{2513} \u{250f}\u{2501}\u{2513}\u{250f}\u{2501}\u{2513}\u{257b}\u{250f}\u{2513}\u{257b}   \u{257a}\u{2533}\u{2578}\u{257b}\u{250f}\u{2533}\u{2513}\u{250f}\u{2501}\u{2578}\u{2502}",
    "\u{2502}\u{2523}\u{253b}\u{2513}\u{2503}\u{2503}\u{257a}\u{2513}   \u{2523}\u{253b}\u{2513}\u{2523}\u{2533}\u{251b}\u{2523}\u{2501}\u{252b}\u{2503}\u{2503}\u{2517}\u{252b}    \u{2503} \u{2503}\u{2503}\u{2503}\u{2503}\u{2523}\u{2578} \u{2502}",
    "\u{2502}\u{2517}\u{2501}\u{251b}\u{2579}\u{2517}\u{2501}\u{251b}   \u{2517}\u{2501}\u{251b}\u{2579}\u{2517}\u{2578}\u{2579} \u{2579}\u{2579}\u{2579} \u{2579}    \u{2579} \u{2579}\u{2579} \u{2579}\u{2517}\u{2501}\u{2578}\u{2502}",
    "\u{2514}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2518}",
];
const HEADER_LINES: u16 = STATUS_LINES + BANNER_LINES.len() as u16;
const LOG_BUFFER_CAPACITY: usize = 2000;
const TICK_RATE: Duration = Duration::from_millis(100);
const SPINNER_FRAMES: [char; 4] = ['|', '/', '-', '\\'];

#[derive(Clone, Default)]
struct ProgressState {
    label: String,
    position: u64,
    total: u64,
    message: String,
    done: bool,
}

#[derive(Clone, Default)]
struct UiState {
    scan_message: String,
    scan_done: bool,
    resource_message: String,
    files: ProgressState,
    bytes: ProgressState,
    current_file: String,
    current_size_bytes: u64,
}

pub struct TerminalUi {
    enabled: bool,
    state: Arc<Mutex<UiState>>,
    log_sender: Option<Sender<String>>,
    stop_flag: Option<Arc<AtomicBool>>,
    render_handle: Option<JoinHandle<()>>,
}

#[derive(Clone)]
pub struct TerminalUiHandle {
    enabled: bool,
    state: Arc<Mutex<UiState>>,
}

impl TerminalUi {
    pub fn new() -> io::Result<Self> {
        if !io::stdout().is_terminal() {
            return Ok(Self::disabled());
        }

        let state = Arc::new(Mutex::new(UiState::default()));
        if let Ok(mut guard) = state.lock() {
            guard.files.label = "files".to_string();
            guard.bytes.label = "bytes".to_string();
        }

        let (log_tx, log_rx) = mpsc::channel::<String>();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let render_state = Arc::clone(&state);
        let stop_flag_clone = Arc::clone(&stop_flag);

        let render_handle = thread::spawn(move || {
            run_ui(render_state, log_rx, stop_flag_clone);
        });

        Ok(Self {
            enabled: true,
            state,
            log_sender: Some(log_tx),
            stop_flag: Some(stop_flag),
            render_handle: Some(render_handle),
        })
    }

    fn disabled() -> Self {
        Self {
            enabled: false,
            state: Arc::new(Mutex::new(UiState::default())),
            log_sender: None,
            stop_flag: None,
            render_handle: None,
        }
    }

    pub fn handle(&self) -> TerminalUiHandle {
        TerminalUiHandle {
            enabled: self.enabled,
            state: Arc::clone(&self.state),
        }
    }

    pub fn log_sender(&self) -> Option<Sender<String>> {
        self.log_sender.clone()
    }
}

impl Drop for TerminalUi {
    fn drop(&mut self) {
        if let Some(flag) = &self.stop_flag {
            flag.store(true, Ordering::Relaxed);
        }

        if let Some(handle) = self.render_handle.take() {
            let _ = handle.join();
        }
    }
}

impl TerminalUiHandle {
    fn update(&self, update_fn: impl FnOnce(&mut UiState)) {
        if !self.enabled {
            return;
        }

        if let Ok(mut guard) = self.state.lock() {
            update_fn(&mut guard);
        }
    }

    pub fn set_scan_message(&self, message: impl Into<String>) {
        self.update(|state| {
            state.scan_message = message.into();
        });
    }

    pub fn finish_scan(&self) {
        self.update(|state| {
            state.scan_done = true;
        });
    }

    pub fn set_resource_message(&self, message: impl Into<String>) {
        self.update(|state| {
            state.resource_message = message.into();
        });
    }

    pub fn set_files_total(&self, total: u64) {
        self.update(|state| {
            state.files.total = total;
        });
    }

    pub fn set_bytes_total(&self, total: u64) {
        self.update(|state| {
            state.bytes.total = total;
        });
    }

    pub fn set_files_position(&self, position: u64) {
        self.update(|state| {
            state.files.position = position;
        });
    }

    pub fn set_bytes_position(&self, position: u64) {
        self.update(|state| {
            state.bytes.position = position;
        });
    }

    pub fn inc_files(&self, delta: u64) {
        self.update(|state| {
            state.files.position = state.files.position.saturating_add(delta);
        });
    }

    pub fn inc_bytes(&self, delta: u64) {
        self.update(|state| {
            state.bytes.position = state.bytes.position.saturating_add(delta);
        });
    }

    pub fn set_files_message(&self, message: impl Into<String>) {
        self.update(|state| {
            state.files.message = message.into();
        });
    }

    pub fn set_bytes_message(&self, message: impl Into<String>) {
        self.update(|state| {
            state.bytes.message = message.into();
        });
    }

    pub fn finish_files(&self, message: impl Into<String>) {
        self.update(|state| {
            state.files.done = true;
            state.files.message = message.into();
            state.files.position = state.files.total;
        });
    }

    pub fn finish_bytes(&self, message: impl Into<String>) {
        self.update(|state| {
            state.bytes.done = true;
            state.bytes.message = message.into();
            state.bytes.position = state.bytes.total;
        });
    }

    pub fn set_current_file(&self, file: impl Into<String>, size_bytes: u64) {
        self.update(|state| {
            state.current_file = file.into();
            state.current_size_bytes = size_bytes;
        });
    }

    pub fn files_position(&self) -> u64 {
        if !self.enabled {
            return 0;
        }

        self.state
            .lock()
            .map(|state| state.files.position)
            .unwrap_or(0)
    }
}

fn run_ui(state: Arc<Mutex<UiState>>, log_rx: Receiver<String>, stop: Arc<AtomicBool>) {
    let mut stdout = io::stdout();
    if enable_raw_mode().is_err() {
        return;
    }
    if execute!(stdout, Hide).is_err() {
        let _ = disable_raw_mode();
        return;
    }

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = match Terminal::new(backend) {
        Ok(terminal) => terminal,
        Err(_) => {
            let _ = disable_raw_mode();
            return;
        }
    };

    let _ = terminal.clear();

    let mut log_lines: VecDeque<String> = VecDeque::with_capacity(LOG_BUFFER_CAPACITY);
    let mut spinner_index = 0usize;

    while !stop.load(Ordering::Relaxed) {
        drain_logs(&log_rx, &mut log_lines);
        let snapshot = match state.lock() {
            Ok(guard) => guard.clone(),
            Err(_) => continue,
        };

        let _ = terminal.draw(|frame| {
            draw_ui(frame, &snapshot, &log_lines, spinner_index);
        });

        spinner_index = (spinner_index + 1) % SPINNER_FRAMES.len();
        thread::sleep(TICK_RATE);
    }

    let _ = terminal.clear();
    let _ = execute!(terminal.backend_mut(), Show);
    let _ = disable_raw_mode();
}

fn drain_logs(log_rx: &Receiver<String>, log_lines: &mut VecDeque<String>) {
    while let Ok(line) = log_rx.try_recv() {
        let cleaned = line.trim_end_matches('\n').trim_end_matches('\r');
        if cleaned.is_empty() {
            continue;
        }
        log_lines.push_back(cleaned.to_string());
        while log_lines.len() > LOG_BUFFER_CAPACITY {
            log_lines.pop_front();
        }
    }
}

fn draw_ui(frame: &mut Frame, state: &UiState, log_lines: &VecDeque<String>, spinner_index: usize) {
    let area = frame.area();
    if area.height < 3 {
        return;
    }

    let header_height = HEADER_LINES.min(area.height.saturating_sub(2));
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(header_height),
            Constraint::Length(1),
            Constraint::Min(1),
        ])
        .split(area);

    if header_height > 0 {
        render_header(frame, layout[0], state, spinner_index);
    }

    render_separator(frame, layout[1]);
    render_logs(frame, layout[2], log_lines);
}

fn render_separator(frame: &mut Frame, area: Rect) {
    let width = area.width as usize;
    if width == 0 {
        return;
    }
    let line = "-".repeat(width);
    frame.render_widget(Paragraph::new(line), area);
}

fn render_logs(frame: &mut Frame, area: Rect, log_lines: &VecDeque<String>) {
    let width = area.width as usize;
    if width == 0 {
        return;
    }

    let max_lines = area.height as usize;
    let start = log_lines.len().saturating_sub(max_lines);
    let mut lines = Vec::with_capacity(max_lines);
    for line in log_lines.iter().skip(start) {
        lines.push(Line::from(format_log_line(line, width)));
    }

    let paragraph = Paragraph::new(lines).wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn build_scan_line(state: &UiState, spinner: char, area: Rect) -> Line<'_> {
    let content = if state.scan_done {
        format!("scan done: {}", state.scan_message)
    } else {
        format!("scan {} {}", spinner, state.scan_message)
    };
    Line::from(truncate_head(&content, area.width as usize))
}

fn build_resource_line(state: &UiState, spinner: char, area: Rect) -> Line<'_> {
    let message = if state.resource_message.is_empty() {
        "idle".to_string()
    } else {
        state.resource_message.clone()
    };
    let content = format!("resource {} {}", spinner, message);
    Line::from(truncate_head(&content, area.width as usize))
}

fn build_current_line(state: &UiState, _spinner: char, area: Rect) -> Line<'_> {
    let size = format!("{}", HumanBytes(state.current_size_bytes));
    let prefix = "current: ";
    let suffix = format!(" size: {}", size);
    let width = area.width as usize;
    let available = width
        .saturating_sub(prefix.len())
        .saturating_sub(suffix.len());
    let file = truncate_tail(&state.current_file, available);
    let content = format!("{}{}{}", prefix, file, suffix);
    Line::from(truncate_head(&content, width))
}

fn build_files_label(state: &UiState) -> String {
    format!(
        "{} {}/{} {}",
        state.files.label,
        state.files.position,
        state.files.total,
        state.files.message
    )
}

fn build_bytes_label(state: &UiState) -> String {
    let position = format!("{}", HumanBytes(state.bytes.position));
    let total = format!("{}", HumanBytes(state.bytes.total));
    format!(
        "{} {}/{} {}",
        state.bytes.label,
        position,
        total,
        state.bytes.message
    )
}

fn truncate_head(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= width {
        return value.to_string();
    }
    if width <= 3 {
        return ".".repeat(width);
    }
    let keep = width - 3;
    let mut out: String = chars[..keep].iter().collect();
    out.push_str("...");
    out
}

fn truncate_tail(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= width {
        return value.to_string();
    }
    if width <= 3 {
        return ".".repeat(width);
    }
    let keep = width - 3;
    let suffix: String = chars[chars.len() - keep..].iter().collect();
    format!("...{}", suffix)
}

fn truncate_banner_line(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    value.chars().take(width).collect()
}

fn build_files_gauge(state: &UiState, area: Rect) -> Gauge<'static> {
    let ratio = if state.files.total == 0 {
        0.0
    } else {
        state.files.position as f64 / state.files.total as f64
    };
    let label = truncate_head(&build_files_label(state), area.width as usize);
    Gauge::default()
        .ratio(ratio.clamp(0.0, 1.0))
        .label(label)
        .gauge_style(Style::default().fg(Color::Cyan))
}

fn build_bytes_gauge(state: &UiState, area: Rect) -> Gauge<'static> {
    let ratio = if state.bytes.total == 0 {
        0.0
    } else {
        state.bytes.position as f64 / state.bytes.total as f64
    };
    let label = truncate_head(&build_bytes_label(state), area.width as usize);
    Gauge::default()
        .ratio(ratio.clamp(0.0, 1.0))
        .label(label)
        .gauge_style(Style::default().fg(Color::Green))
}

fn render_header_line(frame: &mut Frame, area: Rect, line: Line) {
    frame.render_widget(Paragraph::new(line).wrap(Wrap { trim: true }), area);
}

fn render_header(frame: &mut Frame, area: Rect, state: &UiState, spinner_index: usize) {
    let mut constraints = Vec::new();
    let line_count = area.height as usize;
    for _ in 0..line_count {
        constraints.push(Constraint::Length(1));
    }

    let lines = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    let spinner = SPINNER_FRAMES[spinner_index];

    let mut idx = 0;
    for banner_line in BANNER_LINES.iter() {
        if idx >= lines.len() {
            return;
        }
        let line = Line::from(truncate_banner_line(banner_line, lines[idx].width as usize));
        render_header_line(frame, lines[idx], line);
        idx += 1;
    }

    if idx < lines.len() {
        let line = build_scan_line(state, spinner, lines[idx]);
        render_header_line(frame, lines[idx], line);
        idx += 1;
    }

    if idx < lines.len() {
        let line = build_resource_line(state, spinner, lines[idx]);
        render_header_line(frame, lines[idx], line);
        idx += 1;
    }

    if idx < lines.len() {
        let gauge = build_files_gauge(state, lines[idx]);
        frame.render_widget(gauge, lines[idx]);
        idx += 1;
    }

    if idx < lines.len() {
        let gauge = build_bytes_gauge(state, lines[idx]);
        frame.render_widget(gauge, lines[idx]);
        idx += 1;
    }

    if idx < lines.len() {
        let line = build_current_line(state, spinner, lines[idx]);
        render_header_line(frame, lines[idx], line);
    }
}

fn format_log_line(line: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }

    let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
    if trimmed.chars().count() <= width {
        return trimmed.to_string();
    }

    if !trimmed.contains('/') && !trimmed.contains('\\') {
        return truncate_head(trimmed, width);
    }

    let mut tokens: Vec<String> = trimmed.split_whitespace().map(String::from).collect();
    if tokens.len() <= 1 {
        return ellipsize_middle(trimmed, width);
    }

    let mut total_len = tokens.len().saturating_sub(1);
    let mut token_lengths = Vec::with_capacity(tokens.len());
    for token in &tokens {
        let len = token.chars().count();
        token_lengths.push(len);
        total_len = total_len.saturating_add(len);
    }

    let mut extra = total_len.saturating_sub(width);
    for idx in 0..tokens.len() {
        if extra == 0 {
            break;
        }
        if !is_path_token(&tokens[idx]) {
            continue;
        }
        let current_len = token_lengths[idx];
        if current_len <= 12 {
            continue;
        }
        let target_len = (current_len.saturating_sub(extra)).max(12);
        if target_len < current_len {
            tokens[idx] = ellipsize_middle(&tokens[idx], target_len);
            extra = extra.saturating_sub(current_len - target_len);
            token_lengths[idx] = target_len;
        }
    }

    let mut joined = tokens.join(" ");
    if joined.chars().count() > width {
        joined = truncate_head(&joined, width);
    }

    joined
}

fn is_path_token(token: &str) -> bool {
    token.contains('/') || token.contains('\\')
}

fn ellipsize_middle(value: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= width {
        return value.to_string();
    }
    if width <= 3 {
        return ".".repeat(width);
    }

    // for paths, preserve the filename (part after last separator)
    if is_path_token(value) {
        if let Some(sep_pos) = value.rfind(['/', '\\']) {
            let filename = &value[sep_pos + 1..];
            let filename_len = filename.chars().count();

            // if filename fits with ellipsis, keep it intact
            if filename_len + 4 <= width {
                // ".../" + filename
                let prefix_space = width - filename_len - 4; // space for prefix before ".../"
                if prefix_space > 0 {
                    let prefix: String = chars[..prefix_space].iter().collect();
                    return format!("{}.../{}", prefix, filename);
                } else {
                    return format!(".../{}", filename);
                }
            }
        }
    }

    // fallback: even split between prefix and suffix
    let available = width - 3;
    let prefix_len = (available + 1) / 2;
    let suffix_len = available - prefix_len;
    let prefix: String = chars[..prefix_len].iter().collect();
    let suffix: String = chars[chars.len() - suffix_len..].iter().collect();
    format!("{prefix}...{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_head() {
        assert_eq!(truncate_head("hello", 10), "hello");
        assert_eq!(truncate_head("hello world", 5), "he...");
        assert_eq!(truncate_head("hello", 2), "..");
    }

    #[test]
    fn test_truncate_tail() {
        assert_eq!(truncate_tail("hello", 10), "hello");
        assert_eq!(truncate_tail("hello world", 5), "...ld");
        assert_eq!(truncate_tail("hello", 2), "..");
    }

    #[test]
    fn test_format_log_line_path() {
        let line = "info processing file /very/long/path/to/a/file_name.rs";
        let rendered = format_log_line(line, 40);
        assert!(rendered.contains("..."));
        assert!(rendered.contains("file_name.rs"));
    }
}
