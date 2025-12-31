//! Terminal layout with fixed header and scrolling log region.
//!
//! Provides a split terminal view:
//! - Fixed header region at top for progress bars (managed by indicatif)
//! - Scrolling log region below for streaming log output

use crossterm::{
    cursor::{MoveTo, Show},
    execute,
    terminal::{self, DisableLineWrap, EnableLineWrap},
    style::Print,
};
use std::io::{self, Write};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};

/// Messages sent to the log renderer
enum LogMessage {
    /// Log line to display in scroll region
    Line(String),
    /// Shutdown the renderer
    Shutdown,
}

/// Guard that manages terminal state and log rendering thread.
///
/// When dropped, restores terminal state and stops the renderer.
pub struct ScrollingLogGuard {
    sender: Sender<LogMessage>,
    renderer_handle: Option<JoinHandle<()>>,
    original_cursor_visible: bool,
}

impl ScrollingLogGuard {
    /// Get a sender that can be used with tracing's init_tracing
    pub fn log_sender(&self) -> Sender<String> {
        let layout_sender = self.sender.clone();
        let (tx, rx) = mpsc::channel::<String>();

        thread::spawn(move || {
            while let Ok(msg) = rx.recv() {
                if msg == "__bbt_log_close__" {
                    break;
                }
                let _ = layout_sender.send(LogMessage::Line(msg));
            }
        });

        tx
    }
}

impl Drop for ScrollingLogGuard {
    fn drop(&mut self) {
        // signal shutdown
        let _ = self.sender.send(LogMessage::Shutdown);

        // wait for renderer to finish
        if let Some(handle) = self.renderer_handle.take() {
            let _ = handle.join();
        }

        // restore terminal state
        let mut stdout = io::stdout();

        // reset scroll region to full terminal
        let _ = write!(stdout, "\x1b[r");

        // restore cursor visibility
        if self.original_cursor_visible {
            let _ = execute!(stdout, Show);
        }

        // re-enable line wrap
        let _ = execute!(stdout, EnableLineWrap);

        let _ = stdout.flush();
    }
}

/// Set up a scrolling log region below the progress bars.
///
/// Returns a guard that manages the terminal state. The guard provides
/// a log_sender() that can be passed to init_tracing.
///
/// The first `header_lines` of the terminal are reserved for progress bars
/// (managed by indicatif). Logs are displayed in the region below and scroll
/// independently.
pub fn setup_scrolling_logs(header_lines: u16) -> io::Result<ScrollingLogGuard> {
    let (width, height) = terminal::size()?;
    let header_lines = header_lines.min(height.saturating_sub(5));

    let (sender, receiver) = mpsc::channel();

    let mut stdout = io::stdout();

    // disable line wrap to prevent messing up the layout
    execute!(stdout, DisableLineWrap)?;

    // draw separator line
    let separator = "─".repeat(width as usize);
    execute!(
        stdout,
        MoveTo(0, header_lines),
        Print(&separator),
    )?;

    // set scroll region (ANSI escape sequence, 1-indexed)
    // region starts after header + separator line
    let scroll_top = header_lines + 2; // +1 for separator, +1 for 1-indexed
    let scroll_bottom = height;
    write!(stdout, "\x1b[{};{}r", scroll_top, scroll_bottom)?;

    // move cursor to start of scroll region
    execute!(stdout, MoveTo(0, header_lines + 1))?;

    stdout.flush()?;

    // spawn renderer thread
    let renderer_handle = thread::spawn(move || {
        render_logs(receiver, width, height, header_lines);
    });

    Ok(ScrollingLogGuard {
        sender,
        renderer_handle: Some(renderer_handle),
        original_cursor_visible: true,
    })
}

/// Render logs in the scroll region
fn render_logs(receiver: Receiver<LogMessage>, width: u16, height: u16, _header_lines: u16) {
    let mut stdout = io::stdout();

    while let Ok(msg) = receiver.recv() {
        match msg {
            LogMessage::Line(line) => {
                // move to bottom of scroll region
                let _ = execute!(stdout, MoveTo(0, height - 1));

                // print with newline to trigger scroll
                let truncated = truncate_line(&line, width as usize);
                let _ = writeln!(stdout, "{}", truncated);
                let _ = stdout.flush();
            }
            LogMessage::Shutdown => {
                break;
            }
        }
    }

    // reset scroll region before exiting
    let _ = write!(stdout, "\x1b[r");
    let _ = stdout.flush();
}

/// Truncate a line to fit within the terminal width
fn truncate_line(s: &str, width: usize) -> String {
    // remove any existing newlines
    let s = s.trim_end_matches('\n').trim_end_matches('\r');

    if s.chars().count() <= width {
        s.to_string()
    } else {
        let mut result: String = s.chars().take(width.saturating_sub(1)).collect();
        result.push('…');
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_line() {
        assert_eq!(truncate_line("hello", 10), "hello");
        assert_eq!(truncate_line("hello world", 5), "hell…");
        assert_eq!(truncate_line("hello\n", 10), "hello");
    }
}
