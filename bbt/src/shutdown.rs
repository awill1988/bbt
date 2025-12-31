//! Graceful shutdown handling for Ctrl+C signals.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Global shutdown flag that can be checked by any running task.
pub static SHUTDOWN_REQUESTED: once_cell::sync::Lazy<Arc<AtomicBool>> =
    once_cell::sync::Lazy::new(|| Arc::new(AtomicBool::new(false)));

/// Check if shutdown has been requested.
pub fn is_shutdown_requested() -> bool {
    SHUTDOWN_REQUESTED.load(Ordering::Relaxed)
}

/// Set up the Ctrl+C signal handler.
///
/// This should be called once at program startup. The handler will:
/// 1. Set the shutdown flag for graceful termination
/// 2. If heap-profiling is enabled, write the profiler snapshot
/// 3. On second Ctrl+C, force exit immediately
pub fn setup_signal_handler() {
    let first_signal = Arc::new(AtomicBool::new(true));

    ctrlc::set_handler(move || {
        if first_signal.swap(false, Ordering::Relaxed) {
            // first signal: request graceful shutdown
            eprintln!("\nreceived interrupt, shutting down gracefully...");
            SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);

            // if heap-profiling is enabled, write the snapshot
            #[cfg(feature = "heap-profiling")]
            {
                if let Ok(mut guard) = crate::profiler::PROFILER.lock() {
                    eprintln!("writing heap profile...");
                    guard.take(); // drops the profiler, writes dhat-heap.json
                }
            }
        } else {
            // second signal: force exit
            eprintln!("\nreceived second interrupt, forcing exit");
            std::process::exit(130); // 128 + SIGINT
        }
    })
    .expect("failed to set signal handler");
}
