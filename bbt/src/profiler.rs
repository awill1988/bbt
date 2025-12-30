#[cfg(feature = "heap-profiling")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "heap-profiling")]
pub static PROFILER: once_cell::sync::Lazy<Arc<Mutex<Option<dhat::Profiler>>>> =
    once_cell::sync::Lazy::new(|| {
        Arc::new(Mutex::new(Some(dhat::Profiler::new_heap())))
    });

#[cfg(feature = "heap-profiling")]
pub fn setup_signal_handler() {
    let profiler_clone = Arc::clone(&PROFILER);
    ctrlc::set_handler(move || {
        eprintln!("\nreceived interrupt, writing heap profile...");
        if let Ok(mut guard) = profiler_clone.lock() {
            guard.take(); // drops the profiler, writes dhat-heap.json
        }
        std::process::exit(130); // 128 + SIGINT
    })
    .expect("failed to set signal handler");

    eprintln!("heap profiling enabled - will write dhat-heap.json on exit or interrupt");
}
