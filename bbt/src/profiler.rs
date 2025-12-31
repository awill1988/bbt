//! Heap profiling support using dhat.
//!
//! Signal handling for Ctrl+C is managed by shutdown.rs which also
//! writes the heap profile snapshot on interrupt.

#[cfg(feature = "heap-profiling")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "heap-profiling")]
pub static PROFILER: once_cell::sync::Lazy<Arc<Mutex<Option<dhat::Profiler>>>> =
    once_cell::sync::Lazy::new(|| {
        eprintln!("heap profiling enabled - will write dhat-heap.json on exit or interrupt");
        Arc::new(Mutex::new(Some(dhat::Profiler::new_heap())))
    });
