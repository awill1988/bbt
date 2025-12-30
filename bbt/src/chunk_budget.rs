//! Weighted semaphore for GPU memory management
//!
//! Mathematical model:
//! - small files (<= threshold): share total_bytes_budget concurrently
//! - large files (> threshold): exclusive access, one at a time, nothing else runs

use std::sync::{Condvar, Mutex};

/// Weighted semaphore that tracks chunk and byte capacity
///
/// Small files share the total budget concurrently.
/// Large files (> threshold) run exclusively - no other files process concurrently.
pub struct ChunkBudget {
    state: Mutex<BudgetState>,
    condvar: Condvar,
    max_chunks: usize,
    /// total byte budget for concurrent processing
    total_bytes_budget: usize,
    /// bytes threshold for exclusive access
    large_file_threshold: usize,
}

struct BudgetState {
    /// current chunks in flight
    chunks_in_flight: usize,
    /// current bytes in flight
    bytes_in_flight: usize,
    /// is a large file currently running exclusively?
    large_file_running: bool,
}

impl ChunkBudget {
    /// Create a new chunk budget
    ///
    /// # Arguments
    /// * `max_chunks` - Maximum chunks allowed in flight
    /// * `total_bytes_budget` - Total byte budget for concurrent processing
    /// * `large_file_threshold` - Files larger than this get exclusive GPU access
    pub fn new(
        max_chunks: usize,
        total_bytes_budget: usize,
        large_file_threshold: usize,
    ) -> Self {
        Self {
            state: Mutex::new(BudgetState {
                chunks_in_flight: 0,
                bytes_in_flight: 0,
                large_file_running: false,
            }),
            condvar: Condvar::new(),
            max_chunks,
            total_bytes_budget,
            large_file_threshold,
        }
    }

    /// Check if this is a large file requiring exclusive access
    fn is_large_file(&self, bytes: usize) -> bool {
        bytes > self.large_file_threshold
    }

    /// Get the large file threshold
    pub fn large_file_threshold(&self) -> usize {
        self.large_file_threshold
    }

    /// Get the total bytes budget
    pub fn total_bytes_budget(&self) -> usize {
        self.total_bytes_budget
    }

    /// Acquire capacity, blocking if necessary
    ///
    /// - Large files (> threshold): wait for exclusive access, block all others
    /// - Small files: share budget up to total_bytes_budget
    pub fn acquire(&self, chunks: usize, bytes: usize) -> BudgetGuard<'_> {
        let mut state = self.state.lock().unwrap();
        let is_large = self.is_large_file(bytes);

        if is_large {
            // large file: wait until nothing is in flight, then run exclusively
            while state.chunks_in_flight > 0 || state.large_file_running {
                state = self.condvar.wait(state).unwrap();
            }
            state.large_file_running = true;
        } else {
            // small file: wait if large file running OR would exceed total budget
            while state.large_file_running
                || (state.chunks_in_flight + chunks > self.max_chunks)
                || (state.bytes_in_flight + bytes > self.total_bytes_budget)
            {
                state = self.condvar.wait(state).unwrap();
            }
        }

        state.chunks_in_flight += chunks;
        state.bytes_in_flight += bytes;

        BudgetGuard {
            budget: self,
            chunks,
            bytes,
            is_large,
        }
    }

    /// Try to acquire capacity without blocking
    #[allow(dead_code)]
    pub fn try_acquire(&self, chunks: usize, bytes: usize) -> Option<BudgetGuard<'_>> {
        let mut state = self.state.lock().unwrap();
        let is_large = self.is_large_file(bytes);

        if is_large {
            // large file needs exclusive access
            if state.chunks_in_flight > 0 || state.large_file_running {
                return None;
            }
            state.large_file_running = true;
        } else {
            // small file checks limits
            if state.large_file_running
                || state.chunks_in_flight + chunks > self.max_chunks
                || state.bytes_in_flight + bytes > self.total_bytes_budget
            {
                return None;
            }
        }

        state.chunks_in_flight += chunks;
        state.bytes_in_flight += bytes;

        Some(BudgetGuard {
            budget: self,
            chunks,
            bytes,
            is_large,
        })
    }

    /// Release capacity (called by BudgetGuard on drop)
    fn release(&self, chunks: usize, bytes: usize, is_large: bool) {
        let mut state = self.state.lock().unwrap();
        state.chunks_in_flight = state.chunks_in_flight.saturating_sub(chunks);
        state.bytes_in_flight = state.bytes_in_flight.saturating_sub(bytes);
        if is_large {
            state.large_file_running = false;
        }
        drop(state);
        self.condvar.notify_all();
    }

    /// Get current chunks in flight
    pub fn chunks_in_flight(&self) -> usize {
        self.state.lock().unwrap().chunks_in_flight
    }

    /// Get current bytes in flight
    pub fn bytes_in_flight(&self) -> usize {
        self.state.lock().unwrap().bytes_in_flight
    }

    /// Check if a large file is currently running exclusively
    pub fn large_file_running(&self) -> bool {
        self.state.lock().unwrap().large_file_running
    }
}

/// RAII guard that releases chunk capacity when dropped
pub struct BudgetGuard<'a> {
    budget: &'a ChunkBudget,
    chunks: usize,
    bytes: usize,
    is_large: bool,
}

impl Drop for BudgetGuard<'_> {
    fn drop(&mut self) {
        self.budget.release(self.chunks, self.bytes, self.is_large);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    const TEST_THRESHOLD: usize = 1024 * 1024; // 1 MiB for tests

    #[test]
    fn test_basic_acquire_release() {
        // 100 chunks, 10 KiB total budget
        let budget = ChunkBudget::new(100, 10 * 1024, TEST_THRESHOLD);

        {
            let _guard = budget.acquire(50, 1024);
            assert_eq!(budget.chunks_in_flight(), 50);
            assert_eq!(budget.bytes_in_flight(), 1024);
        }

        assert_eq!(budget.chunks_in_flight(), 0);
        assert_eq!(budget.bytes_in_flight(), 0);
    }

    #[test]
    fn test_small_files_share_budget() {
        // 1000 chunks, 8 KiB total budget
        let budget = Arc::new(ChunkBudget::new(1000, 8 * 1024, TEST_THRESHOLD));
        let budget2 = Arc::clone(&budget);

        // first small file: 6 KiB
        let guard = budget.acquire(10, 6 * 1024);
        assert_eq!(budget.bytes_in_flight(), 6 * 1024);

        // second small file: 4 KiB would exceed 8 KiB total, must wait
        let handle = thread::spawn(move || {
            let _guard = budget2.acquire(5, 4 * 1024);
            assert_eq!(budget2.bytes_in_flight(), 4 * 1024);
        });

        thread::sleep(Duration::from_millis(50));
        drop(guard);
        handle.join().unwrap();
    }

    #[test]
    fn test_large_file_gets_exclusive_access() {
        // 1000 chunks, 16 MiB total budget, 1 MiB threshold
        let budget = Arc::new(ChunkBudget::new(1000, 16 * 1024 * 1024, TEST_THRESHOLD));
        let budget2 = Arc::clone(&budget);

        // large file (> 1 MiB) gets exclusive access
        let large_guard = budget.acquire(100, 2 * 1024 * 1024);
        assert!(budget.large_file_running());
        assert_eq!(budget.bytes_in_flight(), 2 * 1024 * 1024);

        // small file must wait for large file to complete
        let handle = thread::spawn(move || {
            let _guard = budget2.acquire(5, 1024);
            assert!(!budget2.large_file_running());
            assert_eq!(budget2.bytes_in_flight(), 1024);
        });

        thread::sleep(Duration::from_millis(50));
        drop(large_guard);
        handle.join().unwrap();
    }

    #[test]
    fn test_large_file_waits_for_drain() {
        // 1000 chunks, 16 MiB total budget, 1 MiB threshold
        let budget = Arc::new(ChunkBudget::new(1000, 16 * 1024 * 1024, TEST_THRESHOLD));
        let budget2 = Arc::clone(&budget);

        // small file starts first
        let small_guard = budget.acquire(10, 100 * 1024); // 100 KiB
        assert!(!budget.large_file_running());

        // large file must wait for small file to drain
        let handle = thread::spawn(move || {
            let _guard = budget2.acquire(100, 2 * 1024 * 1024); // 2 MiB
            assert!(budget2.large_file_running());
        });

        thread::sleep(Duration::from_millis(50));
        // large file should still be waiting
        assert!(!budget.large_file_running());

        drop(small_guard);
        handle.join().unwrap();
    }

    #[test]
    fn test_total_budget_limit() {
        // 1000 chunks, 100 KiB total budget, 1 MiB threshold
        let budget = ChunkBudget::new(1000, 100 * 1024, TEST_THRESHOLD);

        assert_eq!(budget.total_bytes_budget(), 100 * 1024);
        assert_eq!(budget.large_file_threshold(), TEST_THRESHOLD);
    }
}
