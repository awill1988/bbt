#[cfg(not(feature = "heap-profiling"))]
use mimalloc::MiMalloc;

#[cfg(not(feature = "heap-profiling"))]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[cfg(feature = "heap-profiling")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

mod chunk_budget;
mod cmd;
mod resource_monitor;
mod shutdown;
mod terminal_layout;
#[cfg(feature = "heap-profiling")]
mod profiler;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    // set up signal handler for graceful shutdown on Ctrl+C
    shutdown::setup_signal_handler();

    #[cfg(feature = "heap-profiling")]
    {
        // initialize profiler (signal handler in shutdown.rs handles snapshot on Ctrl+C)
        let _ = &*profiler::PROFILER;
    }

    let args = cmd::Cli::parse();
    cmd::run(args)
}
