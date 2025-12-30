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
#[cfg(feature = "heap-profiling")]
mod profiler;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "heap-profiling")]
    {
        // initialize profiler and set up signal handler
        let _ = &*profiler::PROFILER;
        profiler::setup_signal_handler();
    }

    let args = cmd::Cli::parse();
    cmd::run(args)
}
