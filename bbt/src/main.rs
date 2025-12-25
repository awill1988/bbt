use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod cmd;

use clap::Parser;

fn main() -> anyhow::Result<()> {
    let args = cmd::Cli::parse();
    cmd::run(args)
}
