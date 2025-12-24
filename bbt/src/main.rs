use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod cmd;

use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = cmd::Cli::parse();
    args.execute().await
}
