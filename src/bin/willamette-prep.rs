use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use project_willamette::gguf::repack::repack_embedding_q6k;
use project_willamette::memory::mmap::ModelMmap;

#[derive(Debug, Parser)]
#[command(
    name = "willamette-prep",
    version,
    about = "Build a low-memory Willamette artifact from a supported GGUF"
)]
struct Cli {
    /// Source GGUF containing a tied 2-D F16 token embedding.
    #[arg(long)]
    model: PathBuf,

    /// New GGUF path. Existing files are never overwritten.
    #[arg(long)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mmap = ModelMmap::open(&cli.model)
        .with_context(|| format!("opening model file: {}", cli.model.display()))?;
    let size = repack_embedding_q6k(mmap.as_bytes(), &cli.output)
        .with_context(|| format!("writing Q6_K model: {}", cli.output.display()))?;
    println!("Wrote {} bytes to {}", size, cli.output.display());
    Ok(())
}
