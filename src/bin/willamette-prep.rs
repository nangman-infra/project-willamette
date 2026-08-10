use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use project_willamette::gguf::linker::{
    link_artifact, plan_artifact, ArtifactProfile, LinkPlan, TensorAction,
};
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

    /// Artifact policy to validate and apply.
    #[arg(long, default_value = "embedding-q6-k")]
    profile: ArtifactProfile,

    /// Validate and print the complete plan without writing an artifact.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
}

fn print_plan(plan: &LinkPlan, include_tensor_layout: bool) {
    println!("Profile:         {}", plan.profile);
    println!("Architecture:    {}", plan.architecture);
    println!("Source bytes:    {}", plan.source_bytes);
    println!("Output bytes:    {}", plan.output_bytes);
    println!("Alignment:       {}", plan.alignment);
    println!("Tensor count:    {}", plan.tensor_count);
    println!("Changed tensors: {}", plan.changes.len());
    for change in &plan.changes {
        println!(
            "  {}: {} -> {}, {} -> {} bytes, offset {} -> {}",
            change.name,
            change.source_type.name(),
            change.output_type.name(),
            change.source_bytes,
            change.output_bytes,
            change.source_offset,
            change.output_offset,
        );
    }
    if include_tensor_layout {
        println!("Tensor layout:");
        for tensor in &plan.tensors {
            let action = match tensor.action {
                TensorAction::Copy => "copy",
                TensorAction::QuantizeF16ToQ6K => "f16-to-q6-k",
            };
            println!(
                "  {}: {action}, {} -> {}, primary {} -> {}, slot {} -> {}, offset {} -> {}",
                tensor.name,
                tensor.source_type.name(),
                tensor.output_type.name(),
                tensor.source_primary_bytes,
                tensor.output_primary_bytes,
                tensor.source_slot_bytes,
                tensor.output_slot_bytes,
                tensor.source_offset,
                tensor.output_offset,
            );
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mmap = ModelMmap::open(&cli.model)
        .with_context(|| format!("opening model file: {}", cli.model.display()))?;
    if cli.dry_run {
        let plan = plan_artifact(mmap.as_bytes(), cli.profile)
            .with_context(|| format!("planning {} artifact", cli.profile))?;
        print_plan(&plan, true);
        println!("Dry run: no output written");
        return Ok(());
    }

    let report = link_artifact(mmap.as_bytes(), &cli.output, cli.profile)
        .with_context(|| format!("writing {} artifact: {}", cli.profile, cli.output.display()))?;
    print_plan(&report.plan, false);
    println!(
        "Wrote {} bytes to {}",
        report.plan.output_bytes,
        cli.output.display()
    );
    Ok(())
}
