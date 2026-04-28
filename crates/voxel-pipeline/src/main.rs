//! `voxel-pipeline` — Rust rewrite of steps 4-6 of Urban-Auto-Vox.
//!
//! One static binary replaces four Python scripts + their numpy / pandas /
//! geopandas / sqlalchemy dependency stack. See `README.md` for the
//! migration map and container drop-in instructions.

use anyhow::Result;
use clap::{Parser, Subcommand};

mod config;
mod grid;
mod index_build;
mod processor;
mod sinks;
mod swap;
mod translate_build;
mod voxelizer;

/// Urban-Auto-Vox Rust pipeline (steps 4-6).
#[derive(Debug, Parser)]
#[command(name = "voxel-pipeline", version, about, long_about = None)]
struct Cli {
    /// Increase log verbosity (`-v` = debug, `-vv` = trace).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Step 4: build `translate.json` from per-OBJ vertex bounds.
    Translate(translate_build::TranslateArgs),

    /// Step 5: build `index.json` from per-surface sidecar JSONs.
    Index(index_build::IndexArgs),

    /// Between step 4 and step 6: build `grid_mapping.json` from a
    /// translate file and a target voxel size in metres.
    Grid(grid::GridArgs),

    /// Step 6: Y↔Z swap → cuda_voxelizer → ingest to CSV/PostGIS.
    Voxelize(processor::VoxelizeArgs),

    /// End-to-end: translate → index → grid → voxelize in one shot.
    Run(config::RunArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    match cli.command {
        Command::Translate(args) => translate_build::run(args),
        Command::Index(args) => index_build::run(args),
        Command::Grid(args) => grid::run(args),
        Command::Voxelize(args) => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(processor::run(args))
        }
        Command::Run(args) => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(config::run_all(args))
        }
    }
}

fn init_tracing(verbose: u8) {
    use tracing_subscriber::EnvFilter;
    let default = match verbose {
        0 => "info",
        1 => "voxel_pipeline=debug,voxel_binvox=debug,voxel_postgis=debug,info",
        _ => "trace",
    };
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .compact()
        .init();
}
