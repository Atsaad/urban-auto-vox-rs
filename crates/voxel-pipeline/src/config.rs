//! End-to-end orchestration: translate → index → grid → voxelize.
//!
//! Produces the same side effects as running the four subcommands in
//! sequence, but with a single argument surface that matches the Python
//! pipeline's environment-variable contract.

use anyhow::Result;
use clap::Args;
use tracing::info;

use crate::grid::GridArgs;
use crate::index_build::IndexArgs;
use crate::processor::{self, VoxelizeArgs};
use crate::translate_build::TranslateArgs;

#[derive(Debug, Args)]
pub struct RunArgs {
    #[command(flatten)]
    pub voxelize: VoxelizeArgs,

    /// Use absolute paths in the generated `index.json`.
    #[arg(long)]
    pub index_absolute: bool,
}

pub async fn run_all(args: RunArgs) -> Result<()> {
    let input_dir = args.voxelize.input_dir.clone();

    info!("step 4 — translate.json");
    crate::translate_build::run(TranslateArgs {
        input_dir: input_dir.clone(),
        output: None,
    })?;

    info!("step 5 — index.json");
    crate::index_build::run(IndexArgs {
        input_dir: input_dir.clone(),
        output: None,
        srid: args.voxelize.srid,
        absolute: args.index_absolute,
    })?;

    info!("grid_mapping.json");
    crate::grid::run(GridArgs {
        translate_json: input_dir.join("translate.json"),
        target_voxel_size: args.voxelize.target_voxel_size,
        output: Some(input_dir.join("grid_mapping.json")),
    })?;

    info!("step 6 — voxelize + ingest");
    processor::run(args.voxelize).await
}
