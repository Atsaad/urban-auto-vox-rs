//! Grid mapping — `translate.json` + target voxel size → `grid_mapping.json`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Args;
use tracing::info;
use voxel_schema::grid_mapping::GridMappingFile;
use voxel_schema::translate::TranslateFile;

#[derive(Debug, Args)]
pub struct GridArgs {
    /// Path to `translate.json`.
    #[arg(long)]
    pub translate_json: PathBuf,

    /// Target voxel edge length in metres.
    #[arg(long, env = "PIPELINE_VOXEL_SIZE", default_value_t = 0.5)]
    pub target_voxel_size: f64,

    /// Output path. Defaults to `<translate_dir>/grid_mapping.json`.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub fn run(args: GridArgs) -> Result<()> {
    let translate = TranslateFile::load(&args.translate_json)
        .with_context(|| format!("loading {}", args.translate_json.display()))?;
    let mapping = GridMappingFile::from_translate(&translate, args.target_voxel_size);
    let out = args.output.clone().unwrap_or_else(|| {
        args.translate_json
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .join("grid_mapping.json")
    });
    mapping
        .save(&out)
        .with_context(|| format!("writing {}", out.display()))?;

    info!(
        path = %out.display(),
        files = mapping.total_files,
        target_voxel_size = args.target_voxel_size,
        "wrote grid_mapping.json"
    );
    for (g, count) in &mapping.grid_distribution {
        let pct = 100.0 * *count as f64 / mapping.total_files as f64;
        info!("  grid {g:>5}: {count} files ({pct:.1}%)");
    }
    Ok(())
}
