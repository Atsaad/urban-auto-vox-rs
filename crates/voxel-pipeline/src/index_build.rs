//! Step 5 — produce `index.json` from per-surface sidecar JSONs.
//!
//! Input: one `<name>.json` per `<name>.obj`, emitted by
//! `rustcitygml2obj --add-json`. Output: a single `index.json` with one
//! entry per OBJ, plus a `CRS` key. Matches the contract consumed by the
//! voxelizer's semantic lookup.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use tracing::{info, warn};
use voxel_schema::index::{namespaced_tag, Crs, IndexEntry, IndexFile, OneOrMany};
use voxel_schema::surface::SurfaceSidecar;
use walkdir::WalkDir;

#[derive(Debug, Args)]
pub struct IndexArgs {
    /// Directory containing per-surface `*.json` files and their `*.obj`
    /// siblings (non-recursive).
    #[arg(long, env = "PIPELINE_INPUT_DIR")]
    pub input_dir: PathBuf,

    /// Output path. Defaults to `<input_dir>/index.json`.
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// EPSG code for the CRS entry.
    #[arg(long, env = "PIPELINE_DB_SRID", default_value_t = 25832)]
    pub srid: u32,

    /// Use absolute paths for OBJ keys. Default: basenames only.
    #[arg(long)]
    pub absolute: bool,
}

pub fn run(args: IndexArgs) -> Result<()> {
    let out = args
        .output
        .clone()
        .unwrap_or_else(|| args.input_dir.join("index.json"));

    let mut index = IndexFile::new(Crs::epsg(args.srid));
    let mut processed = 0usize;
    let mut skipped = 0usize;

    for entry in WalkDir::new(&args.input_dir).max_depth(1).into_iter().flatten() {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.into_path();
        // Skip non-JSON, skip the two reserved filenames.
        if path.extension().map(|e| e != "json").unwrap_or(true) {
            continue;
        }
        let fname = match path.file_name().and_then(|s| s.to_str()) {
            Some(f) => f.to_ascii_lowercase(),
            None => continue,
        };
        if fname == "index.json" || fname == "translate.json" || fname == "grid_mapping.json" {
            continue;
        }
        let obj_path = matching_obj(&path);
        let obj_key = match obj_path {
            Some(p) if args.absolute => p
                .canonicalize()
                .unwrap_or(p.clone())
                .to_string_lossy()
                .into_owned(),
            Some(p) => p
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            None => {
                warn!(json = %path.display(), "no matching .obj found, skipping");
                skipped += 1;
                continue;
            }
        };
        match parse_sidecar(&path) {
            Ok(entry) => {
                index.insert_entry(obj_key, entry);
                processed += 1;
            }
            Err(e) => {
                warn!(json = %path.display(), error = %e, "failed to parse sidecar");
                skipped += 1;
            }
        }
    }

    if processed == 0 {
        anyhow::bail!("no parseable sidecar JSON files found in {}", args.input_dir.display());
    }
    index
        .save(&out)
        .with_context(|| format!("writing {}", out.display()))?;
    info!(
        path = %out.display(),
        processed,
        skipped,
        "wrote index.json"
    );
    Ok(())
}

fn matching_obj(json_path: &Path) -> Option<PathBuf> {
    let dir = json_path.parent()?;
    let stem = json_path.file_stem()?.to_str()?;
    let candidate = dir.join(format!("{stem}.obj"));
    candidate.exists().then_some(candidate)
}

fn parse_sidecar(path: &Path) -> Result<IndexEntry> {
    let sidecar = SurfaceSidecar::load(path)?;
    let role = sidecar
        .thematic_role
        .clone()
        .unwrap_or_else(|| "WallSurface".into());
    let building_id = sidecar
        .building_id
        .clone()
        .unwrap_or_else(|| "UNKNOWN".into());
    let polygon_id = sidecar
        .polygon_gml_id
        .clone()
        .unwrap_or_default();
    Ok(IndexEntry {
        tag: namespaced_tag(&role),
        parent_id: OneOrMany::Many(vec![building_id]),
        gml_id: OneOrMany::One(polygon_id),
        class: sidecar.class,
    })
}
