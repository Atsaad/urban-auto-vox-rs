//! Step 4 — produce `translate.json` from per-OBJ vertex bounds.
//!
//! Implements the same contract as the Python
//! `create_translate_per_obj.calculate_per_obj_translate`: iterate OBJ
//! files, take the axis-aligned min/max of every `v x y z` line, and
//! write the `{global_bbox, per_file}` structure.
//!
//! OBJ parsing here is deliberately minimal — we never need faces,
//! normals, textures, or materials; vertex lines alone determine the
//! bounding box. This keeps the work proportional to the file size
//! rather than the mesh complexity.

use std::collections::BTreeMap;
use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Args;
use rayon::prelude::*;
use tracing::info;
use voxel_schema::translate::{GlobalBbox, PerFileBbox, TranslateFile};
use walkdir::WalkDir;

#[derive(Debug, Args)]
pub struct TranslateArgs {
    /// Directory containing `*.obj` files (non-recursive).
    #[arg(long, env = "PIPELINE_INPUT_DIR")]
    pub input_dir: PathBuf,

    /// Output path. Defaults to `<input_dir>/translate.json`.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub fn run(args: TranslateArgs) -> Result<()> {
    let out = args
        .output
        .clone()
        .unwrap_or_else(|| args.input_dir.join("translate.json"));

    let obj_paths: Vec<PathBuf> = list_obj_files(&args.input_dir)?;
    if obj_paths.is_empty() {
        anyhow::bail!("no *.obj files found in {}", args.input_dir.display());
    }
    info!(count = obj_paths.len(), "scanning OBJ files for bounding boxes");

    // Parallel bbox computation — each OBJ is independent.
    let per_file: Vec<(String, PerFileBbox)> = obj_paths
        .par_iter()
        .map(|p| bbox_for_obj(p))
        .collect::<Result<Vec<_>>>()?;

    let file = build_translate_file(per_file);
    file.save(&out)
        .with_context(|| format!("writing {}", out.display()))?;
    info!(path = %out.display(), files = file.per_file.len(), "wrote translate.json");
    Ok(())
}

pub fn build_translate_file(per_file_vec: Vec<(String, PerFileBbox)>) -> TranslateFile {
    let (mut gmin, mut gmax) = (
        [f64::INFINITY; 3],
        [f64::NEG_INFINITY; 3],
    );
    let mut per_file = BTreeMap::new();
    for (key, bbox) in per_file_vec {
        gmin[0] = gmin[0].min(bbox.xmin);
        gmin[1] = gmin[1].min(bbox.ymin);
        gmin[2] = gmin[2].min(bbox.zmin);
        gmax[0] = gmax[0].max(bbox.xmax);
        gmax[1] = gmax[1].max(bbox.ymax);
        gmax[2] = gmax[2].max(bbox.zmax);
        per_file.insert(key, bbox);
    }
    TranslateFile {
        global_bbox: GlobalBbox {
            json_featuretype: "translate_model".into(),
            xmin: gmin[0],
            xmax: gmax[0],
            ymin: gmin[1],
            ymax: gmax[1],
            zmin: gmin[2],
            zmax: gmax[2],
        },
        per_file,
    }
}

fn bbox_for_obj(path: &Path) -> Result<(String, PerFileBbox)> {
    let filename = path
        .file_name()
        .and_then(|s| s.to_str())
        .context("invalid OBJ filename")?
        .to_string();
    let stem = filename.strip_suffix(".obj").unwrap_or(&filename).to_string();

    let file = std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let reader = std::io::BufReader::new(file);

    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut count = 0u64;

    for line in reader.lines() {
        let line = line?;
        let bytes = line.as_bytes();
        // Only `v ` vertex lines — skip `vn`, `vt`, faces, groups, etc.
        if bytes.len() < 2 || bytes[0] != b'v' || bytes[1] != b' ' {
            continue;
        }
        let mut parts = line[2..].split_ascii_whitespace();
        let x: f64 = match parts.next().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let y: f64 = match parts.next().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let z: f64 = match parts.next().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        min[0] = min[0].min(x);
        min[1] = min[1].min(y);
        min[2] = min[2].min(z);
        max[0] = max[0].max(x);
        max[1] = max[1].max(y);
        max[2] = max[2].max(z);
        count += 1;
    }

    if count == 0 {
        anyhow::bail!("{}: contains no vertices", path.display());
    }
    Ok((stem, PerFileBbox::from_min_max(filename, min, max)))
}

pub fn list_obj_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in WalkDir::new(dir).max_depth(1).into_iter().flatten() {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.into_path();
        if p.extension().map(|e| e == "obj").unwrap_or(false) {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}
