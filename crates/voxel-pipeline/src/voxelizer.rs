//! Invoke `cuda_voxelizer` once per OBJ, with the per-file adaptive grid
//! size drawn from `grid_mapping.json`.
//!
//! We pool the work across rayon threads so multiple OBJs can drive the
//! GPU concurrently — this mirrors the Python `multiprocessing.Pool` but
//! uses a shared process (no fork, no pickle serialization of workers,
//! no per-task Python interpreter spin-up).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::sync::{atomic::{AtomicU64, Ordering}, Arc};
use tracing::{info, warn};

#[derive(Debug, Clone)]
pub struct VoxelizeBatchConfig {
    pub cuda_voxelizer: PathBuf,
    pub swap_dir: PathBuf,
    pub grid_mapping: BTreeMap<String, u32>,
    pub fallback_grid: u32,
    pub timeout: Duration,
    pub workers: usize,
    /// Incremented by 1 for each OBJ that finishes (ok or fail).
    pub progress: Arc<AtomicU64>,
}

#[derive(Debug, Clone)]
pub struct VoxelizeResult {
    pub obj_filename: String,
    pub grid: u32,
    pub status: VoxelizeStatus,
}

#[derive(Debug, Clone)]
pub enum VoxelizeStatus {
    Ok {
        #[allow(dead_code)] // diagnostic — actual relocation walks swap_dir
        binvox_path: PathBuf,
    },
    Failed {
        error: String,
    },
}

pub fn voxelize_batch(cfg: &VoxelizeBatchConfig) -> Result<Vec<VoxelizeResult>> {
    if !cfg.cuda_voxelizer.is_file() {
        bail!(
            "cuda_voxelizer not found at {}",
            cfg.cuda_voxelizer.display()
        );
    }
    let objs: Vec<PathBuf> = std::fs::read_dir(&cfg.swap_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("obj") | Some("stl") | Some("ply")
            )
        })
        .collect();
    if objs.is_empty() {
        bail!("no meshes to voxelize in {}", cfg.swap_dir.display());
    }

    info!(
        count = objs.len(),
        workers = cfg.workers,
        fallback_grid = cfg.fallback_grid,
        "dispatching CUDA voxelization"
    );

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.workers.max(1))
        .build()?;

    let results = pool.install(|| {
        objs.par_iter()
            .map(|obj| {
                let r = voxelize_one(cfg, obj);
                cfg.progress.fetch_add(1, Ordering::Relaxed);
                r
            })
            .collect::<Vec<_>>()
    });

    let (ok, fail): (Vec<_>, Vec<_>) = results
        .iter()
        .partition(|r| matches!(r.status, VoxelizeStatus::Ok { .. }));
    info!(
        ok = ok.len(),
        failed = fail.len(),
        "CUDA voxelization complete"
    );
    for r in &fail {
        if let VoxelizeStatus::Failed { error } = &r.status {
            warn!(file = %r.obj_filename, grid = r.grid, %error, "voxelization failed");
        }
    }
    if ok.is_empty() {
        bail!(
            "voxelization failed for all {} file(s) — check CUDA driver and OBJ integrity",
            fail.len()
        );
    }

    Ok(results)
}

fn voxelize_one(cfg: &VoxelizeBatchConfig, obj: &Path) -> VoxelizeResult {
    let filename = obj
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("<unknown>")
        .to_string();
    let grid = cfg
        .grid_mapping
        .get(&filename)
        .copied()
        .unwrap_or(cfg.fallback_grid);

    info!(file = %filename, grid, "voxelizing");

    let output = Command::new(&cfg.cuda_voxelizer)
        .current_dir(&cfg.swap_dir)
        .arg("-f")
        .arg(obj)
        .arg("-s")
        .arg(grid.to_string())
        .arg("-o")
        .arg("binvox")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|child| wait_with_timeout(child, cfg.timeout));

    match output {
        Ok(status) if status.success => {
            let bv = cfg
                .swap_dir
                .join(format!("{filename}_{grid}.binvox"));
            VoxelizeResult {
                obj_filename: filename,
                grid,
                status: VoxelizeStatus::Ok { binvox_path: bv },
            }
        }
        Ok(s) => VoxelizeResult {
            obj_filename: filename,
            grid,
            status: VoxelizeStatus::Failed {
                error: format!("cuda_voxelizer exited with code {:?}", s.code),
            },
        },
        Err(e) => VoxelizeResult {
            obj_filename: filename,
            grid,
            status: VoxelizeStatus::Failed {
                error: e.to_string(),
            },
        },
    }
}

struct WaitOutput {
    success: bool,
    code: Option<i32>,
}

fn wait_with_timeout(
    mut child: std::process::Child,
    timeout: Duration,
) -> std::io::Result<WaitOutput> {
    use std::time::Instant;
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait()? {
            Some(status) => {
                return Ok(WaitOutput {
                    success: status.success(),
                    code: status.code(),
                });
            }
            None => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!("cuda_voxelizer exceeded {timeout:?}"),
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

/// Move every produced `*.binvox` file out of `swap_dir` into `dest`,
/// leaving the swapped OBJ copies behind. Matches the Python
/// "move BINVOX back to input dir" step.
pub fn relocate_binvox(swap_dir: &Path, dest: &Path) -> Result<Vec<PathBuf>> {
    let mut moved = Vec::new();
    for entry in std::fs::read_dir(swap_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().map(|e| e == "binvox").unwrap_or(false) {
            let target = dest.join(path.file_name().unwrap());
            std::fs::rename(&path, &target)
                .or_else(|_| -> std::io::Result<()> {
                    std::fs::copy(&path, &target)?;
                    std::fs::remove_file(&path)?;
                    Ok(())
                })
                .with_context(|| format!("moving {} -> {}", path.display(), target.display()))?;
            moved.push(target);
        }
    }
    Ok(moved)
}
