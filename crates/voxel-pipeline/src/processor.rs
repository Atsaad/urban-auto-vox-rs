//! Step 6 — the voxel ingestion core.
//!
//! For each `*.binvox`:
//!   1. Parse the header (mmap-backed).
//!   2. Adjust the `translate` vector from the per-file bbox so voxel
//!      centres land on the source CRS coordinates — the Python reference
//!      uses `center - scale/2` exactly. When no per-file bbox is
//!      available we fall back to the binvox header's own translate
//!      (Y↔Z un-swapped, matching the Python fallback).
//!   3. Iterate occupied voxels with [`voxel_binvox::OccupiedIter`] and
//!      emit one row per occupancy.
//!   4. Dispatch rows to the configured sinks (CSV, PostGIS, or both).
//!
//! Rayon parallelises over binvox files for the CPU path; the PostGIS
//! sink is single-writer-by-design (one COPY stream), so rows from rayon
//! workers funnel through a bounded `mpsc` into the async writer task.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Args, ValueEnum};
use rayon::prelude::*;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn};

use voxel_binvox::BinvoxFile;
use voxel_postgis::{
    apply_schema, connect, schema::upsert_object_and_class, PgConnectionConfig, VoxelCopyWriter,
};
use voxel_schema::grid_mapping::GridMappingFile;
use voxel_schema::index::IndexFile;
use voxel_schema::surface::{ResolvedIds, SurfaceSidecar};
use voxel_schema::translate::{PerFileBbox, TranslateFile};

use crate::sinks::{CsvSink, VoxelPayload};
use crate::swap;
use crate::voxelizer::{relocate_binvox, voxelize_batch, VoxelizeBatchConfig, VoxelizeStatus};
use crate::voxid;

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    Csv,
    Postgis,
    Both,
}

impl OutputFormat {
    pub fn wants_csv(self) -> bool {
        matches!(self, Self::Csv | Self::Both)
    }
    pub fn wants_postgis(self) -> bool {
        matches!(self, Self::Postgis | Self::Both)
    }
}

#[derive(Debug, Args, Clone)]
pub struct VoxelizeArgs {
    /// Directory with OBJ files + sidecar JSONs.
    #[arg(long, env = "PIPELINE_INPUT_DIR")]
    pub input_dir: PathBuf,

    /// Path to `cuda_voxelizer`.
    #[arg(long, env = "PIPELINE_CUDA_VOXELIZER")]
    pub cuda_voxelizer: PathBuf,

    /// Target voxel size in metres. Used when `grid_mapping.json` needs
    /// to be built on the fly.
    #[arg(long, env = "PIPELINE_VOXEL_SIZE", default_value_t = 0.5)]
    pub target_voxel_size: f64,

    /// EPSG code to tag the voxel geometries with.
    #[arg(long, env = "PIPELINE_DB_SRID", default_value_t = 25832)]
    pub srid: u32,

    /// Number of parallel voxelization workers.
    #[arg(long, env = "PIPELINE_NUM_WORKERS", default_value_t = 8)]
    pub workers: usize,

    /// Per-file voxelization timeout in seconds.
    #[arg(long, env = "VOXELIZE_TIMEOUT", default_value_t = 300)]
    pub voxelize_timeout_secs: u64,

    /// Output destination(s).
    #[arg(long, env = "PIPELINE_OUTPUT_FORMAT", value_enum, default_value_t = OutputFormat::Csv)]
    pub output_format: OutputFormat,

    /// Where to write `voxels_output.csv`. Defaults to `<input_dir>`.
    #[arg(long)]
    pub output_csv: Option<PathBuf>,

    /// COPY BINARY flush threshold in bytes.
    #[arg(long, env = "PIPELINE_DB_BATCH_BYTES", default_value_t = 8 * 1024 * 1024)]
    pub db_flush_bytes: usize,

    #[command(flatten)]
    pub db: PostgisConnArgs,
}

#[derive(Debug, Args, Clone)]
pub struct PostgisConnArgs {
    #[arg(long, env = "PIPELINE_DB_HOST", default_value = "")]
    pub db_host: String,
    #[arg(long, env = "PIPELINE_DB_PORT", default_value_t = 5432)]
    pub db_port: u16,
    #[arg(long, env = "PIPELINE_DB_NAME", default_value = "")]
    pub db_name: String,
    #[arg(long, env = "PIPELINE_DB_USERNAME", default_value = "")]
    pub db_username: String,
    #[arg(long, env = "PIPELINE_DB_PASSWORD", default_value = "")]
    pub db_password: String,
}

impl PostgisConnArgs {
    fn to_conn(&self) -> Option<PgConnectionConfig> {
        if self.db_host.is_empty() || self.db_name.is_empty() || self.db_username.is_empty() {
            return None;
        }
        Some(PgConnectionConfig {
            host: self.db_host.clone(),
            port: self.db_port,
            database: self.db_name.clone(),
            user: self.db_username.clone(),
            password: self.db_password.clone(),
            connect_timeout: Duration::from_secs(10),
        })
    }
}

pub async fn run(args: VoxelizeArgs) -> Result<()> {
    let input_dir = args.input_dir.clone();
    let csv_out = args
        .output_csv
        .clone()
        .unwrap_or_else(|| input_dir.join("voxels_output.csv"));

    info!(
        dir = %input_dir.display(),
        voxel_size = args.target_voxel_size,
        srid = args.srid,
        format = ?args.output_format,
        "voxel-pipeline starting"
    );

    // ---- Pre-step: ensure translate.json + index.json + grid_mapping.json
    // exist, generating them on the fly when missing.
    let translate_path = input_dir.join("translate.json");
    let translate = match TranslateFile::load(&translate_path) {
        Ok(t) => t,
        Err(_) => {
            info!("generating translate.json");
            let paths = crate::translate_build::list_obj_files(&input_dir)?;
            let per_file = paths
                .par_iter()
                .map(|p| -> Result<(String, PerFileBbox)> {
                    let stem = p
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                        .unwrap_or_default();
                    let name = p
                        .file_name()
                        .and_then(|s| s.to_str())
                        .map(|s| s.to_string())
                        .unwrap_or_default();
                    let (min, max) = obj_bbox(p)?;
                    Ok((stem, PerFileBbox::from_min_max(name, min, max)))
                })
                .collect::<Result<Vec<_>>>()?;
            let file = crate::translate_build::build_translate_file(per_file);
            file.save(&translate_path)?;
            file
        }
    };

    let grid_path = input_dir.join("grid_mapping.json");
    let grid = GridMappingFile::load(&grid_path).unwrap_or_else(|_| {
        let g = GridMappingFile::from_translate(&translate, args.target_voxel_size);
        if let Err(e) = g.save(&grid_path) {
            warn!(error = %e, "failed to write grid_mapping.json; continuing in-memory");
        }
        g
    });
    let fallback_grid = grid
        .grid_mapping
        .values()
        .copied()
        .max()
        .unwrap_or(256);

    // index.json is optional but strongly recommended for FULL mode.
    let index_path = input_dir.join("index.json");
    let index = IndexFile::load(&index_path).ok();
    if index.is_none() {
        warn!("index.json not found — object_type will fall back to 'Unknown'");
    }

    // ---- Step 6a: Y<->Z swap into scratch dir.
    let swap_dir = input_dir.join("_temp_swapped");
    info!("swapping Y↔Z axes into scratch dir");
    let _swapped = swap::swap_dir(&input_dir, &swap_dir)?;

    // ---- Step 6b: cuda_voxelizer.
    let total_objs = std::fs::read_dir(&swap_dir)
        .map(|rd| rd.filter_map(|e| e.ok()).filter(|e| {
            matches!(e.path().extension().and_then(|x| x.to_str()), Some("obj"|"stl"|"ply"))
        }).count())
        .unwrap_or(0) as u64;
    let vox_progress = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let vox_cfg = VoxelizeBatchConfig {
        cuda_voxelizer: args.cuda_voxelizer.clone(),
        swap_dir: swap_dir.clone(),
        grid_mapping: grid.grid_mapping.clone(),
        fallback_grid,
        timeout: Duration::from_secs(args.voxelize_timeout_secs),
        workers: args.workers,
        progress: Arc::clone(&vox_progress),
    };

    // Periodic progress ticker for the CUDA voxelization phase.
    let vox_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let vox_stop2 = Arc::clone(&vox_stop);
    let vox_prog2 = Arc::clone(&vox_progress);
    let vox_ticker: tokio::task::JoinHandle<()> = tokio::spawn(async move {
        let started = std::time::Instant::now();
        loop {
            tokio::time::sleep(Duration::from_secs(10)).await;
            if vox_stop2.load(std::sync::atomic::Ordering::Relaxed) { break; }
            let done = vox_prog2.load(std::sync::atomic::Ordering::Relaxed);
            info!(elapsed_s = started.elapsed().as_secs(), files_done = done, files_total = total_objs, "CUDA voxelization progress");
        }
    });

    let vox_results =
        tokio::task::spawn_blocking(move || voxelize_batch(&vox_cfg)).await??;

    vox_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = vox_ticker.await;

    // Move successful binvox files back to the input dir, then clean up.
    let binvoxes = relocate_binvox(&swap_dir, &input_dir)?;
    // Leave swapped OBJs behind but clear the scratch dir afterwards.
    let _ = std::fs::remove_dir_all(&swap_dir);

    let ok_count = vox_results
        .iter()
        .filter(|r| matches!(r.status, VoxelizeStatus::Ok { .. }))
        .count();
    info!(
        binvox = binvoxes.len(),
        ok = ok_count,
        total = vox_results.len(),
        "voxelization phase complete"
    );

    // ---- Step 6c: ingest rows.
    ingest(
        &args,
        &input_dir,
        &csv_out,
        &binvoxes,
        &translate,
        index.as_ref(),
    )
    .await?;

    Ok(())
}

async fn ingest(
    args: &VoxelizeArgs,
    input_dir: &Path,
    csv_out: &Path,
    binvoxes: &[PathBuf],
    translate: &TranslateFile,
    index: Option<&IndexFile>,
) -> Result<()> {
    // ---- CSV sink (sync, shared via Arc).
    let csv_sink = if args.output_format.wants_csv() {
        Some(Arc::new(CsvSink::create(csv_out)?))
    } else {
        None
    };

    // ---- Optional PostGIS sink with a writer task.
    let mut pg_handle: Option<JoinHandle<Result<u64>>> = None;
    let mut pg_tx: Option<mpsc::Sender<RowMsg>> = None;
    let copied_total = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let pg_client = if args.output_format.wants_postgis() {
        match args.db.to_conn() {
            Some(cfg) => {
                let client = connect(&cfg).await?;
                apply_schema(&client, args.srid).await?;
                Some(Arc::new(Mutex::new(client)))
            }
            None => {
                warn!("postgis output requested but DB credentials are incomplete — skipping");
                None
            }
        }
    } else {
        None
    };

    if let Some(client_arc) = pg_client.clone() {
        let (tx, mut rx) = mpsc::channel::<RowMsg>(4096);
        pg_tx = Some(tx);
        let flush_bytes = args.db_flush_bytes;
        let copied_counter = Arc::clone(&copied_total);
        // PostgreSQL forbids regular queries while a COPY stream is
        // active on the same connection, so open a dedicated connection
        // for the object/class upserts.
        let upsert_cfg = args.db.to_conn().expect("already validated");
        let upsert_client = connect(&upsert_cfg).await?;
        let handle: JoinHandle<Result<u64>> = tokio::spawn(async move {
            let client = client_arc.lock().await;
            let mut writer = VoxelCopyWriter::begin(&client, flush_bytes).await?;
            let mut copied_batch: u64 = 0;
            while let Some(msg) = rx.recv().await {
                match msg {
                    RowMsg::Row(row) => {
                        let vr = voxel_postgis::VoxelRow {
                            voxel_position: row.voxel_position,
                            x: row.x,
                            y: row.y,
                            z: row.z,
                            srid: row.srid,
                            element_gmlid: row.element_gmlid,
                            surface_gmlid: row.surface_gmlid,
                            building_gmlid: row.building_gmlid,
                        };
                        writer.write_row(&vr).await?;
                        copied_batch += 1;
                        if copied_batch >= 4096 {
                            copied_counter.fetch_add(copied_batch, std::sync::atomic::Ordering::Relaxed);
                            copied_batch = 0;
                        }
                    }
                    RowMsg::Object {
                        element_gmlid,
                        surface_gmlid,
                        building_gmlid,
                        object_type,
                    } => {
                        upsert_object_and_class(
                            &upsert_client,
                            &element_gmlid,
                            &surface_gmlid,
                            &building_gmlid,
                            &object_type,
                        )
                        .await?;
                    }
                }
            }
            if copied_batch > 0 {
                copied_counter.fetch_add(copied_batch, std::sync::atomic::Ordering::Relaxed);
            }
            let n = writer.finish().await?;
            Ok(n)
        });
        pg_handle = Some(handle);
    }

    // ---- CPU-parallel ingestion of binvox files.
    let sent_total = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let files_done = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let total_files = binvoxes.len() as u64;

    // Keep live feedback going during long ingest runs and while COPY drains.
    let progress_stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let progress_sent = Arc::clone(&sent_total);
    let progress_files = Arc::clone(&files_done);
    let progress_stop_flag = Arc::clone(&progress_stop);
    let progress_copied = Arc::clone(&copied_total);
    let progress_task: JoinHandle<()> = tokio::spawn(async move {
        let started = std::time::Instant::now();
        while !progress_stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_secs(10)).await;
            if progress_stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            info!(
                elapsed_s = started.elapsed().as_secs(),
                files_done = progress_files.load(std::sync::atomic::Ordering::Relaxed),
                files_total = total_files,
                voxels_emitted = progress_sent.load(std::sync::atomic::Ordering::Relaxed),
                voxels_copied = progress_copied.load(std::sync::atomic::Ordering::Relaxed),
                "live ingest progress"
            );
        }
    });

    let rayon_result: Result<()> = tokio::task::block_in_place(|| -> Result<()> {
        binvoxes
            .par_iter()
            .try_for_each(|bv_path| -> Result<()> {
                info!(file = %bv_path.display(), "processing binvox");
                let file_stats = process_single_binvox(
                    bv_path,
                    input_dir,
                    args.srid,
                    translate,
                    index,
                    csv_sink.as_deref(),
                    pg_tx.as_ref(),
                    &sent_total,
                )?;
                let done = files_done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                info!(
                    done,
                    total = total_files,
                    file = %bv_path.display(),
                    voxels = file_stats,
                    "binvox file processed"
                );
                Ok(())
            })
    });

    // Close the PostGIS channel so the writer task drains + finishes.
    drop(pg_tx);

    if let Some(arc) = csv_sink {
        match Arc::try_unwrap(arc) {
            Ok(sink) => sink.finish()?,
            // Parallel workers are done here, so this branch is unreachable
            // in practice; keep it defensive rather than panicking.
            Err(_still_shared) => warn!("CSV sink still shared at end of ingest; skipping flush"),
        }
    }
    if let Some(handle) = pg_handle {
        info!("waiting for PostGIS COPY drain + commit");
        match handle.await {
            Ok(Ok(rows)) => info!(rows, "COPY voxel committed"),
            Ok(Err(e)) => warn!(error = %e, "COPY voxel failed"),
            Err(e) => {
                progress_stop.store(true, std::sync::atomic::Ordering::Relaxed);
                let _ = progress_task.await;
                return Err(e.into());
            }
        }
    }
    progress_stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let _ = progress_task.await;

    rayon_result?;

    info!(
        voxels = sent_total.load(std::sync::atomic::Ordering::Relaxed),
        "ingestion complete"
    );
    Ok(())
}

#[derive(Debug)]
enum RowMsg {
    Row(OwnedVoxelRow),
    Object {
        element_gmlid: String,
        surface_gmlid: String,
        building_gmlid: String,
        object_type: String,
    },
}

#[derive(Debug)]
struct OwnedVoxelRow {
    voxel_position: i64,
    x: f64,
    y: f64,
    z: f64,
    srid: u32,
    element_gmlid: String,
    surface_gmlid: String,
    building_gmlid: String,
}

fn process_single_binvox(
    bv_path: &Path,
    input_dir: &Path,
    srid: u32,
    translate: &TranslateFile,
    index: Option<&IndexFile>,
    csv: Option<&CsvSink>,
    pg_tx: Option<&mpsc::Sender<RowMsg>>,
    emitted_counter: &std::sync::atomic::AtomicU64,
) -> Result<u64> {
    let bv = BinvoxFile::open(bv_path)
        .with_context(|| format!("opening {}", bv_path.display()))?;
    let header = *bv.header();

    // Extract grid size from filename: `<stem>.obj_<grid>.binvox`.
    let fname = bv_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    let grid = parse_grid_suffix(fname).unwrap_or(header.dims[0]);
    let obj_key = fname
        .strip_suffix(&format!("_{grid}.binvox"))
        .and_then(|s| s.strip_suffix(".obj"))
        .unwrap_or(fname)
        .to_string();

    // Translate adjustment: use per-file bbox where available, matching
    // the Python `obj_center - scale/2` rule; else the header's
    // un-swapped vector.
    let per_file = translate.per_file.get(&obj_key);
    let origin = per_file
        .map(|b| {
            let c = b.center();
            [
                c[0] - header.scale * 0.5,
                c[1] - header.scale * 0.5,
                c[2] - header.scale * 0.5,
            ]
        })
        .unwrap_or(header.translate_unswapped());

    let [vx, vy, vz] = header.voxel_size_axes();

    // Semantic lookup (object_type only; gml_ids come from the sidecar).
    let object_type = match index {
        Some(idx) => lookup_semantics(idx, &obj_key, grid),
        None => "Unknown".to_string(),
    };

    // Per-surface sidecar IDs (CityGML 3.0: building → surface → element).
    let sidecar_path = input_dir.join(format!("{obj_key}.json"));
    let ids = SurfaceSidecar::load(&sidecar_path)
        .map(|s| s.resolved_ids())
        .unwrap_or_else(|_| ResolvedIds::unknown());

    let element_gmlid = ids.element_gmlid.clone();
    let surface_gmlid = ids.surface_gmlid.clone();
    let building_gmlid = ids.building_gmlid.clone();

    // One (object, object_class) upsert per file, piggy-backed through
    // the same channel so it stays serialised against the COPY.
    if let Some(tx) = pg_tx {
        let _ = tx.blocking_send(RowMsg::Object {
            element_gmlid: element_gmlid.clone(),
            surface_gmlid: surface_gmlid.clone(),
            building_gmlid: building_gmlid.clone(),
            object_type: object_type.clone(),
        });
    }

    // Iterate occupied voxels, emitting rows.
    let mut emitted = 0u64;
    let mut emitted_batch = 0u64;
    for [ix, iy, iz] in bv.occupied_voxels() {
        let x = origin[0] + (ix as f64 + 0.5) * vx;
        let y = origin[1] + (iy as f64 + 0.5) * vy;
        let z = origin[2] + (iz as f64 + 0.5) * vz;
        let vp = voxid::compute(ix, iy, iz, [vx, vy, vz]);

        if let Some(c) = csv {
            c.write(&VoxelPayload {
                voxel_position: vp,
                x,
                y,
                z,
                srid,
                element_gmlid: &element_gmlid,
                surface_gmlid: &surface_gmlid,
                building_gmlid: &building_gmlid,
                object_type: &object_type,
            })?;
        }
        if let Some(tx) = pg_tx {
            let _ = tx.blocking_send(RowMsg::Row(OwnedVoxelRow {
                voxel_position: vp,
                x,
                y,
                z,
                srid,
                element_gmlid: element_gmlid.clone(),
                surface_gmlid: surface_gmlid.clone(),
                building_gmlid: building_gmlid.clone(),
            }));
        }
        emitted += 1;
        emitted_batch += 1;
        if emitted_batch >= 4096 {
            emitted_counter.fetch_add(emitted_batch, std::sync::atomic::Ordering::Relaxed);
            emitted_batch = 0;
        }
    }
    if emitted_batch > 0 {
        emitted_counter.fetch_add(emitted_batch, std::sync::atomic::Ordering::Relaxed);
    }
    debug!(file = %bv_path.display(), voxels = emitted, "processed binvox");
    Ok(emitted)
}

fn parse_grid_suffix(fname: &str) -> Option<u32> {
    // "<stem>.obj_<grid>.binvox"
    let stem = fname.strip_suffix(".binvox")?;
    let (_, grid_s) = stem.rsplit_once('_')?;
    grid_s.parse().ok()
}

fn lookup_semantics(index: &IndexFile, obj_key: &str, _grid: u32) -> String {
    // Exact key first, then basename fallback.
    let entry = index
        .get_entry(&format!("{obj_key}.obj"))
        .or_else(|| index.get_entry(obj_key));
    match entry {
        Some(e) => match (e.class.clone(), strip_namespace(&e.tag)) {
            (Some(cls), _) => cls,
            (None, tag) => tag,
        },
        None => "Unknown".to_string(),
    }
}

fn strip_namespace(tag: &str) -> String {
    match tag.rsplit_once('}') {
        Some((_, rest)) => rest.trim().to_string(),
        None => tag.to_string(),
    }
}

fn obj_bbox(path: &Path) -> Result<([f64; 3], [f64; 3])> {
    use std::io::BufRead;
    let f = std::fs::File::open(path)?;
    let r = std::io::BufReader::new(f);
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for line in r.lines() {
        let line = line?;
        let bytes = line.as_bytes();
        if bytes.len() < 2 || bytes[0] != b'v' || bytes[1] != b' ' {
            continue;
        }
        let mut it = line[2..].split_ascii_whitespace();
        let x: f64 = match it.next().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let y: f64 = match it.next().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let z: f64 = match it.next().and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        for (i, v) in [x, y, z].into_iter().enumerate() {
            if v < min[i] {
                min[i] = v;
            }
            if v > max[i] {
                max[i] = v;
            }
        }
    }
    Ok((min, max))
}

