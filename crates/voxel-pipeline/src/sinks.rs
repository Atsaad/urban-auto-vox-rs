//! Output sinks.
//!
//! The CSV sink writes one row per occupied voxel, mirroring the flat
//! `voxel` table column layout:
//!
//! ```text
//! building_gmlid,surface_gmlid,surface_class,x,y,z,vox_geom
//! ```
//!
//! The PostGIS sink wraps a [`voxel_postgis::VoxelCopyWriter`]. Both
//! sinks accept streamed rows so the whole pipeline stays memory-bounded
//! regardless of grid count.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Mutex;

use anyhow::Result;
use voxel_schema::ewkb::point_z_ewkb_hex;

pub struct VoxelPayload<'a> {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub srid: u32,
    pub surface_class: i16,
    pub surface_gmlid: &'a str,
    pub building_gmlid: &'a str,
}

/// Thread-safe streaming CSV writer. Rows are serialised under a mutex
/// because `csv::Writer` is not `Sync`.
pub struct CsvSink {
    inner: Mutex<csv::Writer<BufWriter<File>>>,
}

impl CsvSink {
    pub fn create(path: &Path) -> Result<Self> {
        let file = File::create(path)?;
        let bw = BufWriter::with_capacity(1 << 20, file);
        let mut w = csv::WriterBuilder::new()
            .has_headers(false)
            .from_writer(bw);
        w.write_record([
            "building_gmlid",
            "surface_gmlid",
            "surface_class",
            "x",
            "y",
            "z",
            "vox_geom",
        ])?;
        Ok(Self {
            inner: Mutex::new(w),
        })
    }

    pub fn write(&self, row: &VoxelPayload<'_>) -> Result<()> {
        let hex = point_z_ewkb_hex(row.x, row.y, row.z, row.srid);
        let cls = row.surface_class.to_string();
        let xs = ryu_f64(row.x);
        let ys = ryu_f64(row.y);
        let zs = ryu_f64(row.z);
        let mut w = self.inner.lock().unwrap();
        w.write_record([
            row.building_gmlid,
            row.surface_gmlid,
            cls.as_str(),
            xs.as_str(),
            ys.as_str(),
            zs.as_str(),
            hex.as_str(),
        ])?;
        Ok(())
    }

    pub fn finish(self) -> Result<()> {
        let mut w = self.inner.into_inner().unwrap();
        w.flush()?;
        Ok(())
    }
}

/// Shortest round-trip float formatting, without pulling in a dependency.
/// `f64::to_string()` is well-behaved enough for CSV; this wrapper
/// exists to keep the call sites tidy if we later swap in `ryu::Buffer`.
#[inline]
fn ryu_f64(v: f64) -> String {
    v.to_string()
}
