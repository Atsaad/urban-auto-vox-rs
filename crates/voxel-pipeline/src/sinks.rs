//! Output sinks.
//!
//! The CSV sink writes the same schema as the Python pipeline:
//!
//! ```text
//! voxel_position,vox_geom,gmlid,building_gmlid,class_gmlid,polygon_gmlid,object_type,X,Y,Z
//! ```
//!
//! The PostGIS sink wraps a [`voxel_postgis::VoxelCopyWriter`] and its
//! matching `object` / `object_class` upserts. Both sinks accept streamed
//! rows so the whole pipeline stays memory-bounded regardless of grid
//! count.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Mutex;

use anyhow::Result;
use voxel_schema::ewkb::point_z_ewkb_hex;

pub struct VoxelPayload<'a> {
    pub voxel_position: i64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub srid: u32,
    pub polygon_gml_id: &'a str,
    pub building_gml_id: &'a str,
    pub class_gml_id: &'a str,
    pub object_type: &'a str,
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
            "voxel_position",
            "vox_geom",
            "gmlid",
            "building_gmlid",
            "class_gmlid",
            "polygon_gmlid",
            "object_type",
            "X",
            "Y",
            "Z",
        ])?;
        Ok(Self {
            inner: Mutex::new(w),
        })
    }

    pub fn write(&self, row: &VoxelPayload<'_>) -> Result<()> {
        let hex = point_z_ewkb_hex(row.x, row.y, row.z, row.srid);
        let vp = row.voxel_position.to_string();
        let xs = ryu_f64(row.x);
        let ys = ryu_f64(row.y);
        let zs = ryu_f64(row.z);
        let mut w = self.inner.lock().unwrap();
        w.write_record([
            vp.as_str(),
            hex.as_str(),
            row.polygon_gml_id,
            row.building_gml_id,
            row.class_gml_id,
            row.polygon_gml_id,
            row.object_type,
            xs.as_str(),
            ys.as_str(),
            zs.as_str(),
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
