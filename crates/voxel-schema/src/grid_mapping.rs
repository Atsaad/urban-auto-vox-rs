//! `grid_mapping.json` — target voxel size + per-OBJ adaptive grid resolution.
//!
//! Given each file's bounding box and a target physical voxel edge length in
//! metres, the minimum grid side length is
//!
//! ```text
//! g = clamp(ceil(max_dimension / target_voxel_size), MIN_GRID, MAX_GRID)
//! ```
//!
//! guaranteeing that the resulting voxel is at most `target_voxel_size`
//! metres on its longest axis. The bounds match the Python reference
//! implementation exactly.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::translate::TranslateFile;
use crate::SchemaError;

pub const MIN_GRID: u32 = 8;
pub const MAX_GRID: u32 = 2048;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridMappingFile {
    pub target_voxel_size: f64,
    pub total_files: usize,
    /// Grid size -> file count. Stringly-keyed to match the Python JSON
    /// (JSON object keys must be strings).
    pub grid_distribution: BTreeMap<String, u32>,
    /// OBJ filename (with `.obj` extension) -> grid size.
    pub grid_mapping: BTreeMap<String, u32>,
}

impl GridMappingFile {
    /// Compute the grid mapping from a loaded [`TranslateFile`] and a target
    /// voxel edge length in metres.
    pub fn from_translate(translate: &TranslateFile, target_voxel_size: f64) -> Self {
        assert!(
            target_voxel_size > 0.0 && target_voxel_size.is_finite(),
            "target_voxel_size must be positive and finite"
        );

        let mut grid_mapping = BTreeMap::new();
        let mut grid_distribution: BTreeMap<String, u32> = BTreeMap::new();

        for bbox in translate.per_file.values() {
            let g = min_grid_size(bbox.max_dimension(), target_voxel_size);
            grid_mapping.insert(bbox.obj_filename.clone(), g);
            *grid_distribution.entry(g.to_string()).or_insert(0) += 1;
        }

        Self {
            target_voxel_size,
            total_files: translate.per_file.len(),
            grid_distribution,
            grid_mapping,
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, SchemaError> {
        let f = std::fs::File::open(path.as_ref())?;
        Ok(serde_json::from_reader(std::io::BufReader::new(f))?)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), SchemaError> {
        let f = std::fs::File::create(path.as_ref())?;
        serde_json::to_writer_pretty(std::io::BufWriter::new(f), self)?;
        Ok(())
    }
}

/// `clamp(ceil(max_dim / target), MIN_GRID, MAX_GRID)`.
#[inline]
pub fn min_grid_size(max_dimension: f64, target_voxel_size: f64) -> u32 {
    let exact = max_dimension / target_voxel_size;
    let required = exact.ceil() as i64;
    required.clamp(MIN_GRID as i64, MAX_GRID as i64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_reference_cases() {
        // From Python: required_grid = int(math.ceil(max_dim / target)), clamped [8, 2048].
        assert_eq!(min_grid_size(10.0, 0.5), 20);
        assert_eq!(min_grid_size(0.1, 0.5), 8); // clamped to min
        assert_eq!(min_grid_size(5000.0, 1.0), 2048); // clamped to max
        assert_eq!(min_grid_size(127.9, 0.5), 256); // 255.8 -> ceil -> 256
    }
}
