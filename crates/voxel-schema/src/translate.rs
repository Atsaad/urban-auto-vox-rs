//! `translate.json` — per-OBJ axis-aligned bounding boxes plus global bbox.
//!
//! Produced by step 4 of the pipeline (`create-translate`), consumed by step 5
//! (`create-index`) and step 6 (voxelizer) to drive per-file adaptive grid
//! sizing and to back-project voxel grid indices into the source CRS.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::SchemaError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateFile {
    pub global_bbox: GlobalBbox,
    pub per_file: BTreeMap<String, PerFileBbox>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalBbox {
    #[serde(default = "default_global_type")]
    pub json_featuretype: String,
    #[serde(rename = "_xmin")]
    pub xmin: f64,
    #[serde(rename = "_xmax")]
    pub xmax: f64,
    #[serde(rename = "_ymin")]
    pub ymin: f64,
    #[serde(rename = "_ymax")]
    pub ymax: f64,
    #[serde(rename = "_zmin")]
    pub zmin: f64,
    #[serde(rename = "_zmax")]
    pub zmax: f64,
}

fn default_global_type() -> String {
    "translate_model".into()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerFileBbox {
    pub obj_filename: String,
    #[serde(rename = "_xmin")]
    pub xmin: f64,
    #[serde(rename = "_xmax")]
    pub xmax: f64,
    #[serde(rename = "_ymin")]
    pub ymin: f64,
    #[serde(rename = "_ymax")]
    pub ymax: f64,
    #[serde(rename = "_zmin")]
    pub zmin: f64,
    #[serde(rename = "_zmax")]
    pub zmax: f64,
    pub translate: [f64; 3],
}

impl PerFileBbox {
    pub fn from_min_max(obj_filename: impl Into<String>, min: [f64; 3], max: [f64; 3]) -> Self {
        Self {
            obj_filename: obj_filename.into(),
            xmin: min[0],
            xmax: max[0],
            ymin: min[1],
            ymax: max[1],
            zmin: min[2],
            zmax: max[2],
            translate: min,
        }
    }

    #[inline]
    pub fn dimensions(&self) -> [f64; 3] {
        [
            self.xmax - self.xmin,
            self.ymax - self.ymin,
            self.zmax - self.zmin,
        ]
    }

    #[inline]
    pub fn max_dimension(&self) -> f64 {
        let [dx, dy, dz] = self.dimensions();
        dx.max(dy).max(dz)
    }

    #[inline]
    pub fn center(&self) -> [f64; 3] {
        [
            0.5 * (self.xmin + self.xmax),
            0.5 * (self.ymin + self.ymax),
            0.5 * (self.zmin + self.zmax),
        ]
    }
}

impl TranslateFile {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_matches_python_schema() {
        let raw = r#"{
          "global_bbox": {
            "json_featuretype": "translate_model",
            "_xmin": 0.0, "_xmax": 10.0,
            "_ymin": 0.0, "_ymax": 20.0,
            "_zmin": 0.0, "_zmax": 5.0
          },
          "per_file": {
            "b1": {
              "obj_filename": "b1.obj",
              "_xmin": 0.0, "_xmax": 4.0,
              "_ymin": 0.0, "_ymax": 3.0,
              "_zmin": 0.0, "_zmax": 2.0,
              "translate": [0.0, 0.0, 0.0]
            }
          }
        }"#;
        let parsed: TranslateFile = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.per_file["b1"].max_dimension(), 4.0);
        let again = serde_json::to_string(&parsed).unwrap();
        // Round-trip preserves `_xmin` style keys.
        assert!(again.contains("\"_xmin\":0"));
    }
}
