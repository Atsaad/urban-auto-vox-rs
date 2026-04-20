//! Per-surface sidecar JSON written by `rustcitygml2obj --add-json`.
//!
//! Emitted once per OBJ, co-located with it (`foo.obj` + `foo.json`). The
//! voxelizer reads this to populate the three gml_id columns in the output
//! schema (building / class / polygon).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::SchemaError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceSidecar {
    #[serde(default, alias = "building_id")]
    pub building_id: Option<String>,
    #[serde(default, alias = "class_gml_id")]
    pub class_gml_id: Option<String>,
    #[serde(default, alias = "polygon_gml_id")]
    pub polygon_gml_id: Option<String>,
    #[serde(default)]
    pub thematic_role: Option<String>,
    /// CityGML `class` attribute (e.g. `IfcWallStandardCase`). When present,
    /// the voxelizer promotes it to `object_type` in preference to the
    /// thematic role.
    #[serde(default)]
    pub class: Option<String>,

    /// Any additional fields produced by upstream tools. Preserved for
    /// round-tripping but not read by the voxelizer.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

impl SurfaceSidecar {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SchemaError> {
        let f = std::fs::File::open(path.as_ref())?;
        Ok(serde_json::from_reader(std::io::BufReader::new(f))?)
    }

    /// Resolve the three GML IDs used by the voxel schema, substituting
    /// `"UNKNOWN"` when a field is absent — matching the Python fallback.
    pub fn resolved_ids(&self) -> ResolvedIds {
        ResolvedIds {
            building_gml_id: self
                .building_id
                .clone()
                .unwrap_or_else(|| "UNKNOWN".into()),
            class_gml_id: self
                .class_gml_id
                .clone()
                .unwrap_or_else(|| "UNKNOWN".into()),
            polygon_gml_id: self
                .polygon_gml_id
                .clone()
                .unwrap_or_else(|| "UNKNOWN".into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedIds {
    pub building_gml_id: String,
    pub class_gml_id: String,
    pub polygon_gml_id: String,
}

impl ResolvedIds {
    pub fn unknown() -> Self {
        Self {
            building_gml_id: "UNKNOWN".into(),
            class_gml_id: "UNKNOWN".into(),
            polygon_gml_id: "UNKNOWN".into(),
        }
    }
}
