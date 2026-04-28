//! Per-surface sidecar JSON written by `rustcitygml2obj --add-json`.
//!
//! Emitted once per OBJ, co-located with it (`foo.obj` + `foo.json`). The
//! voxelizer reads this to populate the three CityGML 3.0 identifier
//! columns in the output schema: building → surface (thematic) → element
//! (polygon/geometry).

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

    /// Resolve the three CityGML 3.0 identifiers used downstream,
    /// substituting `"UNKNOWN"` when a field is absent.
    pub fn resolved_ids(&self) -> ResolvedIds {
        ResolvedIds {
            building_gmlid: self
                .building_id
                .clone()
                .unwrap_or_else(|| "UNKNOWN".into()),
            surface_gmlid: self
                .class_gml_id
                .clone()
                .unwrap_or_else(|| "UNKNOWN".into()),
            element_gmlid: self
                .polygon_gml_id
                .clone()
                .unwrap_or_else(|| "UNKNOWN".into()),
        }
    }
}

/// CityGML 3.0 identifier hierarchy for a single surface:
/// Building → ThematicSurface → geometry element.
#[derive(Debug, Clone)]
pub struct ResolvedIds {
    /// Top-level `Building` gml:id.
    pub building_gmlid: String,
    /// Thematic surface gml:id (e.g. a `RoofSurface`, `WallSurface`).
    pub surface_gmlid: String,
    /// Geometry element gml:id (the Polygon / SurfaceMember — the
    /// most specific identifier, typically the longest).
    pub element_gmlid: String,
}

impl ResolvedIds {
    pub fn unknown() -> Self {
        Self {
            building_gmlid: "UNKNOWN".into(),
            surface_gmlid: "UNKNOWN".into(),
            element_gmlid: "UNKNOWN".into(),
        }
    }
}

/// Map a CityGML thematic surface name to the integer class id stored on
/// each voxel row. The numbering matches the diffusion training labels —
/// `0` is reserved for air / unknown so that an empty cell in a generated
/// tensor naturally encodes "no surface here".
pub fn surface_class_id(role: &str) -> i16 {
    match role {
        "WallSurface" => 1,
        "RoofSurface" => 2,
        "GroundSurface" => 3,
        "OuterCeilingSurface" => 4,
        "ClosureSurface" => 5,
        _ => 0,
    }
}
