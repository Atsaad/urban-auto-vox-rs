//! `index.json` — aggregated semantic index keyed by OBJ filename.
//!
//! Produced by step 5 of the pipeline. Each entry holds the CityGML
//! thematic role (as a namespaced QName), the parent building ID and the
//! polygon GML ID, plus — when available — the CityGML `class` attribute
//! (`IfcWallStandardCase`, `IfcSlab`, ...) used to promote IFC-derived
//! objects to a finer-grained `object_type`.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::SchemaError;

pub const CITYGML_BUILDING_NS: &str = "{http://www.opengis.net/citygml/building/2.0}";
pub const CITYGML_CONSTRUCTION_NS: &str = "{http://www.opengis.net/citygml/construction/3.0}";

/// Known CityGML building-surface thematic roles.
pub const BUILDING_SURFACE_TYPES: &[&str] = &[
    "WallSurface",
    "RoofSurface",
    "GroundSurface",
    "OuterCeilingSurface",
    "OuterFloorSurface",
    "ClosureSurface",
];

/// Apply the Python pipeline's namespace rule: building/2.0 for known
/// semantic surfaces, construction/3.0 otherwise. Preserves the literal
/// role string (no substitution).
pub fn namespaced_tag(thematic_role: &str) -> String {
    if BUILDING_SURFACE_TYPES.contains(&thematic_role) {
        format!("{CITYGML_BUILDING_NS}{thematic_role}")
    } else {
        format!("{CITYGML_CONSTRUCTION_NS}{thematic_role}")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Crs {
    #[serde(rename = "srsName")]
    pub srs_name: String,
    #[serde(rename = "srsDimensions")]
    pub srs_dimensions: Vec<String>,
}

impl Crs {
    pub fn epsg(epsg: u32) -> Self {
        Self {
            srs_name: format!("EPSG:{epsg}"),
            srs_dimensions: vec!["3".into()],
        }
    }
}

/// A single entry in `index.json`.
///
/// `parent_id` and `gml_id` may appear in the wild as either a plain string
/// or a one-element list, and the pipeline has historically written both
/// variants. We accept either via [`OneOrMany`] on read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexEntry {
    pub tag: String,
    #[serde(rename = "parentID")]
    pub parent_id: OneOrMany<String>,
    #[serde(rename = "gmlID")]
    pub gml_id: OneOrMany<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub class: Option<String>,
}

/// Accepts `"x"` or `["x"]` or `["x", "y"]`; always serializes back out as
/// the wire form the caller produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T> {
    One(T),
    Many(Vec<T>),
}

impl<T: Clone> OneOrMany<T> {
    pub fn first(&self) -> Option<T> {
        match self {
            OneOrMany::One(v) => Some(v.clone()),
            OneOrMany::Many(v) => v.first().cloned(),
        }
    }
}

/// `index.json` as a whole.
///
/// Kept as a free-form map so that the `"CRS"` entry coexists with the
/// OBJ-keyed entries — mirroring the Python layout exactly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexFile {
    #[serde(flatten)]
    pub entries: BTreeMap<String, Value>,
}

impl IndexFile {
    pub fn new(crs: Crs) -> Self {
        let mut entries = BTreeMap::new();
        entries.insert("CRS".into(), serde_json::to_value(crs).unwrap());
        Self { entries }
    }

    pub fn insert_entry(&mut self, obj_key: impl Into<String>, entry: IndexEntry) {
        self.entries
            .insert(obj_key.into(), serde_json::to_value(entry).unwrap());
    }

    pub fn get_entry(&self, obj_key: &str) -> Option<IndexEntry> {
        let v = self.entries.get(obj_key)?;
        serde_json::from_value(v.clone()).ok()
    }

    pub fn crs(&self) -> Option<Crs> {
        self.entries
            .get("CRS")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespaced_tag_matches_python() {
        assert_eq!(
            namespaced_tag("WallSurface"),
            "{http://www.opengis.net/citygml/building/2.0}WallSurface"
        );
        assert_eq!(
            namespaced_tag("BuildingConstructiveElement"),
            "{http://www.opengis.net/citygml/construction/3.0}BuildingConstructiveElement"
        );
    }

    #[test]
    fn index_entry_accepts_python_wire_forms() {
        // As written by the Python pipeline: parentID is a list, gmlID is a
        // stringified list (the quirky historical form — we accept both).
        let raw = r#"{
          "tag": "{http://www.opengis.net/citygml/building/2.0}WallSurface",
          "parentID": ["DEBY_LOD2_4907506"],
          "gmlID": "DEBY_LOD2_4907506_abc_poly"
        }"#;
        let entry: IndexEntry = serde_json::from_str(raw).unwrap();
        assert_eq!(entry.parent_id.first().unwrap(), "DEBY_LOD2_4907506");
    }
}
