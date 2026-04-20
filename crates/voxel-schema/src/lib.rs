//! Shared on-disk contracts for the Urban-Auto-Vox pipeline.
//!
//! Every file format exchanged between pipeline steps lives here as a
//! serde-typed struct. The goal is that a rename or schema change is a
//! compile error, not a runtime `KeyError`.

pub mod ewkb;
pub mod grid_mapping;
pub mod index;
pub mod surface;
pub mod translate;

pub use ewkb::{point_z_ewkb_bytes, point_z_ewkb_hex};
pub use grid_mapping::{GridMappingFile, MIN_GRID, MAX_GRID};
pub use index::{namespaced_tag, IndexEntry, IndexFile, Crs, BUILDING_SURFACE_TYPES};
pub use surface::SurfaceSidecar;
pub use translate::{GlobalBbox, PerFileBbox, TranslateFile};

#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}
