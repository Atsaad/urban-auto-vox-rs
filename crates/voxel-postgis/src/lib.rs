//! PostGIS writer for the Urban-Auto-Vox voxel schema.
//!
//! The hot path is `COPY voxel FROM STDIN BINARY`, which sends pre-encoded
//! rows over the wire as a single long-running statement. In the Python
//! reference this path uses SQLAlchemy `executemany` on parameterised
//! `INSERT ... ST_GeomFromText(...)` statements, which pays one parse +
//! one PostGIS geometry-construction per row. Binary COPY avoids both.

pub mod copy_binary;
pub mod schema;

use std::time::Duration;

use tokio_postgres::{Client, NoTls};
use tracing::{debug, info};
use voxel_schema::ewkb::point_z_ewkb_bytes;
use voxel_schema::ewkb::POINT_Z_EWKB_LEN;

pub use copy_binary::VoxelCopyWriter;
pub use schema::apply_schema;

#[derive(Debug, thiserror::Error)]
pub enum PostgisError {
    #[error("postgres: {0}")]
    Postgres(#[from] tokio_postgres::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("configuration: {0}")]
    Config(String),
}

#[derive(Debug, Clone)]
pub struct PgConnectionConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub user: String,
    pub password: String,
    pub connect_timeout: Duration,
}

impl PgConnectionConfig {
    pub fn to_conn_string(&self) -> String {
        format!(
            "host={} port={} dbname={} user={} password={} connect_timeout={}",
            self.host,
            self.port,
            self.database,
            self.user,
            self.password,
            self.connect_timeout.as_secs().max(1),
        )
    }
}

/// Open a PostgreSQL connection and verify that PostGIS is available.
pub async fn connect(cfg: &PgConnectionConfig) -> Result<Client, PostgisError> {
    let (client, conn) = tokio_postgres::connect(&cfg.to_conn_string(), NoTls).await?;
    tokio::spawn(async move {
        if let Err(e) = conn.await {
            tracing::warn!(error = %e, "postgres connection closed");
        }
    });
    let row = client.query_one("SELECT version()", &[]).await?;
    let version: String = row.get(0);
    info!(%version, "connected to PostgreSQL");

    // PostGIS extension probe — fail early with a clear message if it's
    // missing rather than blowing up mid-COPY.
    let has_postgis: bool = client
        .query_one(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'postgis')",
            &[],
        )
        .await?
        .get(0);
    if !has_postgis {
        return Err(PostgisError::Config(
            "PostGIS extension not installed in target database".into(),
        ));
    }
    debug!("PostGIS extension present");
    Ok(client)
}

/// One voxel row in the flat `voxel` table.
///
/// `building_gmlid` joins (without FK) to the `building` table loaded
/// from `building_metadata.csv`. `surface_gmlid` is the thematic-surface
/// gml:id from CityGML — kept for traceability, not required by the
/// diffusion pipeline. `surface_class` is the integer mapping produced
/// by [`voxel_schema::surface::surface_class_id`].
#[derive(Debug, Clone)]
pub struct VoxelRow {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub srid: u32,
    pub surface_class: i16,
    pub building_gmlid: String,
    pub surface_gmlid: String,
}

impl VoxelRow {
    /// Encode the PointZ geometry in EWKB. Length is a compile-time constant.
    #[inline]
    pub fn ewkb(&self) -> [u8; POINT_Z_EWKB_LEN] {
        point_z_ewkb_bytes(self.x, self.y, self.z, self.srid)
    }
}
