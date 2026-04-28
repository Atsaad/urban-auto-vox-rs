//! Schema DDL.
//!
//! Two tables, decoupled by design:
//!
//! * `building` (~9.9M rows) — loaded once, out-of-band, from
//!   `building_metadata.csv` via `psql \COPY`. The Rust pipeline never
//!   writes here; the table is created so that it's available for the
//!   tensor-export JOIN at training time.
//!
//! * `voxel` (~300M+ rows) — the only table the pipeline writes. Flat
//!   and denormalised: each row carries `building_gmlid`, the optional
//!   thematic `surface_gmlid`, the integer `surface_class`, and the
//!   `(x, y, z)` coordinates (plus an optional `vox_geom` for QGIS).
//!
//! No foreign key links the two — ingestion order is unconstrained, and
//! a 300M-row COPY pays no per-row FK validation cost. CityGML gml:id
//! integrity is high enough in practice that a missing JOIN row will be
//! the rare exception, surfaced naturally by the tensor-export query.
//!
//! The SRID is interpolated directly into the DDL because PostGIS
//! column types are not parameterisable.

use tokio_postgres::Client;

use crate::PostgisError;

/// Apply the `building` + `voxel` schema if it does not already exist,
/// plus the matching indexes. Idempotent — safe to run on every
/// pipeline start.
pub async fn apply_schema(client: &Client, srid: u32) -> Result<(), PostgisError> {
    let ddl = format!(
        r#"
        CREATE TABLE IF NOT EXISTS building (
            building_gmlid       TEXT PRIMARY KEY,
            tile_id              TEXT,
            function_code        TEXT,
            function_label       TEXT,
            roof_type_code       TEXT,
            roof_type_label      TEXT,
            measured_height      REAL,
            storeys_above_ground SMALLINT,
            storeys_source       TEXT,
            year_of_construction SMALLINT,
            gemeindeschluessel   TEXT,
            hoehe_dach           REAL,
            hoehe_grund          REAL,
            niedrigste_traufe    REAL,
            city                 TEXT,
            postal_code          TEXT,
            street_name          TEXT,
            house_number         TEXT,
            source               TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_building_gemeindeschluessel
            ON building (gemeindeschluessel);
        CREATE INDEX IF NOT EXISTS idx_building_function_code
            ON building (function_code);

        CREATE TABLE IF NOT EXISTS voxel (
            building_gmlid TEXT             NOT NULL,
            surface_gmlid  TEXT,
            surface_class  SMALLINT         NOT NULL,
            x              DOUBLE PRECISION NOT NULL,
            y              DOUBLE PRECISION NOT NULL,
            z              DOUBLE PRECISION NOT NULL,
            vox_geom       GEOMETRY(PointZ, {srid})
        );

        CREATE INDEX IF NOT EXISTS idx_voxel_building_gmlid
            ON voxel (building_gmlid);
        CREATE INDEX IF NOT EXISTS idx_voxel_geom
            ON voxel USING GIST (vox_geom);
        "#,
        srid = srid
    );
    client.batch_execute(&ddl).await?;
    Ok(())
}
