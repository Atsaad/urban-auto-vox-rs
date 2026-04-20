//! Schema DDL.
//!
//! The identifier columns mirror the CityGML 3.0 containment hierarchy:
//! `building_gmlid` (top) → `surface_gmlid` (thematic surface) →
//! `element_gmlid` (geometry element — the most specific ID, typically
//! the longest). The `element_gmlid` serves as the `object` table's
//! primary key since it is unique per surface.
//!
//! The SRID is interpolated directly into the DDL because PostGIS
//! column types are not parameterisable.

use tokio_postgres::Client;

use crate::PostgisError;

/// Apply the `object` / `object_class` / `voxel` schema if it does not
/// already exist, plus the matching indexes. Idempotent — safe to run
/// on every pipeline start.
pub async fn apply_schema(client: &Client, srid: u32) -> Result<(), PostgisError> {
    let ddl = format!(
        r#"
        CREATE TABLE IF NOT EXISTS object (
            element_gmlid  TEXT PRIMARY KEY,
            surface_gmlid  TEXT,
            building_gmlid TEXT
        );

        CREATE TABLE IF NOT EXISTS object_class (
            id             SERIAL PRIMARY KEY,
            object_type    TEXT,
            element_gmlid  TEXT REFERENCES object(element_gmlid) ON DELETE CASCADE,
            surface_gmlid  TEXT,
            building_gmlid TEXT
        );

        CREATE TABLE IF NOT EXISTS voxel (
            id             SERIAL PRIMARY KEY,
            voxel_position BIGINT NOT NULL,
            vox_geom       GEOMETRY(PointZ, {srid}),
            x              DOUBLE PRECISION NOT NULL,
            y              DOUBLE PRECISION NOT NULL,
            z              DOUBLE PRECISION NOT NULL,
            element_gmlid  TEXT REFERENCES object(element_gmlid) ON DELETE CASCADE,
            surface_gmlid  TEXT,
            building_gmlid TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_voxel_geom             ON voxel USING GIST(vox_geom);
        CREATE INDEX IF NOT EXISTS idx_voxel_element_gmlid    ON voxel(element_gmlid);
        CREATE INDEX IF NOT EXISTS idx_voxel_surface_gmlid    ON voxel(surface_gmlid);
        CREATE INDEX IF NOT EXISTS idx_voxel_building_gmlid   ON voxel(building_gmlid);
        CREATE INDEX IF NOT EXISTS idx_object_class_element   ON object_class(element_gmlid);
        CREATE INDEX IF NOT EXISTS idx_object_class_type      ON object_class(object_type);
        "#,
        srid = srid
    );
    client.batch_execute(&ddl).await?;
    Ok(())
}

/// Idempotent upsert into `object` + one insert into `object_class`
/// per (element_gmlid, object_type) pair.
pub async fn upsert_object_and_class(
    client: &Client,
    element_gmlid: &str,
    surface_gmlid: &str,
    building_gmlid: &str,
    object_type: &str,
) -> Result<(), PostgisError> {
    client
        .execute(
            "INSERT INTO object (element_gmlid, surface_gmlid, building_gmlid) \
             VALUES ($1, $2, $3) \
             ON CONFLICT (element_gmlid) DO UPDATE SET \
                surface_gmlid  = EXCLUDED.surface_gmlid, \
                building_gmlid = EXCLUDED.building_gmlid",
            &[&element_gmlid, &surface_gmlid, &building_gmlid],
        )
        .await?;

    client
        .execute(
            "INSERT INTO object_class \
                 (object_type, element_gmlid, surface_gmlid, building_gmlid) \
             VALUES ($1, $2, $3, $4)",
            &[&object_type, &element_gmlid, &surface_gmlid, &building_gmlid],
        )
        .await?;
    Ok(())
}
