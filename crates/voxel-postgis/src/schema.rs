//! Schema DDL.
//!
//! Kept byte-identical to the Python reference `CREATE TABLE` block so
//! that databases populated by either implementation are interchangeable.
//! The SRID is interpolated directly into the DDL because PostGIS column
//! types are not parameterisable.

use tokio_postgres::Client;

use crate::PostgisError;

/// Apply the normalised `object` / `object_class` / `voxel` schema if it
/// does not already exist, plus the matching indexes.
///
/// Idempotent — safe to run on every pipeline start.
pub async fn apply_schema(client: &Client, srid: u32) -> Result<(), PostgisError> {
    let ddl = format!(
        r#"
        CREATE TABLE IF NOT EXISTS object (
            gmlid          TEXT PRIMARY KEY,
            building_gmlid TEXT,
            class_gmlid    TEXT,
            polygon_gmlid  TEXT
        );

        CREATE TABLE IF NOT EXISTS object_class (
            id             SERIAL PRIMARY KEY,
            object_type    TEXT,
            gmlid          TEXT REFERENCES object(gmlid) ON DELETE CASCADE,
            building_gmlid TEXT,
            class_gmlid    TEXT,
            polygon_gmlid  TEXT
        );

        CREATE TABLE IF NOT EXISTS voxel (
            id             SERIAL PRIMARY KEY,
            voxel_position BIGINT NOT NULL,
            vox_geom       GEOMETRY(PointZ, {srid}),
            gmlid          TEXT REFERENCES object(gmlid) ON DELETE CASCADE,
            building_gmlid TEXT,
            class_gmlid    TEXT,
            polygon_gmlid  TEXT
        );

        ALTER TABLE object       ADD COLUMN IF NOT EXISTS building_gmlid TEXT;
        ALTER TABLE object       ADD COLUMN IF NOT EXISTS class_gmlid    TEXT;
        ALTER TABLE object       ADD COLUMN IF NOT EXISTS polygon_gmlid  TEXT;
        ALTER TABLE object_class ADD COLUMN IF NOT EXISTS building_gmlid TEXT;
        ALTER TABLE object_class ADD COLUMN IF NOT EXISTS class_gmlid    TEXT;
        ALTER TABLE object_class ADD COLUMN IF NOT EXISTS polygon_gmlid  TEXT;
        ALTER TABLE voxel        ADD COLUMN IF NOT EXISTS building_gmlid TEXT;
        ALTER TABLE voxel        ADD COLUMN IF NOT EXISTS class_gmlid    TEXT;
        ALTER TABLE voxel        ADD COLUMN IF NOT EXISTS polygon_gmlid  TEXT;

        CREATE INDEX IF NOT EXISTS idx_voxel_geom           ON voxel USING GIST(vox_geom);
        CREATE INDEX IF NOT EXISTS idx_voxel_gmlid          ON voxel(gmlid);
        CREATE INDEX IF NOT EXISTS idx_voxel_building_gmlid ON voxel(building_gmlid);
        CREATE INDEX IF NOT EXISTS idx_voxel_class_gmlid    ON voxel(class_gmlid);
        CREATE INDEX IF NOT EXISTS idx_object_class_gmlid   ON object_class(gmlid);
        CREATE INDEX IF NOT EXISTS idx_object_class_type    ON object_class(object_type);
        "#,
        srid = srid
    );
    client.batch_execute(&ddl).await?;
    Ok(())
}

/// Idempotent upsert into `object` + one insert into `object_class` per
/// (gmlid, object_type) pair. Matches the Python `insert_object_and_class`
/// semantics.
pub async fn upsert_object_and_class(
    client: &Client,
    polygon_gmlid: &str,
    building_gmlid: &str,
    class_gmlid: &str,
    object_type: &str,
) -> Result<(), PostgisError> {
    client
        .execute(
            "INSERT INTO object (gmlid, building_gmlid, class_gmlid, polygon_gmlid) \
             VALUES ($1, $2, $3, $4) \
             ON CONFLICT (gmlid) DO UPDATE SET \
                building_gmlid = EXCLUDED.building_gmlid, \
                class_gmlid    = EXCLUDED.class_gmlid, \
                polygon_gmlid  = EXCLUDED.polygon_gmlid",
            &[&polygon_gmlid, &building_gmlid, &class_gmlid, &polygon_gmlid],
        )
        .await?;

    client
        .execute(
            "INSERT INTO object_class \
                 (object_type, gmlid, building_gmlid, class_gmlid, polygon_gmlid) \
             VALUES ($1, $2, $3, $4, $5)",
            &[
                &object_type,
                &polygon_gmlid,
                &building_gmlid,
                &class_gmlid,
                &polygon_gmlid,
            ],
        )
        .await?;
    Ok(())
}
