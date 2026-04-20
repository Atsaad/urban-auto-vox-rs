//! Binary `COPY voxel FROM STDIN BINARY` encoder.
//!
//! # Wire format
//!
//! The PostgreSQL binary COPY format is documented in the `SQL-COPY`
//! reference: <https://www.postgresql.org/docs/current/sql-copy.html>.
//! It consists of:
//!
//! 1. Signature: `"PGCOPY\n\xff\r\n\0"` (11 bytes).
//! 2. Flags field: `u32` (0 here — no OID).
//! 3. Header extension length: `u32` (0 here — no extension).
//! 4. For each tuple:
//!    - `i16`: field count (fixed across the COPY).
//!    - For each field, either `-1` (NULL) or `u32` length followed by
//!      the raw binary representation of the value in its column type.
//! 5. Trailer: `i16 = -1`.
//!
//! The target columns we emit, in order, are:
//!
//! ```text
//! voxel_position BIGINT
//! vox_geom       GEOMETRY(PointZ, <SRID>)   (sent as the raw EWKB bytes)
//! gmlid          TEXT
//! building_gmlid TEXT
//! class_gmlid    TEXT
//! polygon_gmlid  TEXT
//! ```
//!
//! PostGIS' `geometry` type accepts the raw EWKB bytes as its binary wire
//! form, which is exactly what [`voxel_schema::ewkb`] produces.

use std::pin::Pin;

use bytes::{BufMut, BytesMut};
use futures_util::SinkExt;
use tokio_postgres::{Client, CopyInSink};
use tracing::debug;

use crate::{PostgisError, VoxelRow};

const SIGNATURE: &[u8; 11] = b"PGCOPY\n\xff\r\n\0";
const FIELD_COUNT: i16 = 6;

/// Streaming writer that owns a `COPY ... FROM STDIN BINARY` sink and
/// buffers encoded tuples before flushing to the server.
pub struct VoxelCopyWriter {
    sink: Pin<Box<CopyInSink<bytes::Bytes>>>,
    buf: BytesMut,
    flush_threshold: usize,
    rows_pending: u64,
    rows_total: u64,
}

impl VoxelCopyWriter {
    /// Start a new binary COPY into the `voxel` table. `flush_threshold`
    /// controls how often the buffered tuples are flushed (in bytes).
    pub async fn begin(client: &Client, flush_threshold: usize) -> Result<Self, PostgisError> {
        let sink: CopyInSink<bytes::Bytes> = client
            .copy_in(
                "COPY voxel (voxel_position, vox_geom, gmlid, \
                             building_gmlid, class_gmlid, polygon_gmlid) \
                 FROM STDIN (FORMAT BINARY)",
            )
            .await?;

        let mut buf = BytesMut::with_capacity(flush_threshold.max(16 * 1024));
        buf.put_slice(SIGNATURE);
        buf.put_u32(0); // flags
        buf.put_u32(0); // header extension length

        Ok(Self {
            sink: Box::pin(sink),
            buf,
            flush_threshold,
            rows_pending: 0,
            rows_total: 0,
        })
    }

    /// Append one tuple.
    pub async fn write_row(&mut self, row: &VoxelRow) -> Result<(), PostgisError> {
        self.buf.put_i16(FIELD_COUNT);

        // voxel_position BIGINT
        put_bigint(&mut self.buf, row.voxel_position);

        // vox_geom GEOMETRY — send raw EWKB (PostGIS accepts this directly)
        let geom = row.ewkb();
        self.buf.put_u32(geom.len() as u32);
        self.buf.put_slice(&geom);

        // Four TEXT columns
        put_text(&mut self.buf, &row.polygon_gml_id);
        put_text(&mut self.buf, &row.building_gml_id);
        put_text(&mut self.buf, &row.class_gml_id);
        put_text(&mut self.buf, &row.polygon_gml_id);

        self.rows_pending += 1;
        self.rows_total += 1;

        if self.buf.len() >= self.flush_threshold {
            self.flush().await?;
        }
        Ok(())
    }

    /// Flush any buffered tuples to the server.
    pub async fn flush(&mut self) -> Result<(), PostgisError> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let chunk = self.buf.split().freeze();
        let size = chunk.len();
        let rows = std::mem::take(&mut self.rows_pending);
        // `Pin<Box<CopyInSink<_>>>: Sink<Bytes>` via the blanket impl for
        // `Pin<P> where P: DerefMut + Unpin`, so `.send()` resolves directly.
        self.sink.send(chunk).await?;
        debug!(bytes = size, rows, "flushed COPY chunk");
        Ok(())
    }

    /// Close the COPY stream, emitting the trailer. Returns the total
    /// number of rows written.
    pub async fn finish(mut self) -> Result<u64, PostgisError> {
        // Trailer: i16 = -1
        self.buf.put_i16(-1);
        self.flush().await?;
        let written = self.sink.as_mut().finish().await?;
        debug!(
            rows = self.rows_total,
            rows_reported = written,
            "COPY committed"
        );
        Ok(self.rows_total)
    }
}

#[inline]
fn put_bigint(buf: &mut BytesMut, v: i64) {
    buf.put_u32(8);
    buf.put_i64(v);
}

#[inline]
fn put_text(buf: &mut BytesMut, s: &str) {
    buf.put_u32(s.len() as u32);
    buf.put_slice(s.as_bytes());
}
