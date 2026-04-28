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
//! building_gmlid TEXT
//! surface_gmlid  TEXT
//! surface_class  SMALLINT
//! x              DOUBLE PRECISION
//! y              DOUBLE PRECISION
//! z              DOUBLE PRECISION
//! vox_geom       GEOMETRY(PointZ, <SRID>)   (sent as the raw EWKB bytes)
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
const FIELD_COUNT: i16 = 7;

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
                "COPY voxel (building_gmlid, surface_gmlid, surface_class, \
                             x, y, z, vox_geom) \
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

        // building_gmlid, surface_gmlid TEXT
        put_text(&mut self.buf, &row.building_gmlid);
        put_text(&mut self.buf, &row.surface_gmlid);

        // surface_class SMALLINT
        put_smallint(&mut self.buf, row.surface_class);

        // x, y, z DOUBLE PRECISION
        put_double(&mut self.buf, row.x);
        put_double(&mut self.buf, row.y);
        put_double(&mut self.buf, row.z);

        // vox_geom GEOMETRY — send raw EWKB (PostGIS accepts this directly)
        let geom = row.ewkb();
        self.buf.put_u32(geom.len() as u32);
        self.buf.put_slice(&geom);

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
fn put_smallint(buf: &mut BytesMut, v: i16) {
    buf.put_u32(2);
    buf.put_i16(v);
}

#[inline]
fn put_double(buf: &mut BytesMut, v: f64) {
    buf.put_u32(8);
    buf.put_f64(v);
}

#[inline]
fn put_text(buf: &mut BytesMut, s: &str) {
    buf.put_u32(s.len() as u32);
    buf.put_slice(s.as_bytes());
}
