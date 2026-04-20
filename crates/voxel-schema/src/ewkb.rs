//! PostGIS EWKB encoding for `Point Z` with an SRID flag.
//!
//! Layout (little-endian):
//!
//! | byte | field                            |
//! |------|----------------------------------|
//! | 0    | 0x01 — little-endian indicator   |
//! | 1..5 | type = 0xA000_0001 (PointZ|SRID) |
//! | 5..9 | SRID (u32)                       |
//! | 9..33| X, Y, Z as f64                   |
//!
//! This matches `struct.pack` calls in the Python reference exactly and
//! round-trips through `pgAdmin`'s `vox_geom` column.

use byteorder::{LittleEndian, WriteBytesExt};

/// `Point Z` flag: bit 0 (Point) | bit 31 (Z) | bit 29 (SRID).
const WKB_TYPE_POINT_Z_SRID: u32 = 0xA000_0001;

pub const POINT_Z_EWKB_LEN: usize = 1 + 4 + 4 + 8 * 3; // 33

/// Encode a `Point Z` with SRID into its 33-byte EWKB representation.
#[inline]
pub fn point_z_ewkb_bytes(x: f64, y: f64, z: f64, srid: u32) -> [u8; POINT_Z_EWKB_LEN] {
    let mut out = [0u8; POINT_Z_EWKB_LEN];
    let mut cur: &mut [u8] = &mut out;
    cur.write_u8(1).unwrap();
    cur.write_u32::<LittleEndian>(WKB_TYPE_POINT_Z_SRID).unwrap();
    cur.write_u32::<LittleEndian>(srid).unwrap();
    cur.write_f64::<LittleEndian>(x).unwrap();
    cur.write_f64::<LittleEndian>(y).unwrap();
    cur.write_f64::<LittleEndian>(z).unwrap();
    out
}

/// Same as [`point_z_ewkb_bytes`] but hex-encoded (uppercase) — matches what
/// pgAdmin shows in the `vox_geom` column and what the Python pipeline
/// writes to CSV.
pub fn point_z_ewkb_hex(x: f64, y: f64, z: f64, srid: u32) -> String {
    let bytes = point_z_ewkb_bytes(x, y, z, srid);
    bytes_to_hex_upper(&bytes)
}

fn bytes_to_hex_upper(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0F) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_bytes_match_python_pack() {
        // Expected 9-byte prefix: 01 | 01 00 00 A0 | E8 64 00 00
        // (byte order | PointZ+SRID type | SRID=25832, all LE)
        let hex = point_z_ewkb_hex(0.0, 0.0, 0.0, 25832);
        assert_eq!(&hex[..18], "01010000A0E8640000");
    }

    #[test]
    fn full_length_is_66_hex_chars() {
        let hex = point_z_ewkb_hex(690800.0, 5335900.0, 100.0, 25832);
        assert_eq!(hex.len(), POINT_Z_EWKB_LEN * 2);
    }

    #[test]
    fn raw_byte_layout_matches_struct_pack() {
        let b = point_z_ewkb_bytes(1.0, 2.0, 3.0, 25832);
        assert_eq!(b[0], 0x01);
        // type = 0xA0000001 little-endian
        assert_eq!(&b[1..5], &[0x01, 0x00, 0x00, 0xA0]);
        // srid = 25832 = 0x000064E8 little-endian
        assert_eq!(&b[5..9], &[0xE8, 0x64, 0x00, 0x00]);
        // X = 1.0 as f64 little-endian
        assert_eq!(&b[9..17], &1.0_f64.to_le_bytes());
        assert_eq!(&b[17..25], &2.0_f64.to_le_bytes());
        assert_eq!(&b[25..33], &3.0_f64.to_le_bytes());
    }
}
