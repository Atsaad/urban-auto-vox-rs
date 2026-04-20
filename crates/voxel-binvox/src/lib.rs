//! Streaming reader for the binvox 1.0 voxel grid format.
//!
//! See <https://www.patrickmin.com/binvox/binvox.html> for the format spec.
//! A binvox file consists of:
//!
//! ```text
//! #binvox 1
//! dim  X Y Z
//! translate  TX TY TZ
//! scale  S
//! data
//! <RLE pairs of (value, count) bytes to EOF>
//! ```
//!
//! Occupancy is stored in X-major order: for `(i, j, k)` in `[0..X) x [0..Y) x [0..Z)`,
//! the linear index is `i*Y*Z + j*Z + k`. The Python reference collapses the
//! RLE into a dense `Vec<u8>`; this reader decodes on the fly and yields only
//! the occupied `(i, j, k)` triples, so a sparse grid costs O(occupied),
//! not O(X·Y·Z), in memory.
//!
//! # Y↔Z swap convention
//!
//! `cuda_voxelizer` emits a Y↔Z-swapped grid because it uses a Z-up
//! convention while OBJ is Y-up. The Python pipeline un-swaps the
//! `translate` vector when reading. We expose both the raw header value
//! and the un-swapped vector so callers can choose explicitly.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use memmap2::Mmap;

#[derive(Debug, thiserror::Error)]
pub enum BinvoxError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a binvox file (missing '#binvox' magic)")]
    BadMagic,
    #[error("malformed header line: {0}")]
    BadHeader(String),
    #[error("missing required header field: {0}")]
    MissingField(&'static str),
    #[error("truncated RLE payload: expected {expected} voxels, decoded {decoded}")]
    Truncated { expected: usize, decoded: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinvoxHeader {
    /// Grid dimensions `[X, Y, Z]` as they appear in the file.
    pub dims: [u32; 3],
    /// `translate` as it appears in the file header. For files produced by
    /// `cuda_voxelizer` this is Y↔Z-swapped relative to the source CRS.
    pub translate_raw: [f64; 3],
    pub scale: f64,
    /// Byte offset of the first RLE byte within the file.
    pub data_offset: usize,
}

impl BinvoxHeader {
    /// `translate` with Y and Z swapped back — i.e. what the Python
    /// reference returns for `cuda_voxelizer`-produced files.
    #[inline]
    pub fn translate_unswapped(&self) -> [f64; 3] {
        [
            self.translate_raw[0],
            self.translate_raw[2],
            self.translate_raw[1],
        ]
    }

    /// Per-axis world-space voxel size.
    #[inline]
    pub fn voxel_size_axes(&self) -> [f64; 3] {
        let s = self.scale;
        [
            s / self.dims[0] as f64,
            s / self.dims[1] as f64,
            s / self.dims[2] as f64,
        ]
    }

    #[inline]
    pub fn total_voxels(&self) -> usize {
        (self.dims[0] as usize) * (self.dims[1] as usize) * (self.dims[2] as usize)
    }
}

/// Parse the ASCII header of a binvox file, stopping at the first byte of
/// RLE payload. Returns the header and the inclusive offset of that byte.
pub fn parse_header<R: Read + BufRead>(reader: &mut R) -> Result<BinvoxHeader, BinvoxError> {
    let mut header_bytes_read = 0usize;
    let mut magic = String::new();
    header_bytes_read += reader.read_line(&mut magic)?;
    if !magic.starts_with("#binvox") {
        return Err(BinvoxError::BadMagic);
    }

    let mut dims: Option<[u32; 3]> = None;
    let mut translate: Option<[f64; 3]> = None;
    let mut scale: Option<f64> = None;

    loop {
        let mut line = String::new();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Err(BinvoxError::MissingField("data"));
        }
        header_bytes_read += n;
        let trimmed = line.trim();
        if trimmed == "data" {
            break;
        }
        if let Some(rest) = trimmed.strip_prefix("dim") {
            dims = Some(parse_triple_u32(rest.trim())?);
        } else if let Some(rest) = trimmed.strip_prefix("translate") {
            translate = Some(parse_triple_f64(rest.trim())?);
        } else if let Some(rest) = trimmed.strip_prefix("scale") {
            scale = Some(
                rest.trim()
                    .parse::<f64>()
                    .map_err(|_| BinvoxError::BadHeader(line.clone()))?,
            );
        } else if trimmed.is_empty() {
            continue;
        } else {
            return Err(BinvoxError::BadHeader(line));
        }
    }

    Ok(BinvoxHeader {
        dims: dims.ok_or(BinvoxError::MissingField("dim"))?,
        translate_raw: translate.ok_or(BinvoxError::MissingField("translate"))?,
        scale: scale.ok_or(BinvoxError::MissingField("scale"))?,
        data_offset: header_bytes_read,
    })
}

fn parse_triple_u32(s: &str) -> Result<[u32; 3], BinvoxError> {
    let mut it = s.split_ascii_whitespace();
    let parse = |tok: Option<&str>| -> Result<u32, BinvoxError> {
        tok.ok_or_else(|| BinvoxError::BadHeader(s.into()))?
            .parse::<u32>()
            .map_err(|_| BinvoxError::BadHeader(s.into()))
    };
    Ok([parse(it.next())?, parse(it.next())?, parse(it.next())?])
}

fn parse_triple_f64(s: &str) -> Result<[f64; 3], BinvoxError> {
    let mut it = s.split_ascii_whitespace();
    let parse = |tok: Option<&str>| -> Result<f64, BinvoxError> {
        tok.ok_or_else(|| BinvoxError::BadHeader(s.into()))?
            .parse::<f64>()
            .map_err(|_| BinvoxError::BadHeader(s.into()))
    };
    Ok([parse(it.next())?, parse(it.next())?, parse(it.next())?])
}

/// A complete binvox file backed by an mmap of the source bytes.
///
/// The payload is never materialised into a dense `Vec<u8>`; instead we
/// iterate the RLE directly via [`BinvoxFile::occupied_voxels`], which is
/// O(occupied) in both time and memory.
pub struct BinvoxFile {
    header: BinvoxHeader,
    _map: Mmap,
    payload_ptr: *const u8,
    payload_len: usize,
}

// SAFETY: `Mmap` is `Send + Sync` so the pointer + length pair it lends us
// is safe to share across threads as long as we don't mutate it.
unsafe impl Send for BinvoxFile {}
unsafe impl Sync for BinvoxFile {}

impl BinvoxFile {
    /// Open a binvox file, memory-mapping its contents and parsing the
    /// header.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, BinvoxError> {
        let file = File::open(path.as_ref())?;
        // SAFETY: mmap of a file we just opened — no concurrent writers
        // expected during a pipeline run.
        let map = unsafe { Mmap::map(&file)? };

        let mut cursor = std::io::Cursor::new(map.as_ref());
        let mut reader = BufReader::new(&mut cursor);
        let header = parse_header(&mut reader)?;

        let data_offset = header.data_offset;
        let payload_len = map.len().saturating_sub(data_offset);
        let payload_ptr = unsafe { map.as_ptr().add(data_offset) };

        Ok(Self {
            header,
            _map: map,
            payload_ptr,
            payload_len,
        })
    }

    #[inline]
    pub fn header(&self) -> &BinvoxHeader {
        &self.header
    }

    fn payload(&self) -> &[u8] {
        // SAFETY: `payload_ptr` + `payload_len` derives from the live mmap
        // we hold in `_map`.
        unsafe { std::slice::from_raw_parts(self.payload_ptr, self.payload_len) }
    }

    /// Iterate over occupied `(i, j, k)` grid indices in the file's native
    /// X-major order. Closed over the RLE stream, so a fully empty grid
    /// costs a single pass over the encoded payload with no decoding work
    /// per skipped run.
    pub fn occupied_voxels(&self) -> OccupiedIter<'_> {
        OccupiedIter::new(self.header.dims, self.payload())
    }

    /// Shortcut: collect occupied voxels into a `Vec`. Useful for tests
    /// and small files; for production pipelines prefer streaming.
    pub fn collect_occupied(&self) -> Result<Vec<[u32; 3]>, BinvoxError> {
        let dims = self.header.dims;
        let expected = self.header.total_voxels();
        let mut out = Vec::new();
        let mut decoded = 0usize;
        let payload = self.payload();
        let mut i = 0usize;
        while i + 1 < payload.len() {
            let value = payload[i];
            let count = payload[i + 1] as usize;
            i += 2;
            if value == 1 {
                for idx in decoded..decoded + count {
                    out.push(linear_to_xyz(idx, dims));
                }
            }
            decoded += count;
        }
        if decoded != expected {
            return Err(BinvoxError::Truncated { expected, decoded });
        }
        Ok(out)
    }
}

/// Streaming iterator over occupied grid indices. Decodes one RLE run at a
/// time; emits one `(i, j, k)` per occupied voxel.
pub struct OccupiedIter<'a> {
    dims: [u32; 3],
    payload: &'a [u8],
    cursor: usize,
    /// Remaining run length of the current value; always `0` when the
    /// iterator needs to consume another `(value, count)` pair.
    run_remaining: u32,
    run_value: u8,
    /// Linear index of the next voxel to emit if `run_value == 1`. Widened
    /// to `u64` because `2048³ > u32::MAX`.
    linear: u64,
}

impl<'a> OccupiedIter<'a> {
    fn new(dims: [u32; 3], payload: &'a [u8]) -> Self {
        Self {
            dims,
            payload,
            cursor: 0,
            run_remaining: 0,
            run_value: 0,
            linear: 0,
        }
    }

    fn next_run(&mut self) -> bool {
        if self.cursor + 1 >= self.payload.len() {
            return false;
        }
        self.run_value = self.payload[self.cursor];
        self.run_remaining = self.payload[self.cursor + 1] as u32;
        self.cursor += 2;
        true
    }
}

impl Iterator for OccupiedIter<'_> {
    type Item = [u32; 3];

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.run_remaining == 0 {
                if !self.next_run() {
                    return None;
                }
                // Degenerate run of length 0 — harmless, just read another.
                if self.run_remaining == 0 {
                    continue;
                }
                // Fast-skip runs of zeros without emitting anything.
                if self.run_value == 0 {
                    self.linear = self.linear.saturating_add(self.run_remaining as u64);
                    self.run_remaining = 0;
                    continue;
                }
            }
            let xyz = linear_to_xyz(self.linear as usize, self.dims);
            self.linear += 1;
            self.run_remaining -= 1;
            return Some(xyz);
        }
    }
}

#[inline]
fn linear_to_xyz(linear: usize, dims: [u32; 3]) -> [u32; 3] {
    // binvox stores values in X-major order: index = x*Y*Z + y*Z + z.
    let (y_sz, z_sz) = (dims[1] as usize, dims[2] as usize);
    let x = linear / (y_sz * z_sz);
    let rem = linear - x * y_sz * z_sz;
    let y = rem / z_sz;
    let z = rem - y * z_sz;
    [x as u32, y as u32, z as u32]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile_stub as tempfile;

    // Tiny stub so we don't need the tempfile crate for one test.
    mod tempfile_stub {
        use std::fs::File;
        use std::io;
        use std::path::PathBuf;

        pub struct NamedTempFile {
            path: PathBuf,
            file: Option<File>,
        }

        impl NamedTempFile {
            pub fn new() -> io::Result<Self> {
                let mut dir = std::env::temp_dir();
                let pid = std::process::id();
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos();
                dir.push(format!("binvox-test-{pid}-{nanos}.binvox"));
                let file = File::create(&dir)?;
                Ok(Self {
                    path: dir,
                    file: Some(file),
                })
            }
            pub fn as_file_mut(&mut self) -> &mut File {
                self.file.as_mut().unwrap()
            }
            pub fn path(&self) -> &std::path::Path {
                &self.path
            }
        }

        impl Drop for NamedTempFile {
            fn drop(&mut self) {
                self.file.take();
                let _ = std::fs::remove_file(&self.path);
            }
        }
    }

    fn write_binvox(dims: [u32; 3], rle: &[(u8, u32)]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        {
            let file = f.as_file_mut();
            writeln!(file, "#binvox 1").unwrap();
            writeln!(file, "dim {} {} {}", dims[0], dims[1], dims[2]).unwrap();
            writeln!(file, "translate 0.0 0.0 0.0").unwrap();
            writeln!(file, "scale 1.0").unwrap();
            writeln!(file, "data").unwrap();
            // Binvox RLE counts are single bytes: split long runs into chunks of 255.
            for (v, total) in rle {
                let mut remaining = *total;
                while remaining > 0 {
                    let chunk = remaining.min(255) as u8;
                    file.write_all(&[*v, chunk]).unwrap();
                    remaining -= chunk as u32;
                }
            }
        }
        f
    }

    #[test]
    fn header_parses_canonical_cuda_voxelizer_output() {
        let f = write_binvox([2, 2, 2], &[(0, 8)]);
        let bv = BinvoxFile::open(f.path()).unwrap();
        assert_eq!(bv.header().dims, [2, 2, 2]);
        assert_eq!(bv.header().scale, 1.0);
        assert_eq!(bv.header().translate_raw, [0.0; 3]);
    }

    #[test]
    fn single_occupied_voxel_at_origin() {
        // 2x2x2 grid: first voxel occupied, rest empty. 1+7 = 8 total.
        let f = write_binvox([2, 2, 2], &[(1, 1), (0, 7)]);
        let bv = BinvoxFile::open(f.path()).unwrap();
        let occ: Vec<_> = bv.occupied_voxels().collect();
        assert_eq!(occ, vec![[0, 0, 0]]);
    }

    #[test]
    fn streaming_and_collect_agree() {
        // All voxels occupied in a 3x3x3 grid.
        let f = write_binvox([3, 3, 3], &[(1, 27)]);
        let bv = BinvoxFile::open(f.path()).unwrap();
        let streamed: Vec<_> = bv.occupied_voxels().collect();
        let collected = bv.collect_occupied().unwrap();
        assert_eq!(streamed, collected);
        assert_eq!(streamed.len(), 27);
        // Last voxel should be the corner (2, 2, 2).
        assert_eq!(streamed.last().copied().unwrap(), [2, 2, 2]);
    }

    #[test]
    fn x_major_ordering_matches_spec() {
        // Exactly one occupied run of 1 at linear index 5, which in a
        // 2x3x4 grid decomposes as x=0, y=1, z=1 (0*12 + 1*4 + 1 = 5).
        let f = write_binvox([2, 3, 4], &[(0, 5), (1, 1), (0, 18)]);
        let bv = BinvoxFile::open(f.path()).unwrap();
        let occ: Vec<_> = bv.occupied_voxels().collect();
        assert_eq!(occ, vec![[0, 1, 1]]);
    }

    #[test]
    fn voxel_size_axes_partitions_scale_by_dim() {
        let f = write_binvox([10, 20, 40], &[(0, 8000)]);
        let bv = BinvoxFile::open(f.path()).unwrap();
        let header = bv.header();
        // We wrote scale=1.0 above.
        let [sx, sy, sz] = header.voxel_size_axes();
        assert!((sx - 0.1).abs() < 1e-12);
        assert!((sy - 0.05).abs() < 1e-12);
        assert!((sz - 0.025).abs() < 1e-12);
    }
}
