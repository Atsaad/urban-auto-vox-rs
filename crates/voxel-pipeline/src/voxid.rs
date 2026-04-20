//! `voxel_position` BIGINT derivation.
//!
//! The Python pipeline computes a 13-digit decimal ID from grid indices:
//!
//! ```python
//! voxid = int(zfill(i, 5) + zfill(j, 5) + zfill(k, 3))
//! ```
//!
//! i.e. `i * 10^8 + j * 10^3 + k`. The `i`, `j`, `k` values come from
//! rescaling world coordinates by half the smallest voxel edge length
//! (`edge_length = min(vx, vy, vz) / 2`), which — for a uniform-cube
//! voxelization — simplifies to `2*ix + 1` etc. We recover the same
//! numbers directly from binvox grid indices, no data-dependent minima
//! needed.

/// Compute `voxel_position` for one voxel from its grid indices.
///
/// `vox_size` is the `[vx, vy, vz]` world-space edge lengths; for a
/// cuda_voxelizer cube these are equal but we support the general case.
///
/// Returns an `i64` to match the PostGIS BIGINT column type. The maximum
/// grid index after scaling is bounded by `MAX_GRID * 2 ≈ 4096`, so
/// `iiiiijjjjjkkk` fits within `10^13` — well below `i64::MAX`.
#[inline]
pub fn compute(ix: u32, iy: u32, iz: u32, vox_size: [f64; 3]) -> i64 {
    let edge = vox_size[0].min(vox_size[1]).min(vox_size[2]) * 0.5;
    // For a uniform grid (vx == vy == vz), factor reduces to 2.
    let fx = (vox_size[0] / edge).round() as i64;
    let fy = (vox_size[1] / edge).round() as i64;
    let fz = (vox_size[2] / edge).round() as i64;
    let i = (ix as i64) * fx + 1;
    let j = (iy as i64) * fy + 1;
    let k = (iz as i64) * fz + 1;
    i.saturating_mul(100_000_000) + j.saturating_mul(1_000) + k
}

#[cfg(test)]
mod tests {
    use super::compute;

    #[test]
    fn uniform_grid_matches_python_reference() {
        // For vx=vy=vz=0.5m: edge = 0.25, factor = 2 on each axis.
        // ix=0 → i=1; ix=1 → i=3; ix=2 → i=5.
        let v = [0.5, 0.5, 0.5];
        assert_eq!(compute(0, 0, 0, v), 1 * 100_000_000 + 1 * 1_000 + 1);
        assert_eq!(compute(1, 2, 3, v), 3 * 100_000_000 + 5 * 1_000 + 7);
    }

    #[test]
    fn non_cube_voxels_scale_per_axis() {
        // vx=1, vy=0.5, vz=0.25 → edge = 0.125; factors = [8, 4, 2].
        let v = [1.0, 0.5, 0.25];
        // ix=1 → i = 1*8 + 1 = 9
        // iy=1 → j = 1*4 + 1 = 5
        // iz=1 → k = 1*2 + 1 = 3
        assert_eq!(compute(1, 1, 1, v), 9 * 100_000_000 + 5 * 1_000 + 3);
    }
}
