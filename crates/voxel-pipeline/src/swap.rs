//! OBJ Y↔Z coordinate swap.
//!
//! `cuda_voxelizer` expects Z-up geometry; CityGML / OBJ uses Y-up. Rather
//! than mutating the source OBJs, we copy them into a scratch directory
//! with `v x y z` lines rewritten as `v x z y`. All non-vertex lines pass
//! through verbatim. The Python reference does this line-by-line with a
//! Python `for` loop; here we stream bytes through `BufWriter` with zero
//! per-line allocations beyond the input buffer.

use std::io::{BufRead, BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rayon::prelude::*;
use tracing::debug;

pub fn swap_dir(input: &Path, output: &Path) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(output)
        .with_context(|| format!("creating {}", output.display()))?;

    let objs: Vec<PathBuf> = std::fs::read_dir(input)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|e| e == "obj").unwrap_or(false))
        .collect();

    objs.par_iter()
        .map(|src| {
            let name = src.file_name().context("bad obj path")?;
            let dst = output.join(name);
            swap_file(src, &dst)?;
            Ok(dst)
        })
        .collect()
}

pub fn swap_file(src: &Path, dst: &Path) -> Result<()> {
    let f_in = std::fs::File::open(src).with_context(|| format!("opening {}", src.display()))?;
    let f_out = std::fs::File::create(dst)
        .with_context(|| format!("creating {}", dst.display()))?;
    let mut reader = std::io::BufReader::new(f_in);
    let mut writer = BufWriter::new(f_out);

    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            break;
        }
        let stripped = line.trim_end_matches(&['\n', '\r'][..]);
        if let Some(rewritten) = rewrite_vertex_line(stripped) {
            writer.write_all(rewritten.as_bytes())?;
            writer.write_all(b"\n")?;
        } else {
            writer.write_all(line.as_bytes())?;
        }
    }
    writer.flush()?;
    debug!(src = %src.display(), dst = %dst.display(), "Y<->Z swapped");
    Ok(())
}

fn rewrite_vertex_line(line: &str) -> Option<String> {
    let bytes = line.as_bytes();
    if bytes.len() < 2 || bytes[0] != b'v' || bytes[1] != b' ' {
        return None;
    }
    let mut it = line[2..].split_ascii_whitespace();
    let x = it.next()?;
    let y = it.next()?;
    let z = it.next()?;
    let mut out = format!("v {x} {z} {y}");
    for extra in it {
        out.push(' ');
        out.push_str(extra);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::rewrite_vertex_line;

    #[test]
    fn rewrites_only_vertex_lines() {
        assert_eq!(
            rewrite_vertex_line("v 1.0 2.0 3.0").as_deref(),
            Some("v 1.0 3.0 2.0")
        );
        assert!(rewrite_vertex_line("vn 0 1 0").is_none());
        assert!(rewrite_vertex_line("f 1 2 3").is_none());
        assert!(rewrite_vertex_line("# comment").is_none());
        assert_eq!(
            rewrite_vertex_line("v 1.0 2.0 3.0 0.5 0.5 0.5").as_deref(),
            Some("v 1.0 3.0 2.0 0.5 0.5 0.5")
        );
    }
}
