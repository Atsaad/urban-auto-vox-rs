#!/usr/bin/env python3
"""
Group 10M split .gml building files into fixed-size chunks sorted by
gemeindeschluessel. Writes one CSV manifest per chunk + a master index CSV.

Manifests are the durable artefact — they list which buildings belong to which
chunk and where each .gml file lives. The batch voxelization driver later
reads one manifest at a time, copies the listed .gml files to data/input/,
runs the pipeline, and writes the per-chunk voxel CSV.

No files are moved or symlinked here — only the manifest CSVs are created.
"""

from __future__ import annotations
import csv
import os
import sys
from pathlib import Path

SOURCE = Path("/home/ge27lof/Documents/github/MetaGML /004 split_buildings/with_address")
META = Path("local comments/projectoverview/building_metadata_clustered_decoded.csv")
OUT = Path("chunks")
CHUNK_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 5000


def main() -> int:
    if not SOURCE.is_dir():
        print(f"ERROR: source dir not found: {SOURCE}", file=sys.stderr)
        return 1
    if not META.is_file():
        print(f"ERROR: metadata CSV not found: {META}", file=sys.stderr)
        return 1

    print(f"[1/4] Reading metadata CSV ({META})...")
    rows: list[tuple[str, str, str]] = []
    with open(META, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["gmlid"], r["gemeindeschluessel"] or "", r["gemeinde"] or ""))
    print(f"      {len(rows):,} metadata rows")

    print(f"[2/4] Scanning {SOURCE} for .gml files...")
    available: set[str] = set()
    with os.scandir(SOURCE) as it:
        for e in it:
            n = e.name
            if n.endswith(".gml"):
                available.add(n[:-4])
    print(f"      {len(available):,} .gml files on disk")

    before = len(rows)
    rows = [r for r in rows if r[0] in available]
    if before != len(rows):
        print(f"      skipped {before - len(rows):,} metadata rows with no matching file")

    orphan_count = len(available) - len(rows)
    if orphan_count > 0:
        print(f"      {orphan_count:,} files on disk have no metadata (excluded from chunks)")

    print(f"[3/4] Sorting by (gemeindeschluessel, gmlid)...")
    rows.sort(key=lambda r: (r[1], r[0]))

    n_chunks = (len(rows) + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"[4/4] Writing {n_chunks:,} chunk manifests (chunk size = {CHUNK_SIZE})...")

    manifests_dir = OUT / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    index_path = OUT / "chunks_index.csv"
    src_str = str(SOURCE)
    with open(index_path, "w", newline="") as idx_f:
        idx_w = csv.writer(idx_f)
        idx_w.writerow([
            "chunk_id",
            "building_count",
            "gemeindeschluessel_first",
            "gemeindeschluessel_last",
            "gemeinde_first",
            "gemeinde_last",
            "n_distinct_gemeindeschluessel",
            "manifest_path",
        ])

        for c in range(n_chunks):
            chunk = rows[c * CHUNK_SIZE : (c + 1) * CHUNK_SIZE]
            cid = f"chunk_{c:05d}"
            mf_path = manifests_dir / f"{cid}.csv"

            distinct_gs: set[str] = set()
            with open(mf_path, "w", newline="") as mf:
                w = csv.writer(mf)
                w.writerow(["gmlid", "gemeindeschluessel", "gemeinde", "gml_path"])
                for gmlid, gs, gm in chunk:
                    w.writerow([gmlid, gs, gm, f"{src_str}/{gmlid}.gml"])
                    distinct_gs.add(gs)

            idx_w.writerow([
                cid,
                len(chunk),
                chunk[0][1],
                chunk[-1][1],
                chunk[0][2],
                chunk[-1][2],
                len(distinct_gs),
                str(mf_path),
            ])

            if (c + 1) % 200 == 0 or c + 1 == n_chunks:
                print(f"      wrote {c + 1:,}/{n_chunks:,} manifests", flush=True)

    print()
    print("=== DONE ===")
    print(f"Buildings chunked:  {len(rows):,}")
    print(f"Chunks written:     {n_chunks:,}")
    print(f"Avg per chunk:      {len(rows) / n_chunks:.1f}")
    print(f"Manifests:          {manifests_dir}/")
    print(f"Master index:       {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
