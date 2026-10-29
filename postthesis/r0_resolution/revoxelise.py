#!/usr/bin/env python3
"""
R0 -- re-voxelise the 10,000 training buildings at 0.25 m.

WHY: X7 showed a gable is destroyed at 0.5 m and fully recovered at 0.25 m
(postthesis/results/x7_resolution/FINDINGS.md). Training at 0.25 m needs the
SAME buildings the thesis trained on, re-voxelised -- not a new corpus.

WHAT IT DOES, per batch of gmlIDs:
    <SRC>/<gmlid>.gml  --citygml-tools-->  gml3/  --rustcitygml2obj-->  objs/
                       --voxel-pipeline-rs @0.25m-->  objs/*.csv
and collects the voxel + grid_sizes CSVs under out/.

`docker run`, never `docker compose`: compose pins container names
(voxel_pipeline_rs) and ./data mounts, so a run here would collide with the
production stack and any batch job. PIPELINE_OUTPUT_FORMAT=csv means the
pipeline never constructs a DB client, so PostGIS is never touched.

Resumable: a batch whose out/batch_NNNN_voxels.csv exists is skipped.
"""
from __future__ import annotations
import argparse, csv, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# The per-building CityGML source. Set R0_SRC, or pass --src.
SRC  = Path(os.environ.get(
    "R0_SRC", "~/Documents/github/MetaGML /004 split_buildings/with_address"
)).expanduser()
HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
OUT  = HERE / "out"

IMG_TOOLS = "citygml4j/citygml-tools:latest"
IMG_OBJ   = "atsaad/rustcitygml2obj:v3.2"
IMG_VOX   = "atsaad/voxel-pipeline-rs:0.3.2"


def run(cmd, timeout, label):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    if p.returncode != 0:
        sys.stderr.write(f"\n--- {label} FAILED (rc={p.returncode}) ---\n")
        sys.stderr.write((p.stdout or "")[-3000:])
        sys.stderr.write((p.stderr or "")[-3000:])
        raise RuntimeError(label)
    return dt


def manifest_ids(manifest: Path) -> list[str]:
    return [r["gmlid"] for r in csv.DictReader(open(manifest, newline=""))]


def do_batch(ids, bi, voxel, workers, keep):
    tag = f"batch_{bi:04d}"
    vox_out = OUT / f"{tag}_voxels.csv"
    if vox_out.exists():
        print(f"  {tag}: already done, skipping")
        return None

    job = WORK / tag
    if job.exists():
        shutil.rmtree(job)
    (job / "gml2").mkdir(parents=True)
    (job / "gml3").mkdir()
    (job / "objs").mkdir()

    missing = []
    for g in ids:
        s = SRC / f"{g}.gml"
        if s.exists():
            shutil.copy2(s, job / "gml2" / f"{g}.gml")
        else:
            missing.append(g)

    timings = {}
    timings["upgrade"] = run([
        "docker", "run", "--rm", "-u", "0",
        "-v", f"{job/'gml2'}:/data/input", "-v", f"{job/'gml3'}:/data/output",
        IMG_TOOLS, "upgrade", "/data/input/*.gml", "-o", "/data/output",
    ], 1800, f"{tag} upgrade")

    timings["obj"] = run([
        "docker", "run", "--rm",
        "-v", f"{job/'gml3'}:/data/input", "-v", f"{job/'objs'}:/data/output",
        IMG_OBJ, "--input", "/data/input", "--output", "/data/output",
        "--tbw", "--add-json",
    ], 1800, f"{tag} obj")

    timings["voxelise"] = run([
        "docker", "run", "--rm", "--gpus", "all",
        "-e", "NVIDIA_VISIBLE_DEVICES=all",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-e", "PIPELINE_INPUT_DIR=/app/data",
        "-e", "PIPELINE_GML_DIR=/app/gml",
        "-e", f"PIPELINE_VOXEL_SIZE={voxel}",
        "-e", "PIPELINE_DB_SRID=25832",
        "-e", f"PIPELINE_NUM_WORKERS={workers}",
        "-e", "PIPELINE_OUTPUT_FORMAT=csv",
        "-e", "PIPELINE_CUDA_VOXELIZER=/app/cuda_voxelizer/build/cuda_voxelizer",
        "-v", f"{job/'objs'}:/app/data:rw",
        "-v", f"{job/'gml3'}:/app/gml:ro",
        "-w", "/app", IMG_VOX, "run",
    ], 7200, f"{tag} voxelise")

    # collect: the pipeline may emit one CSV or several; concatenate, header once
    vcsv = sorted(p for p in (job / "objs").glob("*.csv") if "grid_size" not in p.name)
    gcsv = sorted((job / "objs").glob("*grid_size*.csv"))
    if not vcsv:
        raise RuntimeError(f"{tag}: pipeline produced no voxel CSV")

    def cat(srcs, dst):
        n = 0
        with open(dst, "w", newline="") as w:
            for i, s in enumerate(srcs):
                with open(s, newline="") as r:
                    for j, line in enumerate(r):
                        if j == 0 and i > 0:
                            continue
                        w.write(line)
                        n += 1
        return n

    nv = cat(vcsv, vox_out)
    ng = cat(gcsv, OUT / f"{tag}_grid_sizes.csv") if gcsv else 0

    if not keep:
        shutil.rmtree(job)

    print(f"  {tag}: {len(ids)-len(missing)} bldgs, {nv-1} voxel rows, {max(ng-1,0)} grid rows"
          f"  [upgrade {timings['upgrade']:.0f}s  obj {timings['obj']:.0f}s"
          f"  vox {timings['voxelise']:.0f}s]"
          + (f"  MISSING {len(missing)}" if missing else ""))
    return timings


def main():
    global OUT, WORK, SRC
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "tensorbuilding/shards/manifest.csv"))
    ap.add_argument("--voxel", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=250)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="pilot: only this many buildings")
    ap.add_argument("--keep", action="store_true", help="keep intermediate job dirs")
    ap.add_argument("--out", default=str(OUT),
                    help="where the per-batch CSVs go; a run at a different voxel "
                         "size needs its own directory, because a batch whose CSV "
                         "is already there is skipped")
    ap.add_argument("--work", default=str(WORK), help="intermediate job dirs")
    ap.add_argument("--src", default=str(SRC),
                    help="directory of per-building CityGML, one <gmlid>.gml each")
    a = ap.parse_args()

    # Absolute: these are bind-mounted into containers, and Docker rejects a
    # relative source path as a volume name.
    OUT, WORK = Path(a.out).resolve(), Path(a.work).resolve()
    SRC = Path(a.src).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    ids = manifest_ids(Path(a.manifest))
    if a.limit:
        ids = ids[:a.limit]
    nb = (len(ids) + a.batch - 1) // a.batch
    print(f"R0 re-voxelisation @ {a.voxel} m -- {len(ids)} buildings in {nb} batches of {a.batch}")
    print(f"out: {OUT}\n")

    t0 = time.time()
    for bi in range(nb):
        do_batch(ids[bi*a.batch:(bi+1)*a.batch], bi, a.voxel, a.workers, a.keep)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
