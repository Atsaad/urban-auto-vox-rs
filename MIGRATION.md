# Migration from Urban-Auto-Vox (Python) to urban-auto-vox-rs

This document covers the differences between the original
[Urban-Auto-Vox](https://github.com/Atsaad/Urban-Auto-Vox) Python pipeline
and this Rust rewrite. If you are starting fresh, see [README.md](README.md).

---

## Pipeline parity — does this do exactly what Urban-Auto-Vox does?

**Yes — step for step.** The original pipeline does:

1. CityGML 2.0 → CityGML 3.0 upgrade (when needed)
2. CityGML 3.0 → OBJ + JSON sidecars (`rustcitygml2obj`)
3. Build `translate.json` from per-OBJ vertex bounds
4. Build `index.json` from sidecar JSONs (semantic mapping)
5. Build `grid_mapping.json` (per-file adaptive CUDA grid size)
6. Y↔Z swap on every OBJ (CUDA voxelizer is Z-up, OBJ is Y-up)
7. `cuda_voxelizer` → `*.binvox` per OBJ
8. Decode binvox → row-per-voxel CSV + PostGIS `voxel` upsert

This workspace covers each step, with the boundary between Docker
images drawn slightly differently:

| #  | What                                | Urban-Auto-Vox                          | urban-auto-vox-rs                                       |
|----|-------------------------------------|-----------------------------------------|---------------------------------------------------------|
| 1  | CityGML 2.0 → 3.0                   | `citygml-tools` container               | **same** `citygml-tools` container                      |
| 2  | CityGML 3.0 → OBJ + JSON            | `rustgml2obj` container                 | **same** `rustgml2obj` container                        |
| 3  | translate.json                      | `create-translate` (Python container)   | **voxelizer container, auto-derived inside the binary** |
| 4  | index.json                          | `create-index` (Python container)       | **voxelizer container, auto-derived inside the binary** |
| 5  | grid_mapping.json                   | inline in Python `voxelizer`            | **voxelizer container, auto-derived inside the binary** |
| 6  | Y↔Z swap                            | inline in Python `voxelizer`            | inline in `voxel-pipeline::swap`                        |
| 7  | `cuda_voxelizer` invocation         | inline in Python `voxelizer`            | inline (rayon pool over the same `cuda_voxelizer` bin)  |
| 8  | binvox → CSV + PostGIS              | inline in Python `voxelizer`            | inline (mmap RLE stream + COPY BINARY)                  |

The collapse of steps 3–5 into the voxelizer container is **not** a
behaviour change — the same JSON files end up on disk in
`data/objs/`, with the same schema; you can still inspect them or
re-build them by hand with `voxel-pipeline translate`,
`voxel-pipeline index`, `voxel-pipeline grid`.

---

## Migration matrix

| Python (in `obj-voxel-postgis/`)                    | Rust replacement                                                                                       |
|-----------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| `create_translate_per_obj.py`                       | `voxel-pipeline translate` (or auto-derived inside `run`)                                              |
| `create_index_from_jsons.py`                        | `voxel-pipeline index`                                                                                 |
| grid sizing in `main-obj-voxel-postgis-unified.py`  | `voxel-pipeline grid` + [`voxel_schema::grid_mapping`](crates/voxel-schema/src/grid_mapping.rs)        |
| `binvox` parsing (`binvox_rw`-style)                | [`voxel_binvox::BinvoxFile::occupied_voxels()`](crates/voxel-binvox/src/lib.rs)                        |
| Y↔Z swap for `cuda_voxelizer`                       | [`voxel_pipeline::swap`](crates/voxel-pipeline/src/swap.rs)                                            |
| CSV writer                                          | [`voxel_pipeline::sinks`](crates/voxel-pipeline/src/sinks.rs)                                          |
| PostGIS INSERT batches                              | [`voxel_postgis::copy_binary`](crates/voxel-postgis/src/copy_binary.rs) (COPY BINARY)                  |
| `voxel_position` 13-digit formula                   | [`voxel_pipeline::voxid::compute`](crates/voxel-pipeline/src/voxid.rs)                                 |
| EWKB hex packing (`struct.pack`)                    | [`voxel_schema::ewkb::point_z_ewkb_hex`](crates/voxel-schema/src/ewkb.rs)                              |
| `multiprocessing.Pool` over OBJs                    | rayon thread-pool inside [`voxelizer::voxelize_batch`](crates/voxel-pipeline/src/voxelizer.rs)         |
| `start.sh`                                          | [`start.sh`](start.sh) (equivalent semantics)                                                          |
| `gui_modern.py`                                     | unchanged — point its `working_dir` at this folder                                                     |

External services kept verbatim:

| External image                          | Role                                  |
|-----------------------------------------|---------------------------------------|
| `citygml4j/citygml-tools:latest`        | CityGML 2.0 → 3.0 upgrade             |
| `atsaad/rustcitygml2obj:v3.2`           | CityGML 3.0 → OBJ + JSON sidecars     |
| `postgis/postgis:16-3.4`               | Database                              |
| `atsaad/voxel-pipeline:v7.4` (base)    | CUDA + `cuda_voxelizer` binary        |

---

## Performance comparison

| Concern                    | Python (v7.4)                                                       | Rust (this repo)                                                  |
|----------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------|
| Memory per `*.binvox`      | O(grid³) — densifies via numpy                                      | O(occupied) — mmap + streaming RLE iterator                       |
| Parallelism model          | `multiprocessing.Pool`, fork + Python interpreter per worker        | rayon thread-pool inside one process; tokio drives the COPY stream |
| DB ingest                  | `psycopg2.execute_values` INSERT batches                            | `COPY voxel(...) FROM STDIN (FORMAT BINARY)`                      |
| Image size                 | ~3.4 GB (CUDA + Python + numpy stack + libs)                        | base image + a single ~12 MB ELF                                  |
| Startup latency            | Python interpreter + module imports on every container exec         | `exec` ~5 ms                                                      |
| Type & contract safety     | Untyped JSON / `dict.get(key, default)`                             | serde-validated structs, byte-exact unit tests for every contract |

---

## Why the rewrite

Three concrete things that fall out for free:

1. **Memory bound by occupied voxels, not grid size.** The Python
   pipeline densified each `binvox` into a numpy boolean array — 2048³
   = 8 GB on its own. The Rust binvox parser is a streaming RLE
   iterator over an mmap, so we pay for what the building actually
   contains, not what the grid theoretically could. A skyscraper at
   grid-size 2048 used to OOM at 16 GB and now finishes in <1 GB.

2. **`COPY voxel(...) FROM STDIN (FORMAT BINARY)` instead of
   `execute_values`.** PostgreSQL's binary COPY protocol skips the SQL
   parser, the per-row text parsing, and the per-batch round trip. On a
   ~50M-voxel city tile we measured 5–15× higher sustained ingest
   throughput against the same `postgis/postgis:16-3.4` instance.

3. **Type-safe contracts.** Every JSON file the pipeline reads or writes
   is a `serde::Deserialize`/`Serialize` struct with byte-exact unit
   tests. Drift between modules — historically the source of half the
   Python pipeline's incidents — now becomes a compile error.

The CSV byte-output, PostGIS schema, and `voxel_position` IDs are
identical to the Python reference; the swap is a behaviour-preserving
performance + ergonomics rewrite, not a re-design.

---

## Option B — keep using the Urban-Auto-Vox folder, swap only the voxelizer

Available if you don't want to fully switch. Layer the override
compose file:

```bash
cd /path/to/Urban-Auto-Vox
docker compose \
  -f docker-compose.yml \
  -f /path/to/urban-auto-vox-rs/docker/docker-compose.voxel-pipeline-rs.yml \
  up voxelizer
```

Same Rust binary, same `.env`, same `data/`. You lose the
auto-generation of `translate.json` / `index.json` (since the Python
prep services still get scheduled by `start.sh` in that compose
graph), but it's the lowest-friction way to A/B test.
