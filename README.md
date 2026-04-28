# urban-auto-vox-rs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.85-orange.svg)](rust-toolchain.toml)
[![Docker Compose](https://img.shields.io/badge/docker--compose-ready-blue.svg)](docker-compose.yml)

An automated pipeline for converting 3D city models (CityGML) into voxel
representations, with output to CSV and PostGIS. The entire workflow runs as
a single `docker compose up` invocation on any Linux host with an NVIDIA GPU.

```
CityGML  ──→  OBJ + JSON  ──→  CUDA voxelization  ──→  CSV / PostGIS
```

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Quick start](#quick-start)
3. [Pipeline overview](#pipeline-overview)
4. [Repository layout](#repository-layout)
5. [Configuration](#configuration)
6. [GUI](#gui)
7. [Batch processing](#batch-processing)
8. [CLI reference](#cli-reference)
9. [Data flow and outputs](#data-flow-and-outputs)
10. [Database schema](#database-schema)
11. [Performance](#performance)
12. [Testing](#testing)
13. [Troubleshooting](#troubleshooting)
14. [License](#license)

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Linux host** | Tested on Ubuntu 22.04 / 24.04 |
| **NVIDIA GPU** | Any CUDA-capable card (tested with RTX 4090) |
| **NVIDIA driver** | `nvidia-smi` must succeed on the host |
| **NVIDIA Container Toolkit** | [Install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| **Docker Engine** | 20.10+ with Compose V2 |

Run `./setup-gpu.sh` to verify all prerequisites in one go.

---

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/Atsaad/urban-auto-vox-rs.git
cd urban-auto-vox-rs

# 2. Verify GPU + Docker setup (one-time)
./setup-gpu.sh

# 3. Drop your CityGML files into the input directory
cp /path/to/city_tile.gml data/citygml2/

# 4. Configure (optional — defaults are sensible)
cp .env.example .env
$EDITOR .env

# 5. Run the full pipeline
./start.sh

# 6. Inspect results
ls data/objs/voxels_output.csv
docker compose exec postgis psql -U postgres -d voxel_db \
  -c "SELECT count(*) FROM voxel;"
```

For CityGML 3.0 input, drop files into `data/citygml3/` instead and set
`CITYGML_INPUT_VERSION=3.0` in `.env`.

For processing many tiles, see [Batch processing](#batch-processing).

---

## Pipeline overview

```
   ┌──────────────────────────┐
   │  data/citygml2/*.gml     │   user input (CityGML 2.0)
   └────────────┬─────────────┘
                │   (skipped if CITYGML_INPUT_VERSION=3.0)
                ▼
   ┌──────────────────────────┐
   │  citygml-tools (Java)    │   CityGML 2.0 → 3.0 upgrade
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │  rustgml2obj             │   CityGML 3.0 → one OBJ per building
   │                          │   + one JSON sidecar per surface
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────────────────────────────────────┐
   │  voxel-pipeline (Rust)                                   │
   │                                                          │
   │  1. Generate translate.json, index.json, grid_mapping    │
   │  2. Y↔Z axis swap per OBJ                               │
   │  3. cuda_voxelizer → *.binvox (GPU-accelerated)          │
   │  4. Decode binvox → CSV rows + PostGIS COPY BINARY       │
   │                                                          │
   │  rayon thread-pool for CPU work; tokio async for DB I/O  │
   └────────────┬─────────────────────────────────────────────┘
                ▼
   ┌──────────────────────────┐    ┌──────────────────────────┐
   │  voxels_output.csv       │    │  PostGIS                 │
   │                          │    │  voxel / object /        │
   │                          │    │  object_class tables     │
   └──────────────────────────┘    └──────────────────────────┘
```

Each block maps to a service in [`docker-compose.yml`](docker-compose.yml).
Everything runs as a Docker container — the host only needs Docker and the
NVIDIA Container Toolkit. The first `./start.sh` invocation builds the
voxelizer image once (~3 min) and caches it.

---

## Repository layout

```
urban-auto-vox-rs/
│
├── docker-compose.yml             # Full pipeline stack (4 services)
├── .env.example                   # All configuration knobs, documented
├── start.sh                       # End-to-end orchestration (single tile)
├── batch-process.sh               # End-to-end orchestration (batch of tiles)
├── setup-gpu.sh                   # NVIDIA + Docker prerequisites check
├── clean-all.sh                   # Full cleanup of generated data
├── gui_modern.py                  # Desktop GUI (CustomTkinter)
├── gui-modern.sh                  # GUI launcher (creates venv on first run)
│
├── data/                          # Mounted into containers
│   ├── citygml2/                  # Input: CityGML 2.0 files
│   ├── citygml3/                  # Intermediate: upgraded CityGML 3.0
│   └── objs/                      # Output: OBJ, sidecars, CSV, binvox
│
├── docker/
│   ├── Dockerfile.voxel-pipeline-rs
│   └── docker-compose.voxel-pipeline-rs.yml
│
├── crates/                        # Rust workspace
│   ├── voxel-schema/              # Typed data contracts (translate,
│   │                              #   index, grid_mapping, surface, EWKB)
│   ├── voxel-binvox/              # mmap-backed streaming binvox parser
│   ├── voxel-postgis/             # COPY BINARY writer + DDL
│   └── voxel-pipeline/            # CLI binary (5 subcommands)
│
├── Cargo.toml                     # Workspace manifest
└── rust-toolchain.toml            # Pinned to Rust 1.85
```

---

## Configuration

A single `.env` file drives the entire stack. Copy `.env.example` to `.env`
and edit as needed — all values have sensible defaults.

### CityGML input

| Variable                | Default | Purpose |
|-------------------------|---------|---------|
| `CITYGML_INPUT_VERSION` | `2.0`   | `2.0` runs the CityGML upgrade step; `3.0` skips it |

### Database

| Variable             | Default    | Purpose |
|----------------------|------------|---------|
| `POSTGRES_DB`        | `voxel_db` | Database name |
| `POSTGRES_USER`      | `postgres` | Database user |
| `POSTGRES_PASSWORD`  | `postgres` | Database password |
| `POSTGRES_HOST_PORT` | `5434`     | Host port mapped to container port 5432 |

### Pipeline

| Variable                  | Default   | Purpose |
|---------------------------|-----------|---------|
| `PIPELINE_VOXEL_SIZE`     | `0.5` (m) | Target voxel edge length. Grid size per OBJ: `clamp(ceil(max_dim / size), 8, 2048)` |
| `PIPELINE_DB_SRID`        | `25832`   | EPSG code for all `Point Z` geometries |
| `PIPELINE_NUM_WORKERS`    | `8`       | Concurrent `cuda_voxelizer` invocations |
| `PIPELINE_OUTPUT_FORMAT`  | `csv`     | `csv`, `postgis`, or `both` |
| `PIPELINE_DB_BATCH_BYTES` | `8388608` | COPY BINARY flush threshold (bytes) |

### Advanced (set automatically by Docker Compose)

| Variable                  | Default | Purpose |
|---------------------------|---------|---------|
| `PIPELINE_INPUT_DIR`      | `/app/data` | OBJ directory inside the container |
| `PIPELINE_DB_HOST`        | `postgis`  | PostGIS hostname (compose service name) |
| `PIPELINE_DB_PORT`        | `5432`     | Internal database port |
| `PIPELINE_CUDA_VOXELIZER` | `/app/cuda_voxelizer/build/cuda_voxelizer` | Path inside the container |
| `VOXELIZE_TIMEOUT`        | `300` (s)  | Per-file `cuda_voxelizer` timeout |
| `RUST_LOG`                | `info`     | Log level: `error`, `warn`, `info`, `debug`, `trace` |

If `PIPELINE_DB_HOST` is empty, the pipeline runs in **CSV-only mode**
regardless of `PIPELINE_OUTPUT_FORMAT` — no database connection required.

---

## GUI

A desktop GUI ([`gui_modern.py`](gui_modern.py)) provides a graphical
interface to configure and run the pipeline. It writes `.env` and calls
`./start.sh` or `./batch-process.sh` under the hood.

### Launch

```bash
./gui-modern.sh
```

The launcher creates a Python virtual environment and installs
[CustomTkinter](https://github.com/TomSchimansky/CustomTkinter) on first
run (~30 s). Subsequent launches are instant.

### Features

- Configure all pipeline parameters (voxel size, workers, output format,
  database credentials, processing mode)
- Single-tile and batch processing modes
- Real-time log output from Docker containers
- Archive management for completed tiles

### How it works

```
GUI panels  ──→  writes .env  ──→  start.sh / batch-process.sh
                                        │
                                  docker compose up
                                        │
                                  voxel-pipeline binary
                                  (reads PIPELINE_* env vars via clap)
```

The [`clap`](https://docs.rs/clap) derive macros on the Rust binary declare
`env = "PIPELINE_…"` for every flag, so the GUI → `.env` → Docker Compose →
Rust binary chain requires zero glue code.

---

## Batch processing

For processing multiple CityGML tiles, use
[`./batch-process.sh`](batch-process.sh).

### Features

- **Resume** — tiles with existing output archives are skipped
- **Retry** — transient failures get one automatic retry with a clean state
- **Auto-zip** — per-tile results are compressed immediately after success
- **Hot database** — PostGIS stays up between tiles for continuous ingestion
- **Skip-and-continue** — one failing tile does not stop the batch
- **Mixed input** — tile directories and loose `.gml` files in the same source
- **Per-tile logs** — `batch_logs/batch_<n>_<tile>.log` for each tile

### Configuration

| Variable           | Default            | Purpose |
|--------------------|--------------------|---------|
| `BATCH_SOURCE_DIR` | *(required)*       | Directory with tile folders or loose `.gml` files |
| `BATCH_OUTPUT_DIR` | `./output_batches` | Where per-tile archives are written |
| `BATCH_MAX_BATCHES`| `0`                | `0` = process all; `N` = stop after N tiles |
| `BATCH_AUTO_ZIP`   | `true`             | Compress completed tiles automatically |

All `PIPELINE_*` and `POSTGRES_*` variables are honoured per tile.

### Supported input layouts

```
# Tile directories (one or more .gml files per folder):
/data/source/
  ├── 32_682_5334/
  │     ├── lod1.gml
  │     └── lod2.gml
  └── 32_682_5335/
        └── lod2.gml

# Loose files:
/data/source/
  ├── tile_a.gml
  └── tile_b.gml
```

### Usage

```bash
# Configure
cat >> .env <<'EOF'
BATCH_SOURCE_DIR=/data/source
BATCH_MAX_BATCHES=0
BATCH_AUTO_ZIP=true
EOF

# Run
./batch-process.sh

# Check progress without re-running
./batch-process.sh --summary

# Clean in-progress work (does NOT touch output archives)
./batch-process.sh --clean

# Force a full re-run
./batch-process.sh --reset-progress
```

### Output per tile

```
tile_<name>/
  ├── pipeline_log.txt
  ├── translate.json
  ├── index.json
  ├── grid_mapping.json
  ├── voxels_output.csv
  └── binvox/
        ├── building_a.obj_256.binvox
        └── ...
```

When using `postgis` or `both` output format, voxels from all tiles are
appended to the same `voxel` table — queryable as a single dataset after
the batch completes.

---

## CLI reference

```
voxel-pipeline <SUBCOMMAND>

Subcommands:
  translate   Generate translate.json (per-OBJ + global bounding box)
  index       Generate index.json (semantic mapping from sidecar JSONs)
  grid        Generate grid_mapping.json (adaptive CUDA grid sizes)
  voxelize    Run cuda_voxelizer + ingest (CSV / PostGIS / both)
  run         Full pipeline: translate → index → grid → voxelize
```

`run` is the default entry point used by Docker Compose. Individual
subcommands are available for debugging or re-running a single stage.

All subcommands auto-generate missing intermediate files
(`translate.json`, `index.json`, `grid_mapping.json`) on demand.

### CLI flags and env-var mapping

Every flag has an environment variable fallback. CLI flags take precedence
over environment variables, which take precedence over defaults.

| Flag                       | Env var                   | Default |
|----------------------------|---------------------------|---------|
| `--input-dir`              | `PIPELINE_INPUT_DIR`      | *(required)* |
| `--cuda-voxelizer`         | `PIPELINE_CUDA_VOXELIZER` | *(required)* |
| `--target-voxel-size`      | `PIPELINE_VOXEL_SIZE`     | `0.5` |
| `--srid`                   | `PIPELINE_DB_SRID`        | `25832` |
| `--workers`                | `PIPELINE_NUM_WORKERS`    | `8` |
| `--voxelize-timeout-secs`  | `VOXELIZE_TIMEOUT`        | `300` |
| `--output-format`          | `PIPELINE_OUTPUT_FORMAT`  | `csv` |
| `--output-csv`             | —                         | `<input>/voxels_output.csv` |
| `--db-flush-bytes`         | `PIPELINE_DB_BATCH_BYTES` | `8 MiB` |
| `--db-host` / `--db-port`  | `PIPELINE_DB_HOST/_PORT`  | empty / `5432` |
| `--db-name`                | `PIPELINE_DB_NAME`        | empty |
| `--db-username`            | `PIPELINE_DB_USERNAME`    | empty |
| `--db-password`            | `PIPELINE_DB_PASSWORD`    | empty |

### Building from source

```bash
# Native build
cargo build --release --bin voxel-pipeline
./target/release/voxel-pipeline run \
  --input-dir /path/to/objs \
  --cuda-voxelizer /opt/cuda_voxelizer/build/cuda_voxelizer

# Container build
docker build -f docker/Dockerfile.voxel-pipeline-rs -t voxel-pipeline-rs .
```

`rust-toolchain.toml` pins **Rust 1.85**; rustup fetches it automatically.

---

## Data flow and outputs

| File / Directory                  | Created by           | Purpose |
|-----------------------------------|----------------------|---------|
| `data/citygml3/*.gml`            | citygml-tools        | Upgraded CityGML 3.0 |
| `data/objs/*.obj`                | rustgml2obj          | One OBJ mesh per building |
| `data/objs/*.json` (sidecars)    | rustgml2obj          | Surface semantic metadata |
| `data/objs/translate.json`       | voxel-pipeline       | Per-OBJ + global bounding box |
| `data/objs/index.json`           | voxel-pipeline       | Semantic mapping (GML ID → surface type) |
| `data/objs/grid_mapping.json`    | voxel-pipeline       | Per-file adaptive CUDA grid size |
| `data/objs/*.binvox`             | cuda_voxelizer       | Binary voxel grids |
| `data/objs/voxels_output.csv`    | voxel-pipeline       | Row-per-voxel output |
| PostGIS `voxel` / `object` / `object_class` | voxel-pipeline | Spatial database tables |

### CSV format

```
voxel_position,vox_geom,x,y,z,element_gmlid,surface_gmlid,building_gmlid,object_type
```

- `voxel_position` — 13-digit unique voxel ID ([`voxid::compute`](crates/voxel-pipeline/src/voxid.rs))
- `vox_geom` — hex-encoded EWKB `Point Z` with SRID ([`ewkb::point_z_ewkb_hex`](crates/voxel-schema/src/ewkb.rs))
- `x, y, z` — voxel centre coordinates in the configured SRID
- `element_gmlid`, `surface_gmlid`, `building_gmlid` — CityGML 3.0 identifier hierarchy (see below)

---

## Database schema

Applied automatically on first PostGIS write. See
[`schema::apply_schema`](crates/voxel-postgis/src/schema.rs).

Two tables, decoupled by design:

* **`building`** — created by the pipeline but **not written to it**. Loaded
  once, out-of-band, from `building_metadata.csv` via `psql \COPY`.
  Holds the building-level conditioning attributes (function, roof type,
  height, storeys, address, etc.) used to build the diffusion model's
  conditioning vectors at training time.
* **`voxel`** — the only table the pipeline writes. Flat and denormalised:
  one row per occupied voxel with `(building_gmlid, surface_gmlid,
  surface_class, x, y, z, vox_geom)`. No FK to `building` — ingestion
  order is unconstrained, and a 300M-row COPY pays no per-row FK
  validation cost.

`surface_class` is a `SMALLINT` mapping of the CityGML thematic surface:

| Class | Meaning |
|------:|---------|
| 0 | Unknown / air |
| 1 | WallSurface |
| 2 | RoofSurface |
| 3 | GroundSurface |
| 4 | OuterCeilingSurface |
| 5 | ClosureSurface |

```sql
CREATE TABLE building (
    building_gmlid       TEXT PRIMARY KEY,
    tile_id              TEXT,
    function_code        TEXT,
    function_label       TEXT,
    roof_type_code       TEXT,
    roof_type_label      TEXT,
    measured_height      REAL,
    storeys_above_ground SMALLINT,
    storeys_source       TEXT,
    year_of_construction SMALLINT,
    gemeindeschluessel   TEXT,
    hoehe_dach           REAL,
    hoehe_grund          REAL,
    niedrigste_traufe    REAL,
    city                 TEXT,
    postal_code          TEXT,
    street_name          TEXT,
    house_number         TEXT,
    source               TEXT
);
CREATE INDEX idx_building_gemeindeschluessel ON building (gemeindeschluessel);
CREATE INDEX idx_building_function_code      ON building (function_code);

CREATE TABLE voxel (
    building_gmlid TEXT             NOT NULL,
    surface_gmlid  TEXT,
    surface_class  SMALLINT         NOT NULL,
    x              DOUBLE PRECISION NOT NULL,
    y              DOUBLE PRECISION NOT NULL,
    z              DOUBLE PRECISION NOT NULL,
    vox_geom       GEOMETRY(PointZ, <PIPELINE_DB_SRID>)
);
CREATE INDEX idx_voxel_building_gmlid ON voxel (building_gmlid);
CREATE INDEX idx_voxel_geom           ON voxel USING GIST (vox_geom);
```

Voxel rows are written via `COPY voxel FROM STDIN (FORMAT BINARY)` for
high-throughput ingestion. Load `building` once after schema apply:

```bash
psql -h <host> -p <port> -U <user> -d <db> \
  -c "\COPY building FROM 'building_metadata.csv' CSV HEADER"
```

---

## Performance

- **Memory-efficient voxel decoding** — binvox files are parsed as a
  streaming RLE iterator over memory-mapped I/O. Memory usage is
  proportional to the number of occupied voxels, not the grid volume.
  A building at grid-size 2048 processes in under 1 GB.

- **High-throughput database ingestion** — PostgreSQL's binary COPY protocol
  bypasses SQL parsing and per-row text conversion. On a ~50M-voxel city
  tile, sustained ingest throughput is 5-15x higher than row-based INSERT.

- **Parallel voxelization** — rayon thread-pool parallelises CPU-bound work
  across OBJ files; a single tokio task drains the result channel into
  the database COPY stream.

- **Minimal container footprint** — the entire Rust pipeline is a single
  ~12 MB statically-linked binary added to the CUDA base image.

Benchmarked on an RTX 4090 with ~10k buildings per tile.

---

## Testing

```bash
cargo test --workspace    # unit tests, no GPU or database required
```

Tests cover the byte-level data contracts:

- EWKB encoding round-trip ([`ewkb::tests`](crates/voxel-schema/src/ewkb.rs))
- `voxel_position` ID formula ([`voxid::tests`](crates/voxel-pipeline/src/voxid.rs))
- binvox header parsing + RLE streaming ([`binvox::tests`](crates/voxel-binvox/src/lib.rs))
- JSON schema validation (index, translate, grid_mapping)

---

## Troubleshooting

### Container exits with code 1 immediately

Run with debug logging:

```bash
RUST_LOG=debug ./start.sh 2>&1 | tee run.log
```

Common causes:

- **No OBJ files** — `CITYGML_INPUT_VERSION` doesn't match the files in `data/`.
  Use `2.0` for files in `data/citygml2/`, `3.0` for `data/citygml3/`.
- **No GPU access** — run `./setup-gpu.sh` to verify the NVIDIA runtime.

### `/run/nvidia-persistenced/socket: no such file or directory`

The NVIDIA Container Toolkit requires the persistence daemon socket on the
host. If the daemon isn't running, container init fails before the pipeline
starts:

```
Error response from daemon: failed to create task for container: ...
open /run/nvidia-persistenced/socket: no such file or directory
```

Fix:

```bash
sudo systemctl start nvidia-persistenced
sudo systemctl restart docker
```

`systemctl enable nvidia-persistenced` may report *"no installation config"* —
this is normal. Modern NVIDIA packages activate the daemon via udev on GPU
detection, not through a standard systemd `[Install]` section. If a reboot
doesn't bring the socket back automatically, force persistence mode:

```bash
sudo nvidia-smi -pm 1
```

### `cuda_voxelizer not found`

The runtime image includes `cuda_voxelizer` at
`/app/cuda_voxelizer/build/cuda_voxelizer`. If you customised the base image,
set `PIPELINE_CUDA_VOXELIZER` to the correct path inside the container.

### PostGIS port collision

The host port defaults to `5434`. If already in use, change
`POSTGRES_HOST_PORT` in `.env` and restart the stack.

### `voxel` table is empty after a run

- Check `PIPELINE_OUTPUT_FORMAT` — the default is `csv` (database writes
  disabled). Set to `postgis` or `both`.
- Verify `PIPELINE_DB_HOST` is set — if empty, the pipeline falls back to
  CSV-only mode silently.

### `translate.json missing` warning

Harmless — the pipeline regenerates it automatically. Use
`voxel-pipeline translate` to generate it explicitly if needed.

---

## License

Released under the [MIT License](LICENSE). Copyright 2026 Abdullah Saad.
