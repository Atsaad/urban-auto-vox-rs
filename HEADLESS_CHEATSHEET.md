# urban-auto-vox-rs — headless cheat sheet

All knobs come from `.env` next to `docker-compose.yml`. The GUI just writes that file — these scripts read it directly.

## Single-run mode — `./start.sh`

Drops one set of inputs through the full pipeline, leaves PostGIS running.

```bash
# 1. configure once
cp .env.example .env
$EDITOR .env                    # set PIPELINE_VOXEL_SIZE, PIPELINE_OUTPUT_FORMAT, etc.

# 2. stage inputs (depends on CITYGML_INPUT_VERSION)
#    CITYGML_INPUT_VERSION=2.0  →  data/citygml2/*.gml
#    CITYGML_INPUT_VERSION=3.0  →  data/citygml3/*.gml   (citygml-tools step skipped)

# 3. run
./start.sh
```

Outputs: `data/objs/voxels_output.csv` + (if `PIPELINE_OUTPUT_FORMAT=postgis|both`) the `voxel` PostGIS table on host port `${POSTGRES_HOST_PORT:-5434}`.

## Batch mode — `./batch-process.sh`

Iterates over tile-folders **or** loose `*.gml` files under `BATCH_SOURCE_DIR`, one full pipeline run per tile.

```bash
# .env additions
BATCH_SOURCE_DIR=/path/to/tiles      # required — folders or *.gml
BATCH_OUTPUT_DIR=./output_batches    # default
BATCH_MAX_BATCHES=0                  # 0 = all
BATCH_AUTO_ZIP=true                  # ~98% smaller archives

./batch-process.sh                   # run
./batch-process.sh --summary         # progress so far
./batch-process.sh --clean           # wipe data/ work dirs
./batch-process.sh --reset-progress  # delete every tile_*.zip and start over
./batch-process.sh --help
```

Resume is automatic — tiles with an existing `tile_<name>.zip` (or `<name>.zip` from single mode) are skipped. Failures get one retry, then logged to `./batch_failed_tiles.txt`; per-tile logs land in `./batch_logs/`.

## Useful one-liners

```bash
# Inspect the DB
docker compose exec postgis psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"

# Tear everything down (keeps the postgis_data volume)
docker compose down

# Nuke everything including the DB volume
./clean-all.sh
```

## Common knobs (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `CITYGML_INPUT_VERSION` | `2.0` | `2.0` runs upgrade step; `3.0` skips it |
| `PIPELINE_VOXEL_SIZE` | `0.5` | Voxel edge length, metres |
| `PIPELINE_OUTPUT_FORMAT` | `csv` | `csv` \| `postgis` \| `both` |
| `PIPELINE_NUM_WORKERS` | `8` | Concurrent `cuda_voxelizer` jobs |
| `PIPELINE_DB_SRID` | `25832` | EPSG stamped on `voxel.vox_geom` |
| `RUST_LOG` | `info` | `error\|warn\|info\|debug\|trace` |
