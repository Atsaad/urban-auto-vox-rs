#!/usr/bin/env bash
# =============================================================================
# urban-auto-vox-rs — chunk-based batch processing.
#
# Iterates over chunk manifests under chunks/manifests/, runs the full
# Docker pipeline once per chunk, and stores:
#   - one concatenated voxel CSV per chunk in voxel_csvs/ (kept unzipped,
#     used for tensor-build)
#   - a zip archive per chunk in output_batches/ (per-building CSVs +
#     semantic JSONs + chunk-level sidecars)
#
# Each chunk = 5000 buildings sorted by gemeindeschluessel, produced by
# chunk_buildings.py. Resume is granular per chunk: skip if
# output_batches/<chunk_id>.zip already exists.
#
# Knobs (set in .env or env-vars):
#   PIPELINE_OUTPUT_FORMAT  csv | postgis | both     (csv recommended)
#   CHUNK_MANIFESTS_DIR     default: chunks/manifests
#   CHUNK_VOXEL_CSV_DIR     default: voxel_csvs
#   BATCH_OUTPUT_DIR        default: output_batches
#   CHUNK_PARALLEL_COPIES   default: 8 (parallel cp processes when staging)
#
# CLI:
#   ./chunk-process.sh                       run all remaining chunks
#   ./chunk-process.sh --start 17            start from chunk_00017
#   ./chunk-process.sh --start 17 --end 78   process the RANGE 00017..00078
#                                            (inclusive; resume-safe: already-done
#                                            chunks in the range are counted, not
#                                            redone, and the run STOPS at --end so it
#                                            never bleeds into the next district)
#   ./chunk-process.sh --start 17 --max 5    process up to 5 NEW chunks from 00017
#                                            (count-based cap; prefer --end for city
#                                            ranges — see the kreis-to-chunk table
#                                            in the project log)
#   ./chunk-process.sh --munich-only         shortcut for --start 17 --end 78
#   ./chunk-process.sh --dry-run             list what would be processed
#   ./chunk-process.sh --summary             show progress
#   ./chunk-process.sh --help
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Pull .env into the environment so PIPELINE_* / POSTGRES_* etc. are visible.
if [ -f .env ]; then
	set -a
	# shellcheck disable=SC1091
	source .env
	set +a
fi

# ---------------------------------------------------------------- config
MANIFESTS_DIR="${CHUNK_MANIFESTS_DIR:-chunks/manifests}"
VOXEL_CSV_DIR="${CHUNK_VOXEL_CSV_DIR:-voxel_csvs}"
OUTPUT_DIR="${BATCH_OUTPUT_DIR:-output_batches}"
PARALLEL_COPIES="${CHUNK_PARALLEL_COPIES:-8}"
CITYGML_INPUT_VERSION="${CITYGML_INPUT_VERSION:-2.0}"
POSTGRES_USER_VALUE="${POSTGRES_USER:-postgres}"
OUTPUT_FORMAT="${PIPELINE_OUTPUT_FORMAT:-csv}"

WORK_CITYGML2='./data/citygml2'
WORK_CITYGML3='./data/citygml3'
WORK_OBJS='./data/objs'

LOG_FILE='./chunk_processing.log'
FAILED_FILE='./chunk_failed.txt'
LOG_DIR='./batch_logs'
RUNTIME_FATAL_FILE='./chunk_runtime_fatal.txt'

MAX_CONSEC_FAILS="${CHUNK_MAX_CONSEC_FAILS:-3}"
PIPELINE_RETRIES="${CHUNK_PIPELINE_RETRIES:-1}"
RETRY_BACKOFF_SECS="${CHUNK_RETRY_BACKOFF_SECS:-10}"
REQUIRE_NVIDIA_SOCKET="${CHUNK_REQUIRE_NVIDIA_SOCKET:-1}"

START_AT=0
END_AT=0
MAX_CHUNKS=0
DRY_RUN=0
MUNICH_ONLY=0
START_SET=0
END_SET=0
MAX_SET=0

# ------------------------------------------------------------ arg parse
while [ $# -gt 0 ]; do
	case "$1" in
		--start)         START_AT="$2"; START_SET=1; shift 2 ;;
		--end)           END_AT="$2"; END_SET=1; shift 2 ;;
		--max)           MAX_CHUNKS="$2"; MAX_SET=1; shift 2 ;;
		--munich-only)   MUNICH_ONLY=1; shift ;;
		--dry-run)       DRY_RUN=1; shift ;;
		--summary)       MODE=summary; shift ;;
		--help|-h)
			sed -n '2,32p' "$0"; exit 0 ;;
		*)
			echo "ERROR: unknown arg '$1'. See --help." >&2; exit 2 ;;
	esac
done

# --munich-only sets defaults that explicit --start/--end/--max can still override.
# Munich = chunks 17..78; an --end bound is resume-safe (won't run past chunk 78).
if [ "$MUNICH_ONLY" -eq 1 ]; then
	[ "$START_SET" -eq 0 ] && START_AT=17
	[ "$END_SET"   -eq 0 ] && END_AT=78
fi

# A range end before the start is almost certainly a typo — fail fast.
if [ "$END_AT" -gt 0 ] && [ "$END_AT" -lt "$START_AT" ]; then
	echo "ERROR: --end ($END_AT) is before --start ($START_AT)." >&2; exit 2
fi

# --------------------------------------------------------------- helpers
log()       { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
ensure_dir(){ mkdir -p "$@"; }

clean_work_dirs() {
	for d in "$WORK_CITYGML2" "$WORK_CITYGML3" "$WORK_OBJS"; do
		find "$d" -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null || true
	done
}

is_chunk_processed() {
	local cid="$1"
	[ -f "$OUTPUT_DIR/${cid}.zip" ]
}

ensure_postgis_running() {
	log "Ensuring PostGIS is running..."
	local status
	status=$(docker compose -f ./docker-compose.yml ps postgis --format '{{.Status}}' 2>/dev/null || true)
	if [[ "$status" =~ [Uu]p ]]; then
		log "  PostGIS already running"
		return
	fi
	docker compose -f ./docker-compose.yml up -d postgis 2>&1 | tee -a "$LOG_FILE"
	for _ in $(seq 1 30); do
		if docker compose -f ./docker-compose.yml exec -T postgis \
				pg_isready -U "$POSTGRES_USER_VALUE" >/dev/null 2>&1; then
			log "  PostGIS is ready"; return
		fi
		sleep 2
	done
	log "  WARN: PostGIS health check timed out (continuing)"
}

preflight_runtime_checks() {
	if ! docker info >/dev/null 2>&1; then
		log "  FATAL: Docker daemon is not reachable."
		log "         Start/restart Docker and re-run this command."
		return 1
	fi

	if [ "$REQUIRE_NVIDIA_SOCKET" = "1" ] && [ ! -S /run/nvidia-persistenced/socket ]; then
		log "  FATAL: /run/nvidia-persistenced/socket is missing."
		log "         NVIDIA persistence daemon is not running; voxelizer container cannot start."
		return 1
	fi

	return 0
}

classify_pipeline_failure() {
	local logf="$1"
	if grep -qEi 'docker.sock.*connection reset by peer|error during connect: Get "http://%2Fvar%2Frun%2Fdocker.sock' "$logf"; then
		echo "docker-daemon-connection"
		return 0
	fi
	if grep -qEi 'nvidia-persistenced/socket|OCI runtime create failed|failed to create shim task|failed to create task for container' "$logf"; then
		echo "nvidia-runtime"
		return 0
	fi
	echo "pipeline-error"
}

# Stage a chunk's 5000 .gml files from manifest paths into the input dir.
# Uses Python for robust CSV parsing (gemeinde column has quoted commas).
stage_chunk_inputs() {
	local manifest="$1"
	local target_dir
	case "$CITYGML_INPUT_VERSION" in
		3|3.0) target_dir="$WORK_CITYGML3" ;;
		*)     target_dir="$WORK_CITYGML2" ;;
	esac

	python3 -c "
import csv, sys
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        print(r['gml_path'])
" "$manifest" | xargs -d '\n' -P "$PARALLEL_COPIES" -I{} cp {} "$target_dir/"

	local n; n=$(find "$target_dir" -maxdepth 1 -type f -name '*.gml' | wc -l)
	echo "$n"
}

run_pipeline_for_chunk() {
	local cid="$1"
	local logf="$LOG_DIR/${cid}.log"
	: > "$logf"
	local attempt=0
	local max_attempts=$((PIPELINE_RETRIES + 1))

	while [ "$attempt" -lt "$max_attempts" ]; do
		attempt=$((attempt + 1))
		if [ "$attempt" -gt 1 ]; then
			log "  retry $attempt/$max_attempts for $cid after ${RETRY_BACKOFF_SECS}s backoff"
			sleep "$RETRY_BACKOFF_SECS"
		fi

		if ! preflight_runtime_checks; then
			echo "runtime-preflight" > "$RUNTIME_FATAL_FILE"
			return 99
		fi

		if [ "$CITYGML_INPUT_VERSION" != "3" ] && [ "$CITYGML_INPUT_VERSION" != "3.0" ]; then
			if ! docker compose -f ./docker-compose.yml run --rm --no-deps citygml-tools \
					2>&1 | tee -a "$logf"; then
				if [ "$(classify_pipeline_failure "$logf")" != "pipeline-error" ]; then
					echo "$(classify_pipeline_failure "$logf")" > "$RUNTIME_FATAL_FILE"
					return 99
				fi
				continue
			fi
		fi

		if ! docker compose -f ./docker-compose.yml run --rm --no-deps rustgml2obj \
				2>&1 | tee -a "$logf"; then
			if [ "$(classify_pipeline_failure "$logf")" != "pipeline-error" ]; then
				echo "$(classify_pipeline_failure "$logf")" > "$RUNTIME_FATAL_FILE"
				return 99
			fi
			continue
		fi

		if ! docker compose -f ./docker-compose.yml up --force-recreate \
				--abort-on-container-exit --no-deps voxelizer \
				2>&1 | tee -a "$logf"; then
			if [ "$(classify_pipeline_failure "$logf")" != "pipeline-error" ]; then
				echo "$(classify_pipeline_failure "$logf")" > "$RUNTIME_FATAL_FILE"
				return 99
			fi
			continue
		fi

		return 0
	done

	return 1
}

# Produce the canonical per-chunk voxel CSV at $VOXEL_CSV_DIR/${cid}_voxels.csv.
# Voxelizer v0.3.x writes ONE combined CSV per chunk (named after some arbitrary
# building gmlid) — we just rename it. Older voxelizers wrote one CSV per
# building; this function still handles that case by concatenating.
concat_chunk_csv() {
	local cid="$1"
	local out="$VOXEL_CSV_DIR/${cid}_voxels.csv"
	mapfile -t csvs < <(find "$WORK_OBJS" -maxdepth 1 -type f -name '*.csv' | sort)
	local n=${#csvs[@]}

	if [ "$n" -eq 0 ]; then
		log "  WARN: no CSVs produced (PIPELINE_OUTPUT_FORMAT=postgis?)"
		return 1
	fi

	if [ "$n" -eq 1 ]; then
		# Single combined CSV (v0.3.x) — just copy. ~50% disk savings vs concat.
		cp "${csvs[0]}" "$out"
	else
		# Many per-building CSVs (older voxelizer) — concatenate with single header.
		local first=1
		for csv in "${csvs[@]}"; do
			if [ "$first" -eq 1 ]; then
				head -n 1 "$csv" > "$out"
				first=0
			fi
			tail -n +2 "$csv" >> "$out"
		done
	fi
	wc -l "$out" | awk '{print $1}'
}

# Build $VOXEL_CSV_DIR/${cid}_grid_sizes.csv from the voxel CSV.
# Output cols: gmlid, voxel_count, w_cells, d_cells, h_cells, max_cells, fits_64
# Tensor-build uses this to filter buildings that exceed the 64³ grid.
build_grid_sizes_csv() {
	local cid="$1"
	local src="$VOXEL_CSV_DIR/${cid}_voxels.csv"
	local out="$VOXEL_CSV_DIR/${cid}_grid_sizes.csv"

	if [ ! -f "$src" ]; then
		log "  WARN: $src missing — skipping grid_sizes.csv"
		return 0
	fi

	VOXEL_SIZE_M="${PIPELINE_VOXEL_SIZE:-0.5}" python3 - "$src" "$out" << 'PY' || return 1
import csv, os, sys
from collections import defaultdict

src, out = sys.argv[1], sys.argv[2]
vs = float(os.environ.get("VOXEL_SIZE_M", "0.5"))

mins = defaultdict(lambda: [float('inf')] * 3)
maxs = defaultdict(lambda: [float('-inf')] * 3)
counts = defaultdict(int)

with open(src, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        g = row['building_gmlid']
        x, y, z = float(row['x']), float(row['y']), float(row['z'])
        mn, mx = mins[g], maxs[g]
        if x < mn[0]: mn[0] = x
        if y < mn[1]: mn[1] = y
        if z < mn[2]: mn[2] = z
        if x > mx[0]: mx[0] = x
        if y > mx[1]: mx[1] = y
        if z > mx[2]: mx[2] = z
        counts[g] += 1

with open(out, 'w', newline='') as f:
    w = csv.writer(f, lineterminator='\n')   # unix line endings (awk-friendly)
    w.writerow(['gmlid', 'voxel_count', 'w_cells', 'd_cells', 'h_cells', 'max_cells', 'fits_64'])
    for g in sorted(mins):
        mn, mx = mins[g], maxs[g]
        w_cells = int((mx[0] - mn[0]) / vs) + 1
        d_cells = int((mx[1] - mn[1]) / vs) + 1
        h_cells = int((mx[2] - mn[2]) / vs) + 1
        max_c = max(w_cells, d_cells, h_cells)
        w.writerow([g, counts[g], w_cells, d_cells, h_cells, max_c, 'yes' if max_c <= 64 else 'no'])
PY

	if [ -f "$out" ]; then
		local nb nfit
		nb=$(awk 'NR>1' "$out" | wc -l)
		nfit=$(awk -F',' 'NR>1 && $NF=="yes"' "$out" | wc -l)
		log "  built $(basename "$out") ($nb buildings, $nfit fit 64³)"
	fi
}

# Zip per-chunk artefacts to output_batches/chunk_NNNNN.zip.
# Includes:
#   - 3 chunk-level JSON sidecars (index, translate, grid_mapping)
#   - canonical voxel CSV (single copy — NOT duplicated)
#   - grid_sizes.csv (per-building bbox dims, for tensor-build filtering)
#   - manifest CSV (reproducibility)
#   - pipeline log
# Excludes:
#   - 37K per-surface JSONs (redundant — info is in index.json + CSV's surface_class)
#   - any other raw CSV from data/objs (avoid the v0.3.x duplicate)
#   - .obj files (intermediate, recomputable from CityGML)
#   - .binvox files (redundant — CSV is the source of truth for training/tensor-export;
#     binvox is only the GPU-output intermediate consumed by the decoder)
archive_chunk() {
	local cid="$1"
	local stage="$OUTPUT_DIR/${cid}"
	ensure_dir "$stage"

	# 1. Chunk-level JSON sidecars only (skip the 37K per-surface JSONs)
	for f in index.json translate.json grid_mapping.json; do
		[ -f "$WORK_OBJS/$f" ] && cp "$WORK_OBJS/$f" "$stage/"
	done

	# 2. Canonical voxel CSV (single copy from voxel_csvs/)
	cp "$VOXEL_CSV_DIR/${cid}_voxels.csv" "$stage/" 2>/dev/null || true

	# 3. Per-chunk grid sizes (for tensor-build filtering)
	cp "$VOXEL_CSV_DIR/${cid}_grid_sizes.csv" "$stage/" 2>/dev/null || true

	# 4. Manifest (reproducibility — original gmlid list for this chunk)
	cp "$MANIFESTS_DIR/${cid}.csv" "$stage/${cid}_manifest.csv" 2>/dev/null || true

	# 5. Per-chunk pipeline log
	cp "$LOG_DIR/${cid}.log" "$stage/pipeline_log.txt" 2>/dev/null || true

	( cd "$OUTPUT_DIR" && zip -r -q "${cid}.zip" "${cid}" ) || {
		log "  WARN: zip failed for $cid"
		return 1
	}

	local before after
	before=$(du -sh "$stage" 2>/dev/null | cut -f1)
	after=$(du -sh  "$OUTPUT_DIR/${cid}.zip" 2>/dev/null | cut -f1)
	rm -rf "$stage"
	log "  archived $cid: $before -> $after"
}

# --------------------------------------------------------- summary mode
if [ "${MODE:-}" = "summary" ]; then
	total=$(find "$MANIFESTS_DIR" -maxdepth 1 -name 'chunk_*.csv' 2>/dev/null | wc -l)
	done_n=$(find "$OUTPUT_DIR" -maxdepth 1 -name 'chunk_*.zip' 2>/dev/null | wc -l)
	failed_n=0
	[ -f "$FAILED_FILE" ] && failed_n=$(wc -l < "$FAILED_FILE")
	echo "Total chunks    : $total"
	echo "Completed (zip) : $done_n"
	echo "Failed          : $failed_n"
	echo "Remaining       : $((total - done_n))"
	exit 0
fi

# --------------------------------------------------------------- main
log
log "============================================="
log "  urban-auto-vox-rs — chunk processing"
log "============================================="

if [ ! -d "$MANIFESTS_DIR" ]; then
	log "ERROR: manifest dir not found: $MANIFESTS_DIR (run chunk_buildings.py first)"
	exit 1
fi

if [ "$OUTPUT_FORMAT" != "csv" ]; then
	log "WARN: PIPELINE_OUTPUT_FORMAT is '$OUTPUT_FORMAT'. Set it to 'csv' in .env"
	log "      if you want CSV-only output (PostGIS writes add ~30%+ overhead)."
fi

mapfile -t MANIFESTS < <(find "$MANIFESTS_DIR" -maxdepth 1 -name 'chunk_*.csv' | sort)
total=${#MANIFESTS[@]}
log "Discovered $total chunk manifests"
log "Output format       : $OUTPUT_FORMAT"
log "Voxel CSV dir       : $VOXEL_CSV_DIR"
log "Archive dir         : $OUTPUT_DIR"
log "Start at            : chunk_$(printf '%05d' "$START_AT")"
if [ "$END_AT" -gt 0 ]; then
	log "End at              : chunk_$(printf '%05d' "$END_AT") (inclusive range bound)"
else
	log "End at              : none (run to last manifest)"
fi
log "Max new-work cap    : $MAX_CHUNKS (0 = no cap)"
log "Pipeline retries    : $PIPELINE_RETRIES"
log "Max consec fails    : $MAX_CONSEC_FAILS"

ensure_dir "$OUTPUT_DIR" "$VOXEL_CSV_DIR" "$LOG_DIR" \
           "$WORK_CITYGML2" "$WORK_CITYGML3" "$WORK_OBJS"

if [ "$OUTPUT_FORMAT" = "postgis" ] || [ "$OUTPUT_FORMAT" = "both" ]; then
	ensure_postgis_running
fi

: > "$FAILED_FILE"
rm -f "$RUNTIME_FATAL_FILE"

processed=0 skipped=0 failed=0
consec_failed=0
for manifest in "${MANIFESTS[@]}"; do
	cid="$(basename "$manifest" .csv)"
	idx="${cid#chunk_}"
	idx_num=$((10#$idx))   # strip leading zeros

	if [ "$idx_num" -lt "$START_AT" ]; then
		continue
	fi

	# Range upper bound. Manifests are sorted ascending, so once we pass --end
	# we are done — regardless of how many chunks in the range were already
	# processed. This is what keeps a resumed run inside the requested range.
	if [ "$END_AT" -gt 0 ] && [ "$idx_num" -gt "$END_AT" ]; then
		log "Reached --end $(printf 'chunk_%05d' "$END_AT"), stopping."
		break
	fi

	if is_chunk_processed "$cid"; then
		skipped=$((skipped + 1))
		continue
	fi

	if [ "$MAX_CHUNKS" -gt 0 ] && [ "$processed" -ge "$MAX_CHUNKS" ]; then
		log
		log "Reached --max $MAX_CHUNKS, stopping."
		break
	fi

	log
	log "============================================="
	log "  $cid  ($((processed + 1)) this run, $skipped skipped so far)"
	log "============================================="

	if [ "$DRY_RUN" -eq 1 ]; then
		log "  [dry-run] would process $cid"
		processed=$((processed + 1))
		continue
	fi

	t0=$(date +%s)

	clean_work_dirs
	staged=$(stage_chunk_inputs "$manifest")
	log "  staged $staged .gml files"

	if ! run_pipeline_for_chunk "$cid"; then
		rc=$?
		log "  FAIL: pipeline error for $cid"
		echo "$cid" >> "$FAILED_FILE"
		failed=$((failed + 1))
		consec_failed=$((consec_failed + 1))
		processed=$((processed + 1))
		if [ "$rc" -eq 99 ]; then
			reason="runtime-preflight"
			[ -f "$RUNTIME_FATAL_FILE" ] && reason="$(cat "$RUNTIME_FATAL_FILE")"
			log "  FATAL: detected runtime infrastructure error ($reason)."
			log "         Stopping early to avoid cascading failures."
			break
		fi
		if [ "$consec_failed" -ge "$MAX_CONSEC_FAILS" ]; then
			log "  FATAL: reached $consec_failed consecutive failures."
			log "         Stopping early. Tune with CHUNK_MAX_CONSEC_FAILS in .env if needed."
			break
		fi
		continue
	fi
	consec_failed=0

	if ! lines=$(concat_chunk_csv "$cid"); then
		log "  FAIL: csv concat for $cid"
		echo "$cid" >> "$FAILED_FILE"
		failed=$((failed + 1))
		processed=$((processed + 1))
		continue
	fi
	log "  wrote $VOXEL_CSV_DIR/${cid}_voxels.csv  ($lines lines incl. header)"

	# Per-building grid-size index (used at tensor build to filter > 64³ buildings).
	# Reads the voxel CSV once; non-fatal if it fails (CSV still archives).
	build_grid_sizes_csv "$cid" || log "  WARN: grid_sizes.csv build failed for $cid"

	if ! archive_chunk "$cid"; then
		log "  WARN: archive failed for $cid (voxel CSV is still saved)"
	fi

	t1=$(date +%s)
	log "  $cid done in $((t1 - t0))s"
	processed=$((processed + 1))
done

clean_work_dirs

log
log "============================================="
log "  chunk run complete"
log "============================================="
log "Processed (this run): $processed"
log "Skipped (already done): $skipped"
log "Failed: $failed"
[ "$failed" -gt 0 ] && log "See $FAILED_FILE for failed chunk IDs."
