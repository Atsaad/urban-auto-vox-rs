#!/usr/bin/env bash
# =============================================================================
# urban-auto-vox-rs — automated batch processing.
#
# Iterates over CityGML tiles under $BATCH_SOURCE_DIR, running the full
# Docker pipeline (citygml-tools → rustgml2obj → voxelizer) once per
# tile, and stores per-tile artefacts under $BATCH_OUTPUT_DIR.
#
# Features (parity with Urban-Auto-Vox/batch-process.sh):
#   • resume — skips tiles whose result archive already exists
#   • single retry with backoff on transient failures
#   • optional auto-zip (+ ~98 % space saving)
#   • PostGIS kept hot between tiles when output mode needs it
#   • per-tile log under ./batch_logs/, summary at the end
#   • supports both tile-folders and loose *.gml files in $BATCH_SOURCE_DIR
#
# All knobs come from environment variables (set them in .env, the GUI
# also writes them when its `Processing mode → Batch` switch is on):
#
#   BATCH_SOURCE_DIR    Directory with tile folders OR loose *.gml files
#   BATCH_OUTPUT_DIR    Where to write per-tile archives (default: ./output_batches)
#   BATCH_MAX_BATCHES   0 = process all (default), N = stop after N tiles
#   BATCH_AUTO_ZIP      true (default) | false
#
# Pipeline knobs (PIPELINE_VOXEL_SIZE, PIPELINE_OUTPUT_FORMAT, …) flow
# through unchanged — see .env.example.
#
# CLI:
#   ./batch-process.sh                   run with .env / env-vars
#   ./batch-process.sh --help            show help
#   ./batch-process.sh --summary         print current progress
#   ./batch-process.sh --clean           clear data/ work dirs only
#   ./batch-process.sh --reset-progress  delete every tile_*.zip and start fresh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Pull .env into the environment so PIPELINE_* / BATCH_* / POSTGRES_* are
# all visible to docker compose AND to the variables below.
if [ -f .env ]; then
	set -a
	# shellcheck disable=SC1091
	source .env
	set +a
fi

# ---------------------------------------------------------------- config
SOURCE_DIR="${BATCH_SOURCE_DIR:-}"
MAX_BATCHES="${BATCH_MAX_BATCHES:-0}"
AUTO_ZIP="${BATCH_AUTO_ZIP:-true}"
FINAL_OUTPUT_DIR="${BATCH_OUTPUT_DIR:-./output_batches}"
CITYGML_INPUT_VERSION="${CITYGML_INPUT_VERSION:-2.0}"
POSTGRES_USER_VALUE="${POSTGRES_USER:-postgres}"

WORK_CITYGML2='./data/citygml2'
WORK_CITYGML3='./data/citygml3'
WORK_OBJS='./data/objs'

LOG_FILE='./batch_processing.log'
FAILED_FILE='./batch_failed_tiles.txt'

# --------------------------------------------------------------- helpers
log()       { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
ensure_dir(){ mkdir -p "$@"; }

# Match the GUI's single-mode archive name (<tile>.zip) and the batch
# convention (tile_<tile>.zip / tile_<tile>/) to support resume across
# both modes.
is_tile_processed() {
	local tile_name="$1"
	[ -f "$FINAL_OUTPUT_DIR/${tile_name}.zip" ]      || \
	[ -f "$FINAL_OUTPUT_DIR/tile_${tile_name}.zip" ] || \
	[ -d "$FINAL_OUTPUT_DIR/tile_${tile_name}" ]
}

clean_work_dirs() {
	log "Cleaning data/ work dirs"
	for d in "$WORK_CITYGML2" "$WORK_CITYGML3" "$WORK_OBJS"; do
		find "$d" -mindepth 1 ! -name '.gitkeep' -delete 2>/dev/null || true
	done
}

# Copy a tile's GML files into the appropriate input dir based on
# CITYGML_INPUT_VERSION. Echoes the count for the caller to capture.
copy_tile_inputs() {
	local source="$1" tile="$2"
	local target_dir
	case "$CITYGML_INPUT_VERSION" in
		3|3.0) target_dir="$WORK_CITYGML3" ;;
		*)     target_dir="$WORK_CITYGML2" ;;
	esac

	local count=0
	if [ -f "$source" ]; then
		cp "$source" "$target_dir/"; count=1
	else
		while IFS= read -r gml; do
			cp "$gml" "$target_dir/"; count=$((count + 1))
		done < <(find "$source" -type f -name '*.gml')
	fi
	log "  copied $count GML file(s) for tile '$tile' into $target_dir"
	echo "$count"
}

ensure_postgis_running() {
	log "Ensuring PostGIS is running…"
	local status
	status=$(docker compose -f ./docker-compose.yml ps postgis --format '{{.Status}}' 2>/dev/null || true)
	if [[ "$status" =~ [Uu]p ]]; then
		log "  ✓ PostGIS already running"
		return
	fi
	docker compose -f ./docker-compose.yml up -d postgis 2>&1 | tee -a "$LOG_FILE"
	for _ in $(seq 1 30); do
		if docker compose -f ./docker-compose.yml exec -T postgis \
				pg_isready -U "$POSTGRES_USER_VALUE" >/dev/null 2>&1; then
			log "  ✓ PostGIS is ready"; return
		fi
		sleep 2
	done
	log "  ⚠ PostGIS health check timed out (continuing anyway)"
}

# Run the full per-tile pipeline through docker compose. We
# `--force-recreate` the voxelizer container so a previous "exited 0"
# state doesn't make compose treat the next run as a no-op.
run_pipeline_for_tile() {
	local batch="$1" tile="$2"
	local batch_log="./batch_logs/batch_${batch}_${tile}.log"

	log "  starting pipeline for $tile"

	# Step 1 (skipped automatically for CityGML 3.0 input).
	if [ "$CITYGML_INPUT_VERSION" != "3" ] && [ "$CITYGML_INPUT_VERSION" != "3.0" ]; then
		docker compose -f ./docker-compose.yml run --rm --no-deps citygml-tools \
				2>&1 | tee -a "$batch_log" || return 1
	fi

	# Step 2 — CityGML → OBJ.
	docker compose -f ./docker-compose.yml run --rm --no-deps rustgml2obj \
			2>&1 | tee -a "$batch_log" || return 1

	# Step 3 — voxelize + ingest.
	docker compose -f ./docker-compose.yml up --force-recreate \
			--abort-on-container-exit --no-deps voxelizer \
			2>&1 | tee -a "$batch_log" || return 1

	log "  ✅ pipeline completed for $tile"
}

# Move per-tile artefacts out of data/objs into a per-tile subfolder
# under $FINAL_OUTPUT_DIR. Optionally zip and delete the folder.
save_tile_results() {
	local batch="$1" tile="$2"
	local out_dir="$FINAL_OUTPUT_DIR/tile_${tile}"

	log "  saving artefacts for $tile → $out_dir"
	ensure_dir "$out_dir"

	[ -f "./batch_logs/batch_${batch}_${tile}.log" ] && \
		cp "./batch_logs/batch_${batch}_${tile}.log" "$out_dir/pipeline_log.txt"

	# binvox files (large; mv to free disk for next tile)
	ensure_dir "$out_dir/binvox"
	find "$WORK_OBJS" -maxdepth 1 -type f -name '*.binvox' \
		-exec mv {} "$out_dir/binvox/" \; 2>/dev/null || true

	# JSON metadata + CSV output
	for f in translate.json index.json grid_mapping.json voxels_output.csv; do
		[ -f "$WORK_OBJS/$f" ] && cp "$WORK_OBJS/$f" "$out_dir/"
	done

	if [ "$AUTO_ZIP" = "true" ]; then
		log "  compressing $out_dir"
		local zip_name="tile_${tile}.zip"
		( cd "$FINAL_OUTPUT_DIR" && zip -r -q "$zip_name" "tile_${tile}" ) || {
			log "  ⚠ zip failed — keeping uncompressed folder"
			return
		}
		local before after
		before=$(du -sh "$out_dir" 2>/dev/null | cut -f1)
		after=$(du -sh  "$FINAL_OUTPUT_DIR/$zip_name" 2>/dev/null | cut -f1)
		rm -rf "$out_dir"
		log "  📦 $before → $after"
	fi
}

generate_summary() {
	local summary="$FINAL_OUTPUT_DIR/processing_summary.txt"
	local zips dirs failed_n

	zips=$(find "$FINAL_OUTPUT_DIR" -maxdepth 1 -name 'tile_*.zip'  2>/dev/null | wc -l)
	dirs=$(find "$FINAL_OUTPUT_DIR" -maxdepth 1 -type d -name 'tile_*' 2>/dev/null | wc -l)
	failed_n=0
	[ -f "$FAILED_FILE" ] && [ -s "$FAILED_FILE" ] && failed_n=$(wc -l < "$FAILED_FILE")

	{
		echo "============================================="
		echo "  urban-auto-vox-rs — batch summary"
		echo "============================================="
		echo
		echo "Completed at        : $(date)"
		echo "Source directory    : $SOURCE_DIR"
		echo "Output directory    : $FINAL_OUTPUT_DIR"
		echo "Voxel size (m)      : ${PIPELINE_VOXEL_SIZE:-0.5}"
		echo "Output format       : ${PIPELINE_OUTPUT_FORMAT:-csv}"
		echo
		echo "Completed (zipped)       : $zips"
		echo "Completed (uncompressed) : $dirs"
		echo "Total completed          : $((zips + dirs))"
		echo "Failed tiles             : $failed_n"
		echo
		[ "$failed_n" -gt 0 ] && echo "See $FAILED_FILE for the failed tile list."
		echo
		echo "Logs : ./batch_logs/"
	} | tee "$summary"

	log "Summary saved to $summary"
}

# ----------------------------------------------------------------- main
main() {
	log
	log "============================================="
	log "  urban-auto-vox-rs — batch processing"
	log "============================================="

	if [ -z "$SOURCE_DIR" ]; then
		log "ERROR — BATCH_SOURCE_DIR not set."
		log "        Add it to .env or export it before running."
		exit 1
	fi
	if [ ! -d "$SOURCE_DIR" ]; then
		log "ERROR — source directory not found: $SOURCE_DIR"
		exit 1
	fi

	log "Source              : $SOURCE_DIR"
	log "Output              : $FINAL_OUTPUT_DIR"
	log "Max batches         : $MAX_BATCHES (0 = all)"
	log "Auto-zip            : $AUTO_ZIP"
	log "CityGML version     : $CITYGML_INPUT_VERSION"
	log "Pipeline output fmt : ${PIPELINE_OUTPUT_FORMAT:-csv}"
	log "Voxel size          : ${PIPELINE_VOXEL_SIZE:-0.5} m"

	ensure_dir "$FINAL_OUTPUT_DIR" "$WORK_CITYGML2" "$WORK_CITYGML3" "$WORK_OBJS" ./batch_logs

	# Discover tiles: subdirectories first, then loose .gml files in $SOURCE_DIR
	mapfile -t TILES < <(find "$SOURCE_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
	mapfile -t LOOSE < <(find "$SOURCE_DIR" -maxdepth 1 -type f -name '*.gml' | sort)
	if [ "${#LOOSE[@]}" -gt 0 ]; then
		log "Found ${#LOOSE[@]} loose GML file(s) — each treated as its own tile"
		TILES+=("${LOOSE[@]}")
	fi

	local total=${#TILES[@]}
	if [ "$total" -eq 0 ]; then
		log "ERROR — no tiles found under $SOURCE_DIR"
		exit 1
	fi

	# Resume tally
	local already=0
	for tf in "${TILES[@]}"; do
		local n; n="$(basename "$tf")"; n="${n%.gml}"
		is_tile_processed "$n" && already=$((already + 1))
	done
	[ "$already" -gt 0 ] && log "Resuming — $already / $total tile(s) already done, will skip"

	# Pre-warm the database if we'll be writing to it
	local fmt="${PIPELINE_OUTPUT_FORMAT:-csv}"
	if [ "$fmt" = "postgis" ] || [ "$fmt" = "both" ]; then
		ensure_postgis_running
	fi

	: > "$FAILED_FILE"

	local batch=0 processed=0 skipped=0 failed=0
	for tf in "${TILES[@]}"; do
		local tile; tile="$(basename "$tf")"; tile="${tile%.gml}"

		if is_tile_processed "$tile"; then
			skipped=$((skipped + 1))
			continue
		fi

		batch=$((batch + 1))

		if [ "$MAX_BATCHES" -gt 0 ] && [ "$processed" -ge "$MAX_BATCHES" ]; then
			log
			log "⏹  Reached BATCH_MAX_BATCHES=$MAX_BATCHES — stopping."
			break
		fi

		log
		log "═════════════════════════════════════════════"
		log "Tile $((skipped + processed + failed + 1))/$total — $tile (batch #$batch)"
		log "═════════════════════════════════════════════"

		clean_work_dirs

		if [ "$(copy_tile_inputs "$tf" "$tile")" -eq 0 ]; then
			log "  ⚠ no GML files in $tile — skipping"
			skipped=$((skipped + 1))
			continue
		fi

		if run_pipeline_for_tile "$batch" "$tile"; then
			save_tile_results "$batch" "$tile"
			processed=$((processed + 1))
			log "  ✅ $tile completed ($processed processed, $failed failed)"
		else
			log "  ⚠ first attempt failed — retrying in 5 s"
			sleep 5
			clean_work_dirs
			copy_tile_inputs "$tf" "$tile" >/dev/null
			if run_pipeline_for_tile "$batch" "$tile"; then
				save_tile_results "$batch" "$tile"
				processed=$((processed + 1))
				log "  ✅ $tile completed on retry ($processed processed, $failed failed)"
			else
				failed=$((failed + 1))
				echo "$tile" >> "$FAILED_FILE"
				log "  ❌ $tile FAILED after retry — moving on"
			fi
		fi

		clean_work_dirs
	done

	log
	log "============================================="
	log "  batch processing complete"
	log "============================================="
	log "  processed : $processed"
	log "  skipped   : $skipped"
	log "  failed    : $failed"
	log "  total     : $total"
	log "============================================="

	generate_summary

	if [ "$failed" -gt 0 ]; then
		log "⚠ $failed tile(s) failed — see $FAILED_FILE. Re-run to retry; they are not marked as processed."
	fi
}

# ----------------------------------------------------------------- CLI
case "${1:-}" in
	-h|--help)
		cat <<-EOF
		urban-auto-vox-rs — batch processing.

		Usage: $0 [--help|--clean|--reset-progress|--summary]

		Required env vars:
		  BATCH_SOURCE_DIR     directory with tile folders or loose *.gml

		Optional env vars:
		  BATCH_OUTPUT_DIR     default: ./output_batches
		  BATCH_MAX_BATCHES    default: 0 (= all)
		  BATCH_AUTO_ZIP       default: true
		  CITYGML_INPUT_VERSION default: 2.0   (2.0|3.0)
		  PIPELINE_*           every voxel-pipeline knob is passed through

		Set them in .env or export them before running. The GUI
		writes them automatically when "Processing mode → Batch"
		is selected.
		EOF
		;;
	--clean)            clean_work_dirs ;;
	--reset-progress)
		echo "Removing every tile_* output under $FINAL_OUTPUT_DIR …"
		rm -rf "${FINAL_OUTPUT_DIR:?}"/tile_* 2>/dev/null || true
		rm -f  "$FAILED_FILE"
		echo "Done. Next run will reprocess everything."
		;;
	--summary)
		if [ -d "$FINAL_OUTPUT_DIR" ]; then
			z=$(find "$FINAL_OUTPUT_DIR" -maxdepth 1 -name 'tile_*.zip'    2>/dev/null | wc -l)
			d=$(find "$FINAL_OUTPUT_DIR" -maxdepth 1 -type d -name 'tile_*' 2>/dev/null | wc -l)
			echo "Completed: $((z + d))  (zipped: $z, uncompressed: $d)"
			[ -f "$FAILED_FILE" ] && [ -s "$FAILED_FILE" ] && \
				echo "Failed   : $(wc -l < "$FAILED_FILE")"
		else
			echo "No output directory yet ($FINAL_OUTPUT_DIR does not exist)."
		fi
		;;
	"") main ;;
	*)  echo "Unknown flag: $1   (try --help)"; exit 1 ;;
esac
