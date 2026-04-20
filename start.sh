#!/usr/bin/env bash
# =============================================================================
# urban-auto-vox-rs — full-pipeline orchestration.
#
# Equivalent to Urban-Auto-Vox/start.sh but driving the Rust voxelizer.
#
#   data/citygml2/*.gml  ─►  citygml-tools  ─►  data/citygml3/*.gml
#   data/citygml3/*.gml  ─►  rustgml2obj   ─►  data/objs/*.obj + JSON
#   data/objs/           ─►  voxelizer     ─►  data/objs/voxels_output.csv
#                                              + PostGIS voxel table
#
# Honors the same .env file the Urban-Auto-Vox GUI writes — drop it next
# to this script and the GUI will work unchanged (point its working_dir
# at this folder).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PIPELINE_START_TS="$(date +%s)"
STEP_UPGRADE_SEC=0
STEP_OBJ_SEC=0
STEP_VOXEL_SEC=0

format_duration() {
	local total="$1"
	local h=$(( total / 3600 ))
	local m=$(( (total % 3600) / 60 ))
	local s=$(( total % 60 ))
	printf '%02dh:%02dm:%02ds' "$h" "$m" "$s"
}

cleanup_non_postgis() {
	# Keep PostGIS running for quick iterative runs; remove transient workers.
	docker compose -f ./docker-compose.yml rm -fsv voxelizer rustgml2obj citygml-tools >/dev/null 2>&1 || true
}

trap cleanup_non_postgis EXIT

if [ -f .env ]; then
	set -a
	# shellcheck disable=SC1091
	source .env
	set +a
fi

CITYGML_INPUT_VERSION="${CITYGML_INPUT_VERSION:-2.0}"
POSTGRES_USER_VALUE="${POSTGRES_USER:-postgres}"
EXTERNAL_PORT="${POSTGRES_HOST_PORT:-5434}"

count_gml_files() {
	find "$1" -maxdepth 1 -type f -name '*.gml' 2>/dev/null | wc -l | tr -d ' '
}

run_step() {
	local service="$1" label="$2"
	echo
	echo "─────────────────────────────────────────────────────"
	echo "Step: $label"
	echo "─────────────────────────────────────────────────────"
	docker compose -f ./docker-compose.yml run --rm --no-deps "$service"
}

wait_for_postgis() {
	for _ in $(seq 1 30); do
		if docker compose -f ./docker-compose.yml exec -T postgis \
				pg_isready -U "$POSTGRES_USER_VALUE" >/dev/null 2>&1; then
			return 0
		fi
		sleep 2
	done
	return 1
}

echo "═════════════════════════════════════════════════════"
echo "  urban-auto-vox-rs — pipeline starting"
echo "═════════════════════════════════════════════════════"
echo
echo "Input mode : CityGML ${CITYGML_INPUT_VERSION}"
case "$CITYGML_INPUT_VERSION" in
	3|3.0) echo "Input dir  : data/citygml3/*.gml" ;;
	*)     echo "Input dir  : data/citygml2/*.gml" ;;
esac
echo "Voxel size : ${PIPELINE_VOXEL_SIZE:-0.5} m"
echo "Workers    : ${PIPELINE_NUM_WORKERS:-8}"
echo "Output     : data/objs/voxels_output.csv"
case "${PIPELINE_OUTPUT_FORMAT:-csv}" in
	postgis|both) echo "             + PostGIS voxel table (${POSTGRES_DB:-voxel_db}@${EXTERNAL_PORT})" ;;
esac
echo

case "$CITYGML_INPUT_VERSION" in
	3|3.0)
		count=$(count_gml_files './data/citygml3')
		if [ "$count" -eq 0 ]; then
			echo "ERROR — no CityGML 3.0 files in data/citygml3/*.gml"; exit 1
		fi
		;;
	*)
		count=$(count_gml_files './data/citygml2')
		if [ "$count" -eq 0 ]; then
			echo "ERROR — no CityGML 2.0 files in data/citygml2/*.gml"; exit 1
		fi
		;;
esac

echo "Starting PostGIS…"
docker compose -f ./docker-compose.yml up -d postgis
if ! wait_for_postgis; then
	echo "ERROR — PostGIS did not become ready in time."; exit 1
fi

case "$CITYGML_INPUT_VERSION" in
	3|3.0)
		echo "Skipping CityGML 2.0→3.0 upgrade (input is already CityGML 3.0)."
		;;
	*)
		step_start="$(date +%s)"
		run_step 'citygml-tools' 'CityGML 2.0 → 3.0 upgrade'
		STEP_UPGRADE_SEC=$(( $(date +%s) - step_start ))
		;;
esac

step_start="$(date +%s)"
run_step 'rustgml2obj' 'CityGML 3.0 → OBJ + JSON sidecars'
STEP_OBJ_SEC=$(( $(date +%s) - step_start ))

obj_count=$(find ./data/objs -maxdepth 1 -type f -name '*.obj' 2>/dev/null | wc -l | tr -d ' ')
if [ "$obj_count" -eq 0 ]; then
	echo "ERROR — no OBJ files were produced in data/objs/."
	echo "        Check that CITYGML_INPUT_VERSION matches your input files."
	exit 1
fi

echo
echo "─────────────────────────────────────────────────────"
echo "Step: voxelize + ingest (Rust, GPU)"
echo "─────────────────────────────────────────────────────"
step_start="$(date +%s)"
# Always refresh voxelizer from registry and recreate the container so
# newly pushed images/logging changes are picked up immediately.
docker compose -f ./docker-compose.yml pull voxelizer
docker compose -f ./docker-compose.yml up --force-recreate --no-deps --abort-on-container-exit voxelizer
STEP_VOXEL_SEC=$(( $(date +%s) - step_start ))

TOTAL_SEC=$(( $(date +%s) - PIPELINE_START_TS ))

CSV_PATH="./data/objs/voxels_output.csv"
CSV_ROWS_MSG="n/a"
if [ -f "$CSV_PATH" ]; then
	# Exclude header row when reporting voxel rows.
	line_count=$(wc -l < "$CSV_PATH" | tr -d ' ')
	if [ "$line_count" -gt 0 ]; then
		CSV_ROWS_MSG=$(( line_count - 1 ))
	fi
fi

echo
echo "═════════════════════════════════════════════════════"
echo "  Pipeline complete."
echo "═════════════════════════════════════════════════════"
echo
echo "Outputs:"
echo "  • data/objs/voxels_output.csv"
echo "  • PostGIS at host port ${EXTERNAL_PORT}"
echo
echo "Performance summary:"
echo "  • Total pipeline time : $(format_duration "$TOTAL_SEC") (${TOTAL_SEC}s)"
if [ "$STEP_UPGRADE_SEC" -gt 0 ]; then
	echo "  • CityGML upgrade     : $(format_duration "$STEP_UPGRADE_SEC") (${STEP_UPGRADE_SEC}s)"
fi
echo "  • OBJ generation      : $(format_duration "$STEP_OBJ_SEC") (${STEP_OBJ_SEC}s)"
echo "  • Voxelize + ingest   : $(format_duration "$STEP_VOXEL_SEC") (${STEP_VOXEL_SEC}s)"
echo "  • CSV voxel rows      : ${CSV_ROWS_MSG}"
echo
echo "Cleanup:"
echo "  • Non-PostGIS containers removed (PostGIS kept running)."
echo
echo "Inspect the database:"
echo "  docker compose exec postgis psql -U ${POSTGRES_USER_VALUE} -d ${POSTGRES_DB:-voxel_db}"
