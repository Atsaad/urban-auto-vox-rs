#!/usr/bin/env bash
# =============================================================================
# build_tensors.py recipes — named builds, no flag-memory required.
#
# Usage:
#   bash tensorbuilding/recipes.sh <recipe_name> [extra build_tensors.py flags…]
#   bash tensorbuilding/recipes.sh list           # show available recipes
#
# Each recipe sets the flag block that defines a §12 pattern (or my smoke set)
# and forwards any extra arguments straight to build_tensors.py, so I can
# tune e.g. --max-height or --shard-size at the call site without editing
# this file.
#
# Run from the repo root. Recipes assume the default voxel_csvs/ and
# tensorbuilding/shards/ paths. See claude.md §24 for the full manual.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$REPO_ROOT"

PY="$SCRIPT_DIR/.venv/bin/python"
BT="$SCRIPT_DIR/build_tensors.py"

# Default city list for the 5 thesis cities (§12 Pattern C).
THESIS_CITIES="munich,augsburg,nuernberg,wuerzburg,regensburg"

# ---------------------------------------------------------------- recipes
# Each recipe is a function. Add new ones below and to the `list` recipe.

smoke() {
	# Pattern A — 1 000 Munich residential gabled. ~1 min. Sanity check.
	"$PY" "$BT" \
		--cities munich --function residential --roof gabled \
		--per-city-cap 1000 --max-height 16 --tag munich_smoke "$@"
}

munich_res_gabled() {
	# Pattern F phase 1 — 2 000 Munich residential gabled. The original smoke set.
	"$PY" "$BT" \
		--cities munich --function residential --roof gabled \
		--per-city-cap 2000 --max-height 16 --tag munich_res_gabled "$@"
}

munich_res_gabled_all() {
	# Pattern B (homogeneous) — every Munich residential-gabled fits_64.
	# ~ tens of thousands of buildings, ~10–15 min.
	"$PY" "$BT" \
		--cities munich --function residential --roof gabled \
		--cap 0 --max-height 16 --tag munich_res_gabled_all "$@"
}

munich_all() {
	# Pattern B (full Munich, no function/roof filter). Phase B baseline.
	# ~100 k buildings after height + fits_64 filters, ~15–25 min.
	"$PY" "$BT" \
		--cities munich --cap 0 --max-height 16 --tag munich_all "$@"
}

5cities_balanced() {
	# Pattern C — 2 000 per thesis city. 10 k total, ~10 min. The current set.
	"$PY" "$BT" \
		--cities "$THESIS_CITIES" \
		--per-city-cap 2000 --max-height 16 --tag 5cities_balanced "$@"
}

5cities_all() {
	# Pattern C-wide — every fits_64 building in the 5 thesis cities.
	# ~500 k buildings, ~1–2 h.
	"$PY" "$BT" \
		--cities "$THESIS_CITIES" \
		--cap 0 --max-height 16 --tag 5cities_all "$@"
}

5cities_res_gabled() {
	# Pattern F homogeneous, expanded to all 5 thesis cities.
	# ~150 k residential-gabled buildings, ~30 min.
	"$PY" "$BT" \
		--cities "$THESIS_CITIES" --function residential --roof gabled \
		--cap 0 --max-height 16 --tag 5cities_res_gabled "$@"
}

pattern_e_train() {
	# Pattern E train split — München + Augsburg + Nürnberg.
	"$PY" "$BT" \
		--cities munich,augsburg,nuernberg \
		--cap 0 --max-height 16 --tag pattern_e_train "$@"
}

pattern_e_val() {
	# Pattern E val split — Würzburg held out.
	"$PY" "$BT" \
		--cities wuerzburg \
		--cap 0 --max-height 16 --tag pattern_e_val "$@"
}

pattern_e_test() {
	# Pattern E test split — Regensburg sacred. DO NOT inspect during training.
	"$PY" "$BT" \
		--cities regensburg \
		--cap 0 --max-height 16 --tag pattern_e_test "$@"
}

list() {
	cat <<-'EOF'
		Available recipes:
		  smoke                    Pattern A — 1 000 Munich res-gabled (~1 min)
		  munich_res_gabled        Pattern F phase 1 — 2 000 (~3 min)
		  munich_res_gabled_all    Pattern B homogeneous — all Munich res-gabled
		  munich_all               Pattern B — all Munich (~100 k, ~15–25 min)
		  5cities_balanced         Pattern C — 2 k/city, 10 k total (~10 min)
		  5cities_all              Pattern C-wide — ~500 k (~1–2 h)
		  5cities_res_gabled       Pattern F homogeneous, 5 cities (~150 k, ~30 min)
		  pattern_e_train          Pattern E train (München + Augsburg + Nürnberg)
		  pattern_e_val            Pattern E val (Würzburg)
		  pattern_e_test           Pattern E test (Regensburg — sacred)
		Run:   bash tensorbuilding/recipes.sh <name> [extra flags…]
	EOF
}

# ---------------------------------------------------------------- dispatch
if [ "$#" -lt 1 ]; then
	list
	exit 0
fi

cmd="$1"; shift
case "$cmd" in
	-h|--help|help)  list ;;
	*)
		if declare -f "$cmd" > /dev/null; then "$cmd" "$@"
		else echo "ERROR: unknown recipe '$cmd'. Run with no args to list." >&2; exit 2
		fi ;;
esac
