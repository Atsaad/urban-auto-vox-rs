#!/usr/bin/env bash
# =============================================================================
# Nuclear cleanup — removes every container, image, volume, and build
# cache touched by this compose project. The PostGIS database disk
# is included; back it up first if you care about the data.
# =============================================================================

# No `set -e` — we want every step to attempt even if an earlier one fails.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if ! docker ps >/dev/null 2>&1; then
	echo "ERROR — current user has no Docker permissions."
	echo "        Try: sg docker -c './clean-all.sh'"
	exit 1
fi

echo "═════════════════════════════════════════════════════"
echo "  This will remove:"
echo "    • all compose containers (incl. PostGIS)"
echo "    • the postgis_data volume (database wiped)"
echo "    • all dangling Docker images and build cache"
echo "═════════════════════════════════════════════════════"
read -r -p "Continue? (y/N) " -n 1 reply
echo
if [[ ! $reply =~ ^[Yy]$ ]]; then
	echo "Cancelled."
	exit 0
fi

echo "[1/4] Stopping compose stack…"
docker compose -f ./docker-compose.yml down -v --remove-orphans 2>/dev/null || true

if docker ps -q --filter "name=voxel_postgis" | grep -q .; then
	echo "      voxel_postgis still running — force stop"
	docker stop voxel_postgis 2>/dev/null || true
	docker rm   -f voxel_postgis 2>/dev/null || true
fi

echo "[2/4] Pruning dangling images + volumes…"
docker system prune -a --volumes -f 2>/dev/null || true

echo "[3/4] Pruning build cache…"
docker builder prune -a -f 2>/dev/null || true

echo "[4/4] Verifying…"
remaining=$(docker ps -q --filter "name=voxel_" 2>/dev/null)
if [ -n "$remaining" ]; then
	echo "      Force-removing leftover containers"
	docker rm -f $remaining 2>/dev/null || true
fi

echo
echo "═════════════════════════════════════════════════════"
echo "  Cleanup done. Rebuild with:  ./start.sh"
echo "═════════════════════════════════════════════════════"
