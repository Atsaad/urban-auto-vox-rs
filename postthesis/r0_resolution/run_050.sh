#!/usr/bin/env bash
# Full 0.5 m re-voxelisation of the same 10,000-building manifest, so the
# throughput table has a second resolution that is comparable to 0.25 m.
cd "$(dirname "$0")"
python3 -I memsample.py mem_050.csv 2 >/dev/null 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
python3 -I -u revoxelise.py --voxel 0.5 \
  --manifest ../../tensorbuilding/shards/manifest.csv \
  --out out_050 --work work_050
