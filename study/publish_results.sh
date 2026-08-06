#!/usr/bin/env bash
# Rebuild the Pillar G results page and publish it.
#
# Why this exists: the results page ships the answer key baked in
# (encrypted). Stimulus filenames are content hashes, so ANY re-render of
# study/docs/img/ produces a new set of names and silently invalidates the
# page — it still loads every response from Supabase, then fails the
# `p.key[img]` lookup on all of them and reports "Unrecognised images".
# That is what happened on 2026-08-05: the page was built at 20:32 and the
# stimuli were regenerated at 21:23-21:29. See claude.md §88.
#
# Run this after any stimulus rebuild. It prompts for the results-page
# passphrase twice via getpass — never echoed, never stored, never passed
# on a command line.
#
# It only ever READS Supabase. It never writes or deletes responses.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

export SUPABASE_URL=https://gctypkighhmblxbxdckv.supabase.co
export SUPABASE_PUBLISHABLE=sb_publishable_oZtTcRTl4h84LZQyPQyiHg_DUtLbG3m

PY=model/.venv/bin/python

# --- 1. the answer key must describe exactly the stimuli that are live ---
"$PY" - <<'PYEOF'
import json, pathlib, sys
key = set(json.load(open("study/answer_key.json")))
img = {p.name for p in pathlib.Path("study/docs/img").glob("*.png")}
if key != img:
    sys.exit(f"answer key and docs/img disagree "
             f"({len(key-img)} key-only, {len(img-key)} img-only). "
             f"Re-run make_stimuli.py before publishing.")
print(f"[1/3] answer key matches docs/img — {len(key)} stimuli")
PYEOF

# --- 2. rebuild the page (interactive passphrase) ---
echo "[2/3] rebuilding study/docs/results.html"
"$PY" study/build_results_page.py

# --- 3. publish to the public study repo ---
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone --quiet --depth 1 git@github.com:Atsaad/voxel-roof-study.git "$TMP/site"
cp study/docs/results.html "$TMP/site/results.html"

# That repo is public. The answer key and the dashboard are the study's
# blinding; if either reaches it, every response collected so far is void.
if find "$TMP/site" -path "$TMP/site/.git" -prune -o \
        \( -iname '*answer*' -o -iname 'dashboard*' \) -print | grep -q .; then
  echo "REFUSING TO PUSH — answer key or dashboard found in the deploy tree" >&2
  exit 1
fi

cd "$TMP/site"
git add results.html
if git diff --cached --quiet; then
  echo "[3/3] results.html already current — nothing to publish"
  exit 0
fi
git commit -qm "Rebuild results page against the current stimulus set"
git push -q origin HEAD
echo "[3/3] published — https://atsaad.github.io/voxel-roof-study/results.html"
