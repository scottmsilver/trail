#!/usr/bin/env bash
# Seed the canonical shared cache by COPYING existing populated caches from the
# main checkout (non-destructive: the main checkout keeps its own copies).
# Idempotent: skips anything already present in the shared dir.
#
# The shared cache lets every dev instance reuse the (expensive) OSM/DEM data so
# OSM is fetched once, ever. See docs/multi-instance.md.
set -euo pipefail

SHARED="${TRAIL_SHARED_DIR:-$HOME/development/trail-shared}"
MAIN="${TRAIL_MAIN_CHECKOUT:-$HOME/development/trail}"

mkdir -p "$SHARED"/{osm,http,dem_data,tile_cache,path_cache_v2,logs}

copy_dir_once() { # $1 = src dir, $2 = dest dir  (copies contents, once)
  local src="$1" dest="$2"
  if [[ -d "$src" && -z "$(ls -A "$dest" 2>/dev/null || true)" ]]; then
    echo "copying $src/* -> $dest"
    shopt -s dotglob nullglob
    cp -a "$src"/* "$dest"/ 2>/dev/null || true
    shopt -u dotglob nullglob
  else
    echo "skip $src (missing, or dest already populated)"
  fi
}

# osmnx Overpass cache (the big one): *.json in $MAIN/backend/cache
if [[ -d "$MAIN/backend/cache" && -z "$(ls -A "$SHARED/osm" 2>/dev/null || true)" ]]; then
  echo "copying osmnx json cache -> $SHARED/osm"
  find "$MAIN/backend/cache" -maxdepth 1 -name '*.json' -exec cp -a {} "$SHARED/osm"/ \;
else
  echo "skip osm cache (missing, or $SHARED/osm already populated)"
fi

# DEM py3dep/HyRiver sqlite (if present) -> $SHARED/http
if [[ -f "$MAIN/backend/cache/aiohttp_cache.sqlite" && ! -f "$SHARED/http/aiohttp_cache.sqlite" ]]; then
  echo "copying DEM http cache -> $SHARED/http"
  cp -a "$MAIN/backend/cache/aiohttp_cache.sqlite" "$SHARED/http/"
fi

copy_dir_once "$MAIN/dem_data"               "$SHARED/dem_data"
copy_dir_once "$MAIN/backend/tile_cache"     "$SHARED/tile_cache"
copy_dir_once "$MAIN/backend/path_cache_v2"  "$SHARED/path_cache_v2"

cat > "$SHARED/trail-shared.env" <<EOF
# Shared cache locations for all trail dev instances. Source before launching.
export OSM_CACHE_DIR="$SHARED/osm"
export HYRIVER_CACHE_NAME="$SHARED/http/aiohttp_cache.sqlite"
export TRAIL_DEM_DATA_DIR="$SHARED/dem_data"
export TRAIL_TILE_CACHE_DIR="$SHARED/tile_cache"
export TRAIL_V2_PATH_CACHE_DIR="$SHARED/path_cache_v2"
EOF
echo "wrote $SHARED/trail-shared.env"
echo "done. sizes:"
du -sh "$SHARED"/* 2>/dev/null || true
