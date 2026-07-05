"""osmnx settings helper.

osmnx caches Overpass responses under ``ox.settings.cache_folder`` (default
``./cache``). To share that cache across dev instances we redirect it via the
``OSM_CACHE_DIR`` env var without hardcoding any path in code.
"""

import os


def osm_cache_dir(default: str = "cache") -> str:
    """Absolute path osmnx should use for its Overpass response cache."""
    return os.path.abspath(os.environ.get("OSM_CACHE_DIR", default))


def apply_osm_settings(ox) -> None:
    """Point osmnx's cache at ``OSM_CACHE_DIR`` (no-op change when unset)."""
    ox.settings.cache_folder = osm_cache_dir()
