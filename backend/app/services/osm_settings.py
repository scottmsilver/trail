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
    """Point osmnx's cache at ``OSM_CACHE_DIR`` (no-op change when unset).

    When we're pointed at our own Overpass mirror (``OVERPASS_URLS`` set), turn
    off osmnx's rate-limit pacing. That pacing polls the server's ``/status``
    endpoint and, against a local mirror whose status reports an unlimited rate,
    recurses until it overflows the stack ("maximum recursion depth exceeded").
    A mirror we run has no public rate limit, so skipping the poll is correct.
    """
    ox.settings.cache_folder = osm_cache_dir()
    if os.environ.get("OVERPASS_URLS", "").strip():
        ox.settings.overpass_rate_limit = False
