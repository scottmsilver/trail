"""osmnx settings helper.

osmnx caches Overpass responses under ``ox.settings.cache_folder`` (default
``./cache``). To share that cache across dev instances we redirect it via the
``OSM_CACHE_DIR`` env var without hardcoding any path in code.
"""

import os


def osm_cache_dir(default: str = "cache") -> str:
    """Absolute path osmnx should use for its Overpass response cache."""
    return os.path.abspath(os.environ.get("OSM_CACHE_DIR", default))


def overpass_urls() -> list:
    """Overpass base URLs from ``OVERPASS_URLS`` (comma-separated), or ``[]``.

    Never hard-codes an endpoint (see CLAUDE.md): an empty list means "leave
    osmnx on its own default". Each entry is a base URL like
    ``http://localhost:12345/api`` — osmnx appends ``/interpreter`` itself.
    """
    raw = os.environ.get("OVERPASS_URLS", "")
    return [u.strip() for u in raw.split(",") if u.strip()]


def _rate_limit_enabled(has_custom_endpoint: bool) -> bool:
    """Whether osmnx should self-throttle against the Overpass API.

    ``OVERPASS_RATE_LIMIT`` (1/true/yes | 0/false/no) forces the choice. Absent
    it, default to *off* whenever a custom ``OVERPASS_URLS`` is configured: an
    endpoint we point at is one we control (a local server), so self-throttling
    is pointless — and osmnx's rate-limit path polls ``/api/status`` and recurses
    forever on Overpass builds whose status format it can't parse (e.g. a local
    ``osm3s`` instance). Public overpass-api.de keeps rate limiting on.
    """
    env = os.environ.get("OVERPASS_RATE_LIMIT")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes")
    return not has_custom_endpoint


def apply_osm_settings(ox) -> None:
    """Point osmnx at the shared cache and, when ``OVERPASS_URLS`` is set, at the
    first configured Overpass endpoint (disabling self-rate-limiting for it).

    Both are env-driven so no server URL is hard-coded. Every osmnx call site
    (v2 ``path_layer`` plus the v1 fetches in ``main`` and ``dem_tile_cache``)
    goes through here, so setting ``OVERPASS_URLS`` redirects them all to a local
    Overpass server. Leaving ``overpass_url`` untouched when unset preserves
    osmnx's default behavior.
    """
    ox.settings.cache_folder = osm_cache_dir()
    urls = overpass_urls()
    if urls:
        ox.settings.overpass_url = urls[0]
    ox.settings.overpass_rate_limit = _rate_limit_enabled(bool(urls))
