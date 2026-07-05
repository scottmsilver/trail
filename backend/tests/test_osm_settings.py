"""Tests for the osmnx settings helper.

The Overpass endpoint is env-driven (OVERPASS_URLS) so we can point every osmnx
call site at a local Overpass server without hard-coding a URL in code.
"""

import types

from app.services import osm_settings


def _fake_ox():
    """A stand-in for the osmnx module exposing just the settings we touch."""
    ox = types.SimpleNamespace()
    ox.settings = types.SimpleNamespace(
        cache_folder=None,
        overpass_url="https://overpass-api.de/api",  # osmnx's real default
        overpass_rate_limit=True,  # osmnx's real default
    )
    return ox


def test_overpass_urls_parses_and_strips_comma_list(monkeypatch):
    monkeypatch.setenv("OVERPASS_URLS", " http://localhost:12345/api , http://b/api ,")
    assert osm_settings.overpass_urls() == [
        "http://localhost:12345/api",
        "http://b/api",
    ]


def test_overpass_urls_empty_when_unset(monkeypatch):
    monkeypatch.delenv("OVERPASS_URLS", raising=False)
    assert osm_settings.overpass_urls() == []


def test_apply_points_at_first_overpass_endpoint(monkeypatch):
    monkeypatch.setenv("OVERPASS_URLS", "http://localhost:12345/api,http://b/api")
    ox = _fake_ox()
    osm_settings.apply_osm_settings(ox)
    assert ox.settings.overpass_url == "http://localhost:12345/api"


def test_apply_leaves_default_endpoint_when_unset(monkeypatch):
    monkeypatch.delenv("OVERPASS_URLS", raising=False)
    ox = _fake_ox()
    before = ox.settings.overpass_url
    osm_settings.apply_osm_settings(ox)
    assert ox.settings.overpass_url == before  # osmnx default untouched


def test_custom_endpoint_disables_self_rate_limiting(monkeypatch):
    # A local/self-hosted Overpass we point at should not be self-throttled, and
    # disabling osmnx's rate limit also skips its /api/status poll (which hangs
    # forever on Overpass builds whose status format osmnx can't parse).
    monkeypatch.setenv("OVERPASS_URLS", "http://localhost:12345/api")
    monkeypatch.delenv("OVERPASS_RATE_LIMIT", raising=False)
    ox = _fake_ox()
    osm_settings.apply_osm_settings(ox)
    assert ox.settings.overpass_rate_limit is False


def test_default_endpoint_keeps_rate_limiting_on(monkeypatch):
    monkeypatch.delenv("OVERPASS_URLS", raising=False)
    monkeypatch.delenv("OVERPASS_RATE_LIMIT", raising=False)
    ox = _fake_ox()
    osm_settings.apply_osm_settings(ox)
    assert ox.settings.overpass_rate_limit is True


def test_rate_limit_env_override_wins(monkeypatch):
    monkeypatch.setenv("OVERPASS_URLS", "http://localhost:12345/api")
    monkeypatch.setenv("OVERPASS_RATE_LIMIT", "1")  # force it back on
    ox = _fake_ox()
    osm_settings.apply_osm_settings(ox)
    assert ox.settings.overpass_rate_limit is True


def test_apply_still_sets_cache_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("OSM_CACHE_DIR", str(tmp_path))
    ox = _fake_ox()
    osm_settings.apply_osm_settings(ox)
    assert ox.settings.cache_folder == str(tmp_path)
