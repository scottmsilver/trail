import os

from app.services import osm_settings


def test_osm_cache_dir_defaults_to_local_cache(monkeypatch):
    monkeypatch.delenv("OSM_CACHE_DIR", raising=False)
    assert osm_settings.osm_cache_dir() == os.path.abspath("cache")


def test_osm_cache_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OSM_CACHE_DIR", str(tmp_path / "osm"))
    assert osm_settings.osm_cache_dir() == os.path.abspath(str(tmp_path / "osm"))


def test_apply_osm_settings_sets_cache_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("OSM_CACHE_DIR", str(tmp_path / "osm"))

    class FakeSettings:
        cache_folder = "cache"

    class FakeOx:
        settings = FakeSettings()

    ox = FakeOx()
    osm_settings.apply_osm_settings(ox)
    assert ox.settings.cache_folder == os.path.abspath(str(tmp_path / "osm"))
