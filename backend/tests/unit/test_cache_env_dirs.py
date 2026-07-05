import os

from app.services.dem_tile_cache import DEMTileCache


def test_dem_data_dir_defaults(monkeypatch):
    monkeypatch.delenv("TRAIL_DEM_DATA_DIR", raising=False)
    cache = DEMTileCache()
    assert cache.dem_data_dir == os.path.abspath("dem_data")


def test_dem_data_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAIL_DEM_DATA_DIR", str(tmp_path / "dem"))
    cache = DEMTileCache()
    assert cache.dem_data_dir == os.path.abspath(str(tmp_path / "dem"))


def test_tile_cache_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("TRAIL_TILE_CACHE_DIR", str(tmp_path / "tiles"))
    cache = DEMTileCache()
    assert cache.tiled_cache.cache_dir == os.path.abspath(str(tmp_path / "tiles"))
