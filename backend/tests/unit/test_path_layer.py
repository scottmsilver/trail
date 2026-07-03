# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
"""Unit tests for the v2 path layer (OSM classification + rasterization)."""

from app.engine_v2.path_layer import PathType, classify_feature, get_path_type_name


class TestClassifyFeature:
    def test_hiking_path_is_trail(self):
        assert classify_feature({"highway": "path"}) == PathType.TRAIL

    def test_track_is_trail(self):
        assert classify_feature({"highway": "track"}) == PathType.TRAIL

    def test_piste_is_trail(self):
        assert classify_feature({"piste:type": "downhill"}) == PathType.TRAIL

    def test_hiking_route_is_trail(self):
        assert classify_feature({"route": "hiking"}) == PathType.TRAIL

    def test_steps_and_cycleway_are_path(self):
        assert classify_feature({"highway": "steps"}) == PathType.PATH
        assert classify_feature({"highway": "cycleway"}) == PathType.PATH

    def test_footway_and_pedestrian_are_footway(self):
        assert classify_feature({"highway": "footway"}) == PathType.FOOTWAY
        assert classify_feature({"highway": "pedestrian"}) == PathType.FOOTWAY

    def test_roads_are_residential(self):
        for hw in ["residential", "living_street", "service", "unclassified", "tertiary", "secondary", "primary"]:
            assert classify_feature({"highway": hw}) == PathType.RESIDENTIAL, hw

    def test_park_and_meadow_are_natural(self):
        assert classify_feature({"leisure": "park"}) == PathType.NATURAL
        assert classify_feature({"natural": "meadow"}) == PathType.NATURAL
        assert classify_feature({"landuse": "grass"}) == PathType.NATURAL

    def test_highway_takes_priority_over_landuse(self):
        # A road through a park is still a road
        assert classify_feature({"highway": "residential", "landuse": "grass"}) == PathType.RESIDENTIAL

    def test_unrecognized_is_unknown(self):
        assert classify_feature({"building": "yes"}) == PathType.UNKNOWN
        assert classify_feature({}) == PathType.UNKNOWN

    def test_none_values_ignored(self):
        # osmnx GeoDataFrames contain NaN/None for absent tags
        assert classify_feature({"highway": None, "leisure": "park"}) == PathType.NATURAL


def test_path_type_names():
    assert get_path_type_name(PathType.TRAIL) == "trail"
    assert get_path_type_name(PathType.UNKNOWN) == "off_path"
    assert get_path_type_name(255) == "invalid"
