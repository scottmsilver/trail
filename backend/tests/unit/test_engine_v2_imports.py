# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
"""Smoke test: engine_v2 package imports cleanly."""


def test_elevation_library_imports():
    from app.engine_v2.elevation import Bounds

    b = Bounds(south=40.0, north=40.1, west=-111.6, east=-111.5)
    assert b.north > b.south


def test_fd_safe_wrapper_imports():
    from app.engine_v2.elevation_fd_safe import FDManagedElevationLibrary

    assert hasattr(FDManagedElevationLibrary, "close_all")
