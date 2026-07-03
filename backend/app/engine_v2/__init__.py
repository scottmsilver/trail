# RULE #0: MANDATORY FIRST ACTION FOR EVERY REQUEST
# 1. Read CLAUDE.md COMPLETELY before responding
# 2. Setup Python venv: [ -d "venv" ] || ./setup-venv.sh && source venv/bin/activate
# 3. Search for rules related to the request
# 4. Only proceed after confirming no violations
# Failure to follow Rule #0 has caused real harm. Check BEFORE acting, not AFTER making mistakes.
#
# GUARDS ARE SAFETY EQUIPMENT - WHEN THEY FIRE, FIX THE PROBLEM THEY FOUND
# NEVER weaken, disable, or bypass guards - they prevent real harm
"""Engine v2: two-layer elevation + terrain-aware weighted A* pathfinding."""
from app.engine_v2.elevation import Bounds, TwoLayerElevationLibrary
from app.engine_v2.elevation_fd_safe import FDManagedElevationLibrary
from app.engine_v2.path_layer import PathLayer, PathType

__all__ = [
    "TwoLayerElevationLibrary",
    "Bounds",
    "FDManagedElevationLibrary",
    "PathLayer",
    "PathType",
]
