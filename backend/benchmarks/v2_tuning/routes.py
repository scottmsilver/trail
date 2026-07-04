"""Fixed route set for v2 pathfinder speed tuning.

All routes lie inside the locally-cached Park City / Wasatch DEM tiles
(lat 40.57-40.71, lon -111.64 to -111.51) so they run fully offline.

Each route also gets synthetic-terrain variants (see harness.build_inputs)
so the benchmark exercises the cost function's terrain-multiplier,
obstacle-skip, and trail-transition branches -- not just the all-UNKNOWN
grid you get offline. That keeps "faster" honest across all code paths.
"""

# (name, start_lat, start_lon, end_lat, end_lon)
# Chosen for a spread of lengths / directions / node counts.
ROUTES = [
    ("short_ne", 40.6050, -111.6250, 40.6120, -111.6150),
    ("short_sw", 40.6300, -111.5800, 40.6220, -111.5900),
    ("med_diag", 40.6050, -111.6250, 40.6250, -111.6000),
    ("med_ns", 40.5900, -111.6000, 40.6200, -111.6000),
    ("med_ew", 40.6100, -111.6300, 40.6100, -111.5900),
    ("long_diag", 40.5850, -111.6350, 40.6350, -111.5900),
    ("long_ne", 40.5900, -111.6300, 40.6400, -111.5700),
    ("park_a", 40.6572, -111.5706, 40.6486, -111.5639),
    ("park_b", 40.6599, -111.5662, 40.6450, -111.5700),
    ("hillclimb", 40.6000, -111.6200, 40.6100, -111.6050),
]
