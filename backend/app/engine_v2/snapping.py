"""Snap a drawn polyline onto the nearest OSM trail/path geometry.

"Snap to trail" is the Eval UI's default interpretation of a hand-drawn path:
the user means "go via this trail", not "go exactly along my wobbly line". Each
vertex is pulled onto the nearest trail segment within a threshold; vertices with
no trail nearby (genuine off-trail intent) are left as drawn.

The geometry here is pure and offline-testable; fetching the trail lines for a
bounding box lives in the service layer.
"""

import math
from typing import List, Sequence, Tuple

from app.models.route import Coordinate

_EARTH_R = 6371000.0

# A line is a sequence of (lat, lon) points.
Line = Sequence[Tuple[float, float]]


def _to_xy(lat: float, lon: float, lat0: float) -> Tuple[float, float]:
    """Equirectangular projection to local meters around reference latitude lat0."""
    x = math.radians(lon) * math.cos(math.radians(lat0)) * _EARTH_R
    y = math.radians(lat) * _EARTH_R
    return x, y


def _to_latlon(x: float, y: float, lat0: float) -> Tuple[float, float]:
    lat = math.degrees(y / _EARTH_R)
    lon = math.degrees(x / (_EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon


def _project_to_segment(px, py, ax, ay, bx, by) -> Tuple[float, float]:
    """Nearest point on segment AB to P, all in planar coords."""
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return ax, ay
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return ax + t * dx, ay + t * dy


def snap_polyline_to_lines(
    path: List[Coordinate], trail_lines: List[Line], threshold_m: float = 25.0
) -> Tuple[List[Coordinate], bool]:
    """Pull each vertex onto the nearest trail segment within ``threshold_m``.

    Returns ``(snapped_path, did_snap)``. Vertices with no trail within the
    threshold are kept as-is. Idempotent: snapping an already-snapped path does
    not move points (they are already at distance ~0 from their trail).
    """
    if not trail_lines:
        return list(path), False
    # Clamp the projection reference latitude away from the poles, where the
    # equirectangular cos(lat) factor becomes numerically ill-conditioned.
    lat0 = max(-89.9, min(89.9, path[0].lat))
    snapped: List[Coordinate] = []
    did = False
    for pt in path:
        px, py = _to_xy(pt.lat, pt.lon, lat0)
        best_xy = None
        best_d = threshold_m
        for line in trail_lines:
            for (alat, alon), (blat, blon) in zip(line, line[1:]):
                ax, ay = _to_xy(alat, alon, lat0)
                bx, by = _to_xy(blat, blon, lat0)
                qx, qy = _project_to_segment(px, py, ax, ay, bx, by)
                d = math.hypot(px - qx, py - qy)
                if d < best_d:
                    best_d = d
                    best_xy = (qx, qy)
        if best_xy is not None:
            lat, lon = _to_latlon(best_xy[0], best_xy[1], lat0)
            snapped.append(Coordinate(lat=lat, lon=lon))
            if best_d > 1e-6:
                did = True
        else:
            snapped.append(pt)
    return snapped, did


def densify_polyline(path: List[Coordinate], step_m: float, max_points: int) -> List[Coordinate]:
    """Insert interpolated vertices so consecutive points are ~``step_m`` apart.

    Per-vertex snapping only moves the points you clicked; between clicks the
    line stays straight. Densifying first means the whole line gets pulled onto
    a trail (each interpolated point snaps), so a sparse hand-drawn path visibly
    hugs the trail instead of only nudging its endpoints. Original vertices are
    always kept. For very long paths the step is coarsened so the output stays
    near ``max_points`` (bounding downstream scoring work)."""
    if len(path) < 2:
        return list(path)
    lat0 = max(-89.9, min(89.9, path[0].lat))
    xy = [_to_xy(p.lat, p.lon, lat0) for p in path]
    seg_len = [math.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1]) for i in range(len(xy) - 1)]
    total = sum(seg_len)
    if total == 0:
        return list(path)
    # Coarsen the step if a fine one would blow past max_points.
    eff_step = max(step_m, total / max(1, max_points - len(path)))
    out: List[Coordinate] = [path[0]]
    for i in range(len(path) - 1):
        (ax, ay), (bx, by) = xy[i], xy[i + 1]
        d = seg_len[i]
        n = int(d // eff_step) if d > 0 else 0
        for k in range(1, n + 1):
            t = (k * eff_step) / d
            if t >= 1.0:
                break
            x, y = ax + (bx - ax) * t, ay + (by - ay) * t
            lat, lon = _to_latlon(x, y, lat0)
            out.append(Coordinate(lat=lat, lon=lon))
        out.append(path[i + 1])
    return out
