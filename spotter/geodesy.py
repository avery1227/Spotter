"""Geodetic helpers: ENU conversion, refraction, horizon geometry.

Everything downstream works in a local East-North-Up frame anchored at the
camera's surveyed position. ``pymap3d.geodetic2enu`` does the ellipsoidal part
exactly, so Earth curvature is already accounted for in the Up component: a ship
30 km away sits about 70 m *below* the tangent plane before refraction.

Refraction is layered on top with the standard effective-Earth-radius trick.
"""

from __future__ import annotations

import numpy as np
import pymap3d

#: Mean Earth radius, metres. Matches the value behind the k-factor convention.
EARTH_RADIUS_M = 6_371_008.8

#: 7/6 is the usual optical/radio refraction factor for near-horizontal paths
#: over water in temperate conditions.
DEFAULT_REFRACTION_K = 7.0 / 6.0


def geodetic_to_enu(lat, lon, alt_m, ref_lat: float, ref_lon: float,
                    ref_alt_m: float) -> np.ndarray:
    """Convert geodetic coordinates to ENU metres about a reference point.

    Accepts scalars or arrays; always returns an ``(N, 3)`` float array of
    ``[east, north, up]``.
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))
    alt = np.atleast_1d(np.asarray(alt_m, dtype=float))
    east, north, up = pymap3d.geodetic2enu(lat, lon, alt, ref_lat, ref_lon, ref_alt_m)
    return np.column_stack([np.asarray(east, dtype=float),
                            np.asarray(north, dtype=float),
                            np.asarray(up, dtype=float)])


def enu_to_geodetic(enu: np.ndarray, ref_lat: float, ref_lon: float,
                    ref_alt_m: float) -> np.ndarray:
    """Inverse of :func:`geodetic_to_enu`; returns ``(N, 3)`` lat/lon/alt."""
    enu = np.atleast_2d(np.asarray(enu, dtype=float))
    lat, lon, alt = pymap3d.enu2geodetic(enu[:, 0], enu[:, 1], enu[:, 2],
                                         ref_lat, ref_lon, ref_alt_m)
    return np.column_stack([np.asarray(lat, dtype=float),
                            np.asarray(lon, dtype=float),
                            np.asarray(alt, dtype=float)])


def refraction_lift_m(horizontal_range_m, k: float = DEFAULT_REFRACTION_K) -> np.ndarray:
    """Apparent upward displacement caused by atmospheric refraction, metres.

    Geometric curvature drop over a ground distance ``d`` is ``d^2 / (2R)``. Under
    the effective-radius model the *apparent* drop is ``d^2 / (2kR)``, so a target
    appears higher than its geometric position by the difference::

        lift = d^2 / (2R) * (1 - 1/k)

    At k = 7/6 and d = 30 km this is about 10 m -- small in absolute terms, but
    a tenth of a degree at that range, which is several pixels.
    """
    d = np.asarray(horizontal_range_m, dtype=float)
    if k <= 0:
        return np.zeros_like(d)
    return (d * d) / (2.0 * EARTH_RADIUS_M) * (1.0 - 1.0 / k)


def apply_refraction(enu: np.ndarray, k: float = DEFAULT_REFRACTION_K) -> np.ndarray:
    """Return ``enu`` with the Up component raised by the refraction lift."""
    enu = np.atleast_2d(np.asarray(enu, dtype=float)).astype(float, copy=True)
    horizontal = np.hypot(enu[:, 0], enu[:, 1])
    enu[:, 2] += refraction_lift_m(horizontal, k)
    return enu


def horizon_distance_m(height_m: float, k: float = DEFAULT_REFRACTION_K) -> float:
    """Distance to the visible horizon from ``height_m`` above the surface."""
    if height_m <= 0:
        return 0.0
    return float(np.sqrt(2.0 * k * EARTH_RADIUS_M * height_m))


def max_visible_range_m(observer_height_m: float, target_height_m: float,
                        k: float = DEFAULT_REFRACTION_K) -> float:
    """Greatest range at which a target of the given height is above the horizon.

    The classic geometric-range formula: each end contributes its own horizon
    distance, and they meet at the tangent point.
    """
    return (horizon_distance_m(max(observer_height_m, 0.0), k)
            + horizon_distance_m(max(target_height_m, 0.0), k))


def is_above_horizon(horizontal_range_m, target_height_m,
                     observer_height_m: float,
                     k: float = DEFAULT_REFRACTION_K,
                     slack_m: float = 0.0) -> np.ndarray:
    """Elementwise horizon visibility test.

    ``slack_m`` is added to the target height, which is the cheap way to say
    "a ship's superstructure peeks over before its hull does".
    """
    d = np.atleast_1d(np.asarray(horizontal_range_m, dtype=float))
    h = np.atleast_1d(np.asarray(target_height_m, dtype=float)) + float(slack_m)
    observer_horizon = horizon_distance_m(max(observer_height_m, 0.0), k)
    target_horizon = np.sqrt(2.0 * k * EARTH_RADIUS_M * np.maximum(h, 0.0))
    return d <= (observer_horizon + target_horizon)


def bearing_deg(enu: np.ndarray) -> np.ndarray:
    """Compass bearing (degrees clockwise from North) of each ENU vector."""
    enu = np.atleast_2d(np.asarray(enu, dtype=float))
    return np.degrees(np.arctan2(enu[:, 0], enu[:, 1])) % 360.0


def elevation_deg(enu: np.ndarray) -> np.ndarray:
    """Elevation angle above the local horizontal plane, degrees."""
    enu = np.atleast_2d(np.asarray(enu, dtype=float))
    horizontal = np.hypot(enu[:, 0], enu[:, 1])
    return np.degrees(np.arctan2(enu[:, 2], np.maximum(horizontal, 1e-9)))


def slant_range_m(enu: np.ndarray) -> np.ndarray:
    enu = np.atleast_2d(np.asarray(enu, dtype=float))
    return np.linalg.norm(enu, axis=1)


def horizontal_range_m(enu: np.ndarray) -> np.ndarray:
    enu = np.atleast_2d(np.asarray(enu, dtype=float))
    return np.hypot(enu[:, 0], enu[:, 1])


def dead_reckon(lat: float, lon: float, course_deg: float, speed_mps: float,
                dt_s: float) -> tuple[float, float]:
    """Advance a position along a constant course at constant speed.

    Used for ships between AIS reports. Over the ranges and intervals involved
    (a few km at most) a flat-Earth step in ENU is well under a metre of error,
    and it avoids a geodesic solve per target per frame.
    """
    if dt_s == 0.0 or speed_mps == 0.0:
        return lat, lon
    distance = speed_mps * dt_s
    theta = np.radians(course_deg)
    east = distance * np.sin(theta)
    north = distance * np.cos(theta)
    new_lat, new_lon, _ = pymap3d.enu2geodetic(east, north, 0.0, lat, lon, 0.0)
    return float(new_lat), float(new_lon)
