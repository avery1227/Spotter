"""Just enough astronomy to place satellites and decide whether they are lit.

Everything here is low precision on purpose: a hundredth of a degree is a
fraction of a pixel at any focal length a live camera uses, and staying in
numpy avoids an ephemeris download at runtime.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pymap3d

#: WGS-84 equatorial radius, km. The shadow test uses a cylinder this wide.
EARTH_RADIUS_KM = 6378.137

_J2000_JD = 2451545.0
_UNIX_EPOCH_JD = 2440587.5


def julian_date(when: datetime) -> float:
    """Julian date (UT) of an aware datetime."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return _UNIX_EPOCH_JD + when.timestamp() / 86400.0


def split_julian_date(when: datetime) -> tuple[float, float]:
    """(whole, fraction) form that sgp4 wants, to keep float precision."""
    jd = julian_date(when)
    whole = np.floor(jd - 0.5) + 0.5
    return float(whole), float(jd - whole)


def gmst_rad(jd_ut: float) -> float:
    """Greenwich mean sidereal time, radians (IAU 1982, as SGP4 assumes)."""
    t = (jd_ut - _J2000_JD) / 36525.0
    seconds = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t
               + 0.093104 * t * t - 6.2e-6 * t * t * t)
    return float(np.radians((seconds % 86400.0) / 240.0) % (2.0 * np.pi))


def teme_to_ecef(r_teme_km: np.ndarray, jd_ut: float) -> np.ndarray:
    """Rotate SGP4's TEME positions into Earth-fixed coordinates.

    Only the sidereal rotation is applied; polar motion moves things by about
    ten metres, which is invisible at satellite ranges.
    """
    theta = gmst_rad(jd_ut)
    c, s = np.cos(theta), np.sin(theta)
    r = np.atleast_2d(r_teme_km)
    return np.column_stack([c * r[:, 0] + s * r[:, 1],
                            -s * r[:, 0] + c * r[:, 1],
                            r[:, 2]])


def sun_direction_eci(jd_ut: float) -> np.ndarray:
    """Unit vector towards the Sun in an Earth-centred inertial frame.

    The Astronomical Almanac's low-precision formula, good to about 0.01
    degrees between 1950 and 2050.
    """
    n = jd_ut - _J2000_JD
    mean_lon = np.radians((280.460 + 0.9856474 * n) % 360.0)
    anomaly = np.radians((357.528 + 0.9856003 * n) % 360.0)
    ecliptic_lon = (mean_lon + np.radians(1.915) * np.sin(anomaly)
                    + np.radians(0.020) * np.sin(2.0 * anomaly))
    obliquity = np.radians(23.439 - 4.0e-7 * n)
    return np.array([np.cos(ecliptic_lon),
                     np.cos(obliquity) * np.sin(ecliptic_lon),
                     np.sin(obliquity) * np.sin(ecliptic_lon)])


def sun_elevation_deg(when: datetime, lat: float, lon: float) -> float:
    """Geometric elevation of the Sun as seen from ``lat``/``lon``."""
    jd = julian_date(when)
    sun_ecef = teme_to_ecef(sun_direction_eci(jd) * 1.496e8, jd)[0] * 1000.0
    e, n, u = pymap3d.ecef2enu(sun_ecef[0], sun_ecef[1], sun_ecef[2],
                               lat, lon, 0.0)
    return float(np.degrees(np.arctan2(u, np.hypot(e, n))))


def in_earth_shadow(r_eci_km: np.ndarray, sun_dir: np.ndarray) -> np.ndarray:
    """True where a satellite is inside Earth's (cylindrical) shadow."""
    r = np.atleast_2d(r_eci_km)
    along = r @ sun_dir
    perpendicular = np.linalg.norm(r - np.outer(along, sun_dir), axis=1)
    return (along < 0.0) & (perpendicular < EARTH_RADIUS_KM)


def refraction_deg(elevation_deg: np.ndarray) -> np.ndarray:
    """Astronomical refraction, degrees to add to a *geometric* elevation.

    Saemundsson's formula (the inverse of Bennett's, which starts from the
    apparent elevation instead). About 0.48 degrees at the horizon and 0.16
    degrees at 5 degrees up: several pixels, and exactly where the low
    satellite passes this camera sees sit.
    """
    h = np.maximum(np.asarray(elevation_deg, dtype=float), -1.0)
    return 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11))) / 60.0


def lift_by_refraction(rel_enu: np.ndarray) -> np.ndarray:
    """Raise observer-relative ENU vectors to their apparent elevation."""
    rel = np.atleast_2d(np.asarray(rel_enu, dtype=float))
    horizontal = np.hypot(rel[:, 0], rel[:, 1])
    elevation = np.degrees(np.arctan2(rel[:, 2], np.maximum(horizontal, 1e-9)))
    apparent = np.radians(elevation + refraction_deg(elevation))
    distance = np.linalg.norm(rel, axis=1)
    scale = np.cos(apparent) * distance / np.maximum(horizontal, 1e-9)
    return np.column_stack([rel[:, 0] * scale, rel[:, 1] * scale,
                            distance * np.sin(apparent)])
