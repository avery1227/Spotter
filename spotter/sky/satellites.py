"""Satellites from CelesTrak orbital elements, propagated with SGP4.

A pass is computed rather than reported: given two-line elements (TLEs), a
satellite's position at any instant is a few microseconds of arithmetic. So
there is no feed to poll at frame rate, only a catalogue to refresh a couple
of times a day.

Geometry matters more here than anywhere else in the overlay. The camera
looks along the water, so a satellite only enters the frame low in the sky,
at 1,000-2,000 km range. The usual track pipeline would get that wrong three
ways: its range limit culls it, its horizon test assumes a surface target,
and its terrestrial refraction lift (tuned for ships tens of km away) would
raise a satellite by kilometres. Satellites are therefore projected here,
with astronomical refraction instead.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pymap3d
import requests

from ..logging_setup import get_logger
from ..projection import ProjectedTarget
from ..tracks.model import Position, TrackKind
from ..tracks.store import Track, TrackState
from . import astro

log = get_logger(__name__)

CELESTRAK_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=tle"

#: CelesTrak names are catalogue names; these read better on screen.
DISPLAY_NAMES = {
    "ISS (ZARYA)": "ISS",
    "CSS (TIANHE)": "Tiangong",
    "HST": "Hubble",
}

#: Two objects closer than this are drawn as one. Docked modules and visiting
#: vehicles are catalogued separately but share the station's orbit.
MERGE_DISTANCE_KM = 15.0


def parse_tle_text(text: str) -> list[tuple[str, str, str]]:
    """Split a 3-line TLE file into (name, line1, line2) triples."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    out = []
    i = 0
    while i + 2 < len(lines):
        name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
        if line1.startswith("1 ") and line2.startswith("2 "):
            out.append((name.strip(), line1, line2))
            i += 3
        else:
            i += 1      # resynchronise on a malformed entry
    return out


class SatelliteCatalog:
    """Downloads TLEs for the configured groups and keeps them fresh.

    CelesTrak asks clients not to fetch the same data more than once every
    two hours, and will block those that do. The catalogue is therefore
    cached on disk and only refetched when it is older than ``refresh_h``, so
    restarts cost nothing.
    """

    def __init__(self, groups: list[str], cache_path: Optional[Path],
                 refresh_h: float = 12.0, timeout_s: float = 20.0):
        self.groups = [g for g in groups if g]
        self.cache_path = cache_path
        self.refresh_s = max(2.0, refresh_h) * 3600.0
        self.timeout_s = timeout_s
        self._lock = threading.Lock()
        self._entries: list[tuple[str, str, str]] = []
        self._loaded_at: Optional[float] = None     # epoch of the data we hold
        self.last_error: Optional[str] = None

    @property
    def entries(self) -> list[tuple[str, str, str]]:
        with self._lock:
            return list(self._entries)

    def age_s(self) -> Optional[float]:
        return None if self._loaded_at is None else time.time() - self._loaded_at

    def needs_refresh(self) -> bool:
        age = self.age_s()
        return age is None or age > self.refresh_s

    def load_cache(self) -> bool:
        if self.cache_path is None or not self.cache_path.is_file():
            return False
        try:
            entries = parse_tle_text(self.cache_path.read_text(encoding="utf-8"))
        except OSError as exc:
            log.warning("could not read TLE cache", extra={"error": str(exc)})
            return False
        if not entries:
            return False
        self._set(entries, self.cache_path.stat().st_mtime)
        log.info("satellite elements loaded from cache", extra={
            "satellites": len(entries),
            "age_h": round((self.age_s() or 0) / 3600.0, 1)})
        return True

    def fetch(self, session: requests.Session) -> bool:
        texts = []
        for group in self.groups:
            url = CELESTRAK_URL.format(group=group)
            for attempt in range(3):
                try:
                    response = session.get(url, timeout=self.timeout_s)
                    response.raise_for_status()
                    break
                except requests.RequestException as exc:
                    error = exc
                    # A 403 is CelesTrak rate-limiting us; retrying makes it worse.
                    status = getattr(exc.response, "status_code", None)
                    if status in (403, 404) or attempt == 2:
                        self.last_error = f"{group}: {error}"
                        log.warning("satellite elements fetch failed", extra={
                            "group": group, "error": str(error)})
                        return False
                    time.sleep(2.0 * (attempt + 1))
            texts.append(response.text)

        combined = "\n".join(texts)
        entries = parse_tle_text(combined)
        if not entries:
            self.last_error = "CelesTrak returned no elements"
            log.warning("satellite elements fetch returned nothing",
                        extra={"groups": self.groups})
            return False

        if self.cache_path is not None:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(combined, encoding="utf-8")
            except OSError as exc:
                log.warning("could not write TLE cache", extra={"error": str(exc)})
        self.last_error = None
        self._set(entries, time.time())
        log.info("satellite elements updated", extra={
            "groups": self.groups, "satellites": len(entries)})
        return True

    def _set(self, entries, loaded_at: float) -> None:
        # Groups overlap (the ISS is in both "stations" and "visual").
        unique = {}
        for name, line1, line2 in entries:
            unique.setdefault(line1[2:7].strip(), (name, line1, line2))
        with self._lock:
            self._entries = list(unique.values())
            self._loaded_at = loaded_at


class SatelliteLayer:
    """Projects every catalogued satellite that is above the horizon and in frame."""

    def __init__(self, cfg):
        node = cfg.sub("sky.satellites")
        self.enabled = bool(node.get("enabled", True))
        self.catalog = SatelliteCatalog(
            groups=list(node.get("groups", ["stations", "visual"]) or []),
            cache_path=cfg.path("sky.satellites.cache", "./state/satellites.tle"),
            refresh_h=float(node.get("refresh_h", 12.0)),
            timeout_s=float(node.get("timeout_s", 20.0)),
        )
        self.min_elevation_deg = float(node.get("min_elevation_deg", 0.0))
        self.show_in_shadow = bool(node.get("show_in_shadow", True))
        # Nothing but the Sun and Moon is visible in a daylight sky, so by
        # default only the stations keep their labels then.
        self.show_in_daylight = bool(node.get("show_in_daylight", False))
        self.always_show = {str(n) for n in (node.get("always_show",
                                                      ["ISS", "Tiangong"]) or [])}
        self.margin_px = float(cfg.get("projection.frame_margin_px", 80))
        self.user_agent = str(node.get("user_agent", "spotter-overlay/1.0"))

        self._array = None
        self._names: list[str] = []
        self._ids: list[str] = []
        self._built_from: Optional[float] = None
        self._tracks: dict[str, Track] = {}
        self._sun_cache: tuple[float, float] = (-1e18, 0.0)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.in_view = 0

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "SatelliteLayer":
        if not self.enabled:
            return self
        self.catalog.load_cache()
        self._thread = threading.Thread(target=self._refresh_loop,
                                        name="satellite-elements", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _refresh_loop(self) -> None:
        session = requests.Session()
        session.headers["User-Agent"] = self.user_agent
        retry_s = 600.0
        while not self._stop.is_set():
            if self.catalog.needs_refresh():
                if not self.catalog.fetch(session):
                    # Keep whatever we have (a stale catalogue is still good to
                    # a few km for days) and try again later, gently.
                    self._stop.wait(retry_s)
                    retry_s = min(retry_s * 2.0, 6 * 3600.0)
                    continue
                retry_s = 600.0
            self._stop.wait(300.0)

    # -- projection ---------------------------------------------------------
    def _ensure_array(self) -> bool:
        entries = self.catalog.entries
        if not entries:
            return False
        marker = self.catalog._loaded_at
        if self._array is not None and self._built_from == marker:
            return True
        from sgp4.api import Satrec, SatrecArray

        records, names, ids = [], [], []
        for name, line1, line2 in entries:
            try:
                records.append(Satrec.twoline2rv(line1, line2))
            except Exception:
                continue
            names.append(DISPLAY_NAMES.get(name, name))
            ids.append(line1[2:7].strip())
        if not records:
            return False
        self._array = SatrecArray(records)
        self._names, self._ids = names, ids
        self._built_from = marker
        return True

    def _observer_sun_elevation(self, when: datetime, lat: float, lon: float) -> float:
        stamp = when.timestamp()
        cached_at, value = self._sun_cache
        if abs(stamp - cached_at) > 30.0:
            value = astro.sun_elevation_deg(when, lat, lon)
            self._sun_cache = (stamp, value)
        return value

    def project(self, when: datetime, model) -> list[ProjectedTarget]:
        if not self.enabled or model is None or not self._ensure_array():
            self.in_view = 0
            return []

        jd, fraction = astro.split_julian_date(when)
        errors, r_teme, _v = self._array.sgp4(np.array([jd]), np.array([fraction]))
        r_teme = r_teme[:, 0, :]
        ok = (errors[:, 0] == 0) & np.all(np.isfinite(r_teme), axis=1)
        if not np.any(ok):
            self.in_view = 0
            return []

        jd_ut = jd + fraction
        ecef_m = astro.teme_to_ecef(r_teme, jd_ut) * 1000.0
        east, north, up = pymap3d.ecef2enu(ecef_m[:, 0], ecef_m[:, 1], ecef_m[:, 2],
                                           model.ref_lat, model.ref_lon, 0.0)
        enu = np.column_stack([east, north, up])
        rel = astro.lift_by_refraction(enu - model.camera_enu)
        horizontal = np.hypot(rel[:, 0], rel[:, 1])
        elevation = np.degrees(np.arctan2(rel[:, 2], np.maximum(horizontal, 1e-9)))

        candidate = ok & (elevation >= self.min_elevation_deg)
        if not np.any(candidate):
            self.in_view = 0
            return []

        uv, in_front, _ = model.project_enu(rel + model.camera_enu, refract=False)
        inside = (in_front
                  & (uv[:, 0] >= -self.margin_px)
                  & (uv[:, 0] <= model.width + self.margin_px)
                  & (uv[:, 1] >= -self.margin_px)
                  & (uv[:, 1] <= model.height + self.margin_px))
        keep = candidate & inside
        if not np.any(keep):
            self.in_view = 0
            return []

        sun_dir = astro.sun_direction_eci(jd_ut)
        shadow = astro.in_earth_shadow(r_teme, sun_dir)
        cam_lat, cam_lon = model.camera_latlon
        sun_el = self._observer_sun_elevation(when, cam_lat, cam_lon)
        sky = "day" if sun_el > -0.8 else ("twilight" if sun_el > -12.0 else "dark")

        distance = np.linalg.norm(rel, axis=1)
        bearing = np.degrees(np.arctan2(rel[:, 0], rel[:, 1])) % 360.0

        out: list[ProjectedTarget] = []
        kept_positions: list[np.ndarray] = []
        # Lowest catalogue number first, so the station itself wins over the
        # modules and spacecraft docked to it.
        for index in sorted(np.flatnonzero(keep), key=lambda i: int(self._ids[i])):
            name = self._names[index]
            if name not in self.always_show:
                if not self.show_in_shadow and shadow[index]:
                    continue
                if sky == "day" and not self.show_in_daylight:
                    continue
            if any(np.linalg.norm(ecef_m[index] - p) < MERGE_DISTANCE_KM * 1000.0
                   for p in kept_positions):
                continue
            kept_positions.append(ecef_m[index])

            lat, lon, alt = pymap3d.ecef2geodetic(*ecef_m[index])
            norad = self._ids[index]
            track = self._tracks.get(norad)
            if track is None:
                track = Track(f"sat:{norad}", TrackKind.SATELLITE)
                self._tracks[norad] = track
            track.labels = {
                "name": name, "norad": norad,
                "sunlit": not bool(shadow[index]), "sky": sky,
            }
            u, v = float(uv[index, 0]), float(uv[index, 1])
            out.append(ProjectedTarget(
                state=TrackState(track=track, position=Position(
                    lat=float(lat), lon=float(lon), alt_m=float(alt),
                    mode="orbit")),
                u=u, v=v,
                range_m=float(distance[index]),
                ground_range_m=float(horizontal[index]),
                bearing_deg=float(bearing[index]),
                elevation_deg=float(elevation[index]),
                category="satellite" if not shadow[index] else "satellite_shadow",
                on_screen=bool(0 <= u <= model.width and 0 <= v <= model.height),
            ))
        self.in_view = len(out)
        return out

    def status(self) -> dict:
        age = self.catalog.age_s()
        return {
            "enabled": self.enabled,
            "catalogued": len(self.catalog.entries),
            "elements_age_h": round(age / 3600.0, 1) if age is not None else None,
            "in_view": self.in_view,
            "error": self.catalog.last_error,
        }

    def attribution(self) -> Optional[str]:
        return "Orbits: CelesTrak" if self.enabled and self.catalog.entries else None
