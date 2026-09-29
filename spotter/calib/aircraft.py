"""Frozen frames for calibrating against aircraft.

A plane is the best high control point there is: ADS-B gives its position to a
few metres, it is far above the horizon where shoreline landmarks cannot
reach, and there is a new one every few minutes. What makes it awkward is that
it moves. So the pipeline hands over one frame *exactly as decoded* (before any
overlay is drawn on it) together with every aircraft's position at that
frame's timestamp, and the user clicks where each plane really is.

The capture also records each aircraft's velocity and the
``stream.encoder_delay_s`` in force, which lets the solver work out how far
that delay is off as well as how the camera is pointed.
"""

from __future__ import annotations

import math
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

from ..geodesy import geodetic_to_enu
from ..tracks.model import TrackKind


@dataclass
class FrozenAircraft:
    track_id: str
    label: str
    lat: float
    lon: float
    alt_m: float
    #: "geom" (GNSS height) or "baro" (pressure altitude, can be 100 m+ off).
    alt_source: str
    ve: float
    vn: float
    vu: float
    range_km: float
    bearing_deg: float
    #: Where the current calibration draws it, if in front of the camera.
    predicted: Optional[tuple[float, float]] = None
    details: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "id": self.track_id, "label": self.label,
            "lat": round(self.lat, 7), "lon": round(self.lon, 7),
            "alt_m": round(self.alt_m, 1), "alt_source": self.alt_source,
            "ve": round(self.ve, 2), "vn": round(self.vn, 2), "vu": round(self.vu, 2),
            "range_km": round(self.range_km, 2),
            "bearing_deg": round(self.bearing_deg, 2),
            "predicted": ([round(self.predicted[0], 1), round(self.predicted[1], 1)]
                          if self.predicted else None),
            "details": self.details,
        }


@dataclass
class FreezeCapture:
    id: str
    image: np.ndarray                  # BGRA, as decoded
    frame_time: datetime
    delay_s: Optional[float]
    aircraft: list[FrozenAircraft]

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "frame_time": self.frame_time.isoformat(),
            "delay_s": self.delay_s,
            "width": self.width, "height": self.height,
            "aircraft": [a.to_json() for a in self.aircraft],
        }

    def find(self, track_id: str) -> Optional[FrozenAircraft]:
        return next((a for a in self.aircraft if a.track_id == track_id), None)


def build_capture(image: np.ndarray, frame_time: datetime, states: Sequence,
                  model, delay_s: Optional[float],
                  max_range_km: float = 150.0) -> FreezeCapture:
    """Snapshot every airborne aircraft at ``frame_time``.

    ``states`` are the :class:`TrackState` objects already resolved for this
    frame, so the positions are exactly the ones the overlay drew.
    """
    aircraft: list[FrozenAircraft] = []
    for state in states:
        if state.kind is not TrackKind.AIRCRAFT:
            continue
        labels = state.labels
        position = state.position
        if labels.get("on_ground") or position.alt_m is None or position.alt_m <= 0:
            continue

        speed = position.speed_mps or 0.0
        course = position.course_deg
        ve = speed * math.sin(math.radians(course)) if course is not None else 0.0
        vn = speed * math.cos(math.radians(course)) if course is not None else 0.0
        latest = state.track.latest
        vu = (latest.vertical_rate_mps if latest is not None
              and latest.vertical_rate_mps is not None else 0.0)

        enu = geodetic_to_enu(position.lat, position.lon, position.alt_m,
                              model.ref_lat, model.ref_lon, 0.0)
        rel = model.relative_enu(enu, refract=True)[0]
        range_km = float(np.hypot(rel[0], rel[1])) / 1000.0
        if range_km > max_range_km:
            continue
        uv, in_front, _ = model.project_enu(enu, refract=True)

        label = (labels.get("callsign") or labels.get("registration")
                 or labels.get("icao") or state.id)
        aircraft.append(FrozenAircraft(
            track_id=state.id, label=str(label),
            lat=position.lat, lon=position.lon, alt_m=float(position.alt_m),
            alt_source=str(labels.get("alt_source", "unknown")),
            ve=ve, vn=vn, vu=float(vu),
            range_km=range_km,
            bearing_deg=float(math.degrees(math.atan2(rel[0], rel[1])) % 360.0),
            predicted=((float(uv[0, 0]), float(uv[0, 1])) if in_front[0] else None),
            details={k: labels[k] for k in ("icao", "callsign", "registration", "type")
                     if labels.get(k)},
        ))
    aircraft.sort(key=lambda a: a.range_km)
    return FreezeCapture(id=secrets.token_hex(6), image=image.copy(),
                         frame_time=frame_time, delay_s=delay_s, aircraft=aircraft)


class FreezeRequest:
    """One pending request, fulfilled by the pipeline thread on its next frame."""

    def __init__(self):
        self.done = threading.Event()
        self.capture: Optional[FreezeCapture] = None
        self.error: Optional[str] = None
