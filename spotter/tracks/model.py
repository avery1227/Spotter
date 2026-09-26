"""The normalised track model.

Every source -- readsb, adsb.lol, AIS-catcher, aisstream -- is reduced to
:class:`TrackReport`. Nothing downstream should ever need to know which feed a
target came from, except to attribute it on screen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

KNOTS_TO_MPS = 0.514444
FEET_TO_M = 0.3048
FPM_TO_MPS = 0.00508


class TrackKind(str, Enum):
    AIRCRAFT = "aircraft"
    SHIP = "ship"

    def __str__(self) -> str:  # so f-strings and config lookups read naturally
        return self.value


@dataclass
class TrackReport:
    """One position report from one source, in SI units."""

    track_id: str
    kind: TrackKind
    timestamp: datetime
    lat: float
    lon: float

    alt_m: Optional[float] = None
    speed_mps: Optional[float] = None
    course_deg: Optional[float] = None      # course over ground
    heading_deg: Optional[float] = None     # true heading (may differ from course)
    vertical_rate_mps: Optional[float] = None

    #: Free-form descriptive fields: callsign, registration, type, name,
    #: destination, mmsi, icao, category, flight, operator...
    labels: dict[str, Any] = field(default_factory=dict)
    source: str = ""

    def is_valid(self) -> bool:
        return (
            -90.0 <= self.lat <= 90.0
            and -180.0 <= self.lon <= 180.0
            # (0, 0) is the classic "no fix yet" sentinel in both AIS and ADS-B.
            and not (abs(self.lat) < 1e-6 and abs(self.lon) < 1e-6)
            and math.isfinite(self.lat) and math.isfinite(self.lon)
        )


@dataclass
class Position:
    """A track's state resolved to one instant."""

    lat: float
    lon: float
    alt_m: float
    speed_mps: Optional[float] = None
    course_deg: Optional[float] = None
    heading_deg: Optional[float] = None
    #: How the value was produced, for the debug overlay.
    mode: str = "interpolated"              # interpolated | dead_reckoned | held
    #: Seconds between the query time and the nearest real report.
    age_s: float = 0.0


def category_of(labels: dict, kind: TrackKind) -> str:
    """Map a track to a colour/priority category used by the renderer."""
    if kind is TrackKind.AIRCRAFT:
        if labels.get("military"):
            return "aircraft_military"
        return "aircraft"

    # AIS ship_type is a two-digit code; the tens digit carries the class.
    ship_type = labels.get("ship_type")
    try:
        code = int(ship_type)
    except (TypeError, ValueError):
        return "ship"
    if 60 <= code <= 69:
        return "ship_passenger"
    if 80 <= code <= 89:
        return "ship_tanker"
    if 70 <= code <= 79:
        return "ship_cargo"
    return "ship"


def default_height_m(labels: dict, kind: TrackKind, fallback: float) -> float:
    """Best guess at a surface target's above-waterline height, for horizon tests.

    Only a rough class-based estimate: what matters is distinguishing a kayak
    from a container ship when deciding whether the target is over the horizon.
    """
    if kind is TrackKind.AIRCRAFT:
        return fallback
    try:
        code = int(labels.get("ship_type"))
    except (TypeError, ValueError):
        code = -1
    length = labels.get("length_m")
    if isinstance(length, (int, float)) and length > 0:
        # Air draught scales roughly with the cube root of displacement; this
        # linear approximation is close enough over the range that matters.
        return max(3.0, min(60.0, 0.22 * float(length) + 4.0))
    if 60 <= code <= 69:
        return 25.0     # passenger / ferry
    if 70 <= code <= 89:
        return 30.0     # cargo / tanker
    if 30 <= code <= 39:
        return 8.0      # fishing / tug / sailing
    return fallback
