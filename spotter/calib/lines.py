"""Traced line features: the same edge drawn on the camera frame and on the map.

A point needs the *same* spot found in both views, which is hard for anything
without a sharp corner. A line only needs the same *feature*: trace a seawall,
a path edge or a roof line in the frame, trace it again on the satellite map,
and the solver moves the camera until the map line, projected into the frame,
lies on the traced one. Sliding along the line costs nothing, so neither
tracing has to start or stop at the same place -- only the map tracing has to
cover everything traced in the frame.

What the map cannot say is how high the feature is, and close to the camera
height matters: at 100 m a metre of error is half a degree. Each line
therefore carries a height and an uncertainty. Zero uncertainty means exact
(the waterline, give or take the tide); anything else lets the solver fit the
line's height within that range, because a single edge sits at one height.

Stored as JSON (``calibration.lines``), since a line is a list of vertices.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..geodesy import geodetic_to_enu


@dataclass
class LineFeature:
    name: str
    #: Traced in the frame, full-resolution pixels.
    image: list[tuple[float, float]]
    #: Traced on the map, (lat, lon).
    map: list[tuple[float, float]]
    elev_m: float = 0.0
    #: 0 = height known exactly; otherwise fit within about this many metres.
    elev_sigma_m: float = 1.0
    enabled: bool = True
    note: str = ""

    def validate(self) -> None:
        if len(self.image) < 2:
            raise ValueError(f"line '{self.name}' needs at least 2 points in the frame")
        if len(self.map) < 2:
            raise ValueError(f"line '{self.name}' needs at least 2 points on the map")
        for lat, lon in self.map:
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f"line '{self.name}' has coordinates out of range")
        if self.elev_sigma_m < 0:
            raise ValueError("elev_sigma_m cannot be negative")

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "image": [[round(float(x), 2), round(float(y), 2)] for x, y in self.image],
            "map": [[round(float(a), 8), round(float(b), 8)] for a, b in self.map],
            "elev_m": float(self.elev_m),
            "elev_sigma_m": float(self.elev_sigma_m),
            "enabled": bool(self.enabled),
            "note": self.note,
        }

    @classmethod
    def from_json(cls, data: dict) -> "LineFeature":
        line = cls(
            name=str(data.get("name") or "line").strip() or "line",
            image=[(float(x), float(y)) for x, y in data.get("image") or []],
            map=[(float(a), float(b)) for a, b in data.get("map") or []],
            elev_m=float(data.get("elev_m") or 0.0),
            elev_sigma_m=float(data.get("elev_sigma_m", 1.0) or 0.0),
            enabled=bool(data.get("enabled", True)),
            note=str(data.get("note") or ""),
        )
        line.validate()
        return line

    def map_enu(self, ref_lat: float, ref_lon: float) -> np.ndarray:
        lat = [p[0] for p in self.map]
        lon = [p[1] for p in self.map]
        return geodetic_to_enu(lat, lon, [self.elev_m] * len(lat), ref_lat, ref_lon, 0.0)


@dataclass
class LineSet:
    lines: list[LineFeature] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.lines)

    @property
    def active(self) -> list[LineFeature]:
        return [line for line in self.lines if line.enabled]


def load_lines(path: Optional[str | Path]) -> LineSet:
    """Read ``lines.json``; a missing file is simply no lines."""
    if path is None:
        return LineSet()
    path = Path(path)
    if not path.is_file():
        return LineSet()
    data = json.loads(path.read_text(encoding="utf-8") or "{}")
    return LineSet(lines=[LineFeature.from_json(entry)
                          for entry in data.get("lines", [])])


def save_lines(path: str | Path, line_set: LineSet) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        # Tracing is slow work; keep the previous version next to the points
        # history so a bad edit is recoverable.
        history = path.parent / "state" / "lines_history"
        history.mkdir(parents=True, exist_ok=True)
        target = history / f"lines-{time.strftime('%Y%m%d-%H%M%S')}.json"
        if not target.exists():
            target.write_bytes(path.read_bytes())
        for stale in sorted(history.glob("lines-*.json"))[:-40]:
            stale.unlink()
    payload = {"lines": [line.to_json() for line in line_set.lines]}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)
