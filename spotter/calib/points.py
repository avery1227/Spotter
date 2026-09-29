"""Reading and writing ``points.csv``, the hand-collected control points.

Format (header required)::

    name,px,py,lat,lon,elev_m[,enabled,note,kind,time,delay_s,ve,vn,vu]

``px``/``py`` are pixel coordinates in the calibration image; ``elev_m`` is the
point's height above mean sea level, which matters a lot for anything close to
the camera and hardly at all for a light on the far shore.

The trailing columns are for aircraft points, clicked on a frozen frame: when
the frame was captured, the ``stream.encoder_delay_s`` in force at the time,
and the aircraft's velocity (east/north/up, m/s). The velocity is what lets the
solver measure a timing error as well as a pointing one: a plane that was
really a second further along its track than the overlay thought moves by its
velocity, not by a rotation of the camera. Older files without these columns
load unchanged.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..geodesy import geodetic_to_enu

FIELDNAMES = ["name", "px", "py", "lat", "lon", "elev_m", "enabled", "note",
              "kind", "time", "delay_s", "ve", "vn", "vu"]

#: A fixed feature: a rock, a light, a roof corner.
KIND_LANDMARK = "landmark"
#: A moving aircraft, positioned from ADS-B at the moment the frame was frozen.
KIND_AIRCRAFT = "aircraft"


def _optional_float(value) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _format_optional(value: Optional[float], digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


@dataclass
class ControlPoint:
    name: str
    px: float
    py: float
    lat: float
    lon: float
    elev_m: float = 0.0
    enabled: bool = True
    note: str = ""
    kind: str = KIND_LANDMARK
    #: ISO 8601 frame time, for aircraft points.
    time: str = ""
    delay_s: Optional[float] = None
    ve: Optional[float] = None
    vn: Optional[float] = None
    vu: Optional[float] = None

    @property
    def velocity(self) -> Optional[tuple[float, float, float]]:
        if self.ve is None or self.vn is None:
            return None
        return (self.ve, self.vn, self.vu or 0.0)

    def as_row(self) -> dict:
        return {
            "name": self.name,
            "px": f"{self.px:.2f}",
            "py": f"{self.py:.2f}",
            "lat": f"{self.lat:.8f}",
            "lon": f"{self.lon:.8f}",
            "elev_m": f"{self.elev_m:.2f}",
            "enabled": "1" if self.enabled else "0",
            "note": self.note,
            "kind": self.kind,
            "time": self.time,
            "delay_s": _format_optional(self.delay_s, 2),
            "ve": _format_optional(self.ve),
            "vn": _format_optional(self.vn),
            "vu": _format_optional(self.vu),
        }


@dataclass
class ControlPointSet:
    """A collection of control points plus the image they were clicked on."""

    points: list[ControlPoint] = field(default_factory=list)
    image_width: Optional[int] = None
    image_height: Optional[int] = None

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    @property
    def active(self) -> list[ControlPoint]:
        return [p for p in self.points if p.enabled]

    def pixels(self, active_only: bool = True) -> np.ndarray:
        pts = self.active if active_only else self.points
        if not pts:
            return np.empty((0, 2))
        return np.array([[p.px, p.py] for p in pts], dtype=float)

    def enu(self, ref_lat: float, ref_lon: float,
            active_only: bool = True) -> np.ndarray:
        """ENU positions of the points relative to sea level at the reference."""
        pts = self.active if active_only else self.points
        if not pts:
            return np.empty((0, 3))
        return geodetic_to_enu(
            [p.lat for p in pts], [p.lon for p in pts], [p.elev_m for p in pts],
            ref_lat, ref_lon, 0.0)

    def velocities(self, active_only: bool = True) -> np.ndarray:
        """ENU velocity (m/s) per point; zero for anything that does not move."""
        pts = self.active if active_only else self.points
        return np.array([p.velocity or (0.0, 0.0, 0.0) for p in pts],
                        dtype=float).reshape(-1, 3)

    def names(self, active_only: bool = True) -> list[str]:
        pts = self.active if active_only else self.points
        return [p.name for p in pts]

    def add(self, point: ControlPoint) -> None:
        self.points.append(point)

    def subset(self, indices: Iterable[int]) -> "ControlPointSet":
        """A new set containing only the given indices *of the active points*."""
        active = self.active
        return ControlPointSet(points=[active[i] for i in indices],
                               image_width=self.image_width,
                               image_height=self.image_height)


def load_points(path: str | Path) -> ControlPointSet:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"control points file not found: {path}")

    points: list[ControlPoint] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty")
        missing = {"name", "px", "py", "lat", "lon"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for lineno, row in enumerate(reader, start=2):
            if not row.get("name") or str(row["name"]).startswith("#"):
                continue
            try:
                enabled_raw = str(row.get("enabled", "1") or "1").strip().lower()
                points.append(ControlPoint(
                    name=row["name"].strip(),
                    px=float(row["px"]),
                    py=float(row["py"]),
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    elev_m=float(row.get("elev_m") or 0.0),
                    enabled=enabled_raw not in ("0", "false", "no", "n"),
                    note=(row.get("note") or "").strip(),
                    kind=(row.get("kind") or KIND_LANDMARK).strip() or KIND_LANDMARK,
                    time=(row.get("time") or "").strip(),
                    delay_s=_optional_float(row.get("delay_s")),
                    ve=_optional_float(row.get("ve")),
                    vn=_optional_float(row.get("vn")),
                    vu=_optional_float(row.get("vu")),
                ))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc

    return ControlPointSet(points=points)


def save_points(path: str | Path, point_set: ControlPointSet) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for point in point_set.points:
            writer.writerow(point.as_row())
