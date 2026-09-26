"""Reading and writing ``points.csv``, the hand-collected control points.

Format (header required)::

    name,px,py,lat,lon,elev_m[,enabled,note]

``px``/``py`` are pixel coordinates in the calibration image; ``elev_m`` is the
point's height above mean sea level, which matters a lot for anything close to
the camera and hardly at all for a light on the far shore.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ..geodesy import geodetic_to_enu

FIELDNAMES = ["name", "px", "py", "lat", "lon", "elev_m", "enabled", "note"]


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
