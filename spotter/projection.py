"""Project tracks into the frame and decide which ones are actually visible.

Four independent reasons a target gets culled:

1. It is behind the camera.
2. It projects outside the frame (plus a margin, so labels can hang off-edge).
3. It is nearer or further than the configured range limits.
4. It is geometrically below the horizon -- a ship hull-down beyond line of
   sight. This one matters on Long Island Sound: from 15 m up, the horizon is
   under 15 km, but a container ship's superstructure stays visible well past
   that, so the test has to account for the target's own height.

Everything is done on whole arrays: a few hundred targets per frame at 30 fps
is enough that per-target Python would show up in the frame budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .calib.model import CameraModel
from .geodesy import (bearing_deg, elevation_deg, geodetic_to_enu,
                      horizon_distance_m, is_above_horizon)
from .logging_setup import get_logger
from .tracks.model import TrackKind, category_of, default_height_m
from .tracks.store import TrackState

log = get_logger(__name__)


@dataclass
class ProjectedTarget:
    """A track resolved to a pixel position, with the geometry that got it there."""

    state: TrackState
    u: float
    v: float
    range_m: float              # slant range from the camera
    ground_range_m: float
    bearing_deg: float
    elevation_deg: float
    category: str
    #: True when the pixel lies inside the frame proper (not just the margin).
    on_screen: bool = True

    @property
    def id(self) -> str:
        return self.state.id

    @property
    def kind(self) -> TrackKind:
        return self.state.kind

    @property
    def labels(self) -> dict:
        return self.state.labels


@dataclass
class CullStats:
    """Why targets did not make it, for the debug overlay and logs."""

    total: int = 0
    behind: int = 0
    out_of_frame: int = 0
    out_of_range: int = 0
    below_horizon: int = 0
    visible: int = 0

    def as_dict(self) -> dict:
        return {"total": self.total, "visible": self.visible,
                "behind": self.behind, "out_of_frame": self.out_of_frame,
                "out_of_range": self.out_of_range,
                "below_horizon": self.below_horizon}


class TargetProjector:
    """Turns :class:`TrackState` objects into :class:`ProjectedTarget` objects."""

    def __init__(self, cfg, model: CameraModel):
        self.cfg = cfg
        self.model = model
        projection = cfg.sub("projection")
        self.max_range_m = float(projection.get("max_range_m", 80000))
        self.min_range_m = float(projection.get("min_range_m", 50))
        self.margin_px = float(projection.get("frame_margin_px", 80))
        self.horizon_culling = bool(projection.get("horizon_culling", True))
        self.horizon_slack_m = float(projection.get("horizon_slack_m", 2.0))
        self.default_ship_height_m = float(
            projection.get("default_ship_height_m", 12.0))
        self.last_stats = CullStats()

    def set_model(self, model: CameraModel) -> None:
        self.model = model

    @property
    def horizon_km(self) -> float:
        return horizon_distance_m(self.model.height_m,
                                  self.model.refraction_k) / 1000.0

    def project(self, states: Sequence[TrackState]) -> list[ProjectedTarget]:
        stats = CullStats(total=len(states))
        if not states:
            self.last_stats = stats
            return []

        model = self.model
        lat = np.fromiter((s.position.lat for s in states), float, len(states))
        lon = np.fromiter((s.position.lon for s in states), float, len(states))
        alt = np.fromiter((s.position.alt_m for s in states), float, len(states))

        enu = geodetic_to_enu(lat, lon, alt, model.ref_lat, model.ref_lon, 0.0)
        relative = model.relative_enu(enu, refract=True)
        ground_range = np.hypot(relative[:, 0], relative[:, 1])
        slant_range = np.linalg.norm(relative, axis=1)

        uv, in_front, _depth = model.project_enu(enu, refract=True)

        keep = in_front.copy()
        stats.behind = int(np.sum(~keep))

        range_ok = (slant_range <= self.max_range_m) & (slant_range >= self.min_range_m)
        stats.out_of_range = int(np.sum(keep & ~range_ok))
        keep &= range_ok

        inside = ((uv[:, 0] >= -self.margin_px)
                  & (uv[:, 0] <= model.width + self.margin_px)
                  & (uv[:, 1] >= -self.margin_px)
                  & (uv[:, 1] <= model.height + self.margin_px))
        stats.out_of_frame = int(np.sum(keep & ~inside))
        keep &= inside

        if self.horizon_culling:
            visible = self._horizon_mask(states, ground_range, alt)
            stats.below_horizon = int(np.sum(keep & ~visible))
            keep &= visible

        bearings = bearing_deg(relative)
        elevations = elevation_deg(relative)

        out: list[ProjectedTarget] = []
        for index in np.flatnonzero(keep):
            state = states[index]
            u, v = float(uv[index, 0]), float(uv[index, 1])
            out.append(ProjectedTarget(
                state=state, u=u, v=v,
                range_m=float(slant_range[index]),
                ground_range_m=float(ground_range[index]),
                bearing_deg=float(bearings[index]),
                elevation_deg=float(elevations[index]),
                category=category_of(state.labels, state.kind),
                on_screen=bool(0 <= u <= model.width and 0 <= v <= model.height),
            ))

        stats.visible = len(out)
        self.last_stats = stats
        return out

    def _horizon_mask(self, states, ground_range: np.ndarray,
                      alt: np.ndarray) -> np.ndarray:
        """True where the target is above the geometric horizon.

        Aircraft are only tested on their real altitude. Surface targets get a
        class-based estimate of how far their superstructure reaches above the
        waterline, because AIS reports a position, not an air draught.
        """
        heights = np.empty(len(states), dtype=float)
        for i, state in enumerate(states):
            if state.kind is TrackKind.AIRCRAFT:
                heights[i] = max(alt[i], 0.0)
            else:
                heights[i] = default_height_m(state.labels, state.kind,
                                              self.default_ship_height_m)
        return is_above_horizon(ground_range, heights, self.model.height_m,
                                self.model.refraction_k, self.horizon_slack_m)


#: Rare and brief, so they are never the labels the cap drops. Each layer
#: limits its own count, so these cannot crowd everything else out.
PRIORITY_KINDS = (TrackKind.SATELLITE, TrackKind.LIGHTNING)


def sort_by_priority(targets: Sequence[ProjectedTarget]) -> list[ProjectedTarget]:
    """Sky events first, then nearest first. Decides which labels survive the cap."""
    return sorted(targets, key=lambda t: (t.kind not in PRIORITY_KINDS, t.range_m))
