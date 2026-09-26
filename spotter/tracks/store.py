"""Time-indexed track history, with interpolation and dead reckoning.

The central idea of this project: a frame is rendered for *the frame's* wall
clock time, not for "now". Because the pipeline runs a configurable delay behind
real time (``stream.encoder_delay_s`` plus segment buffering), by the time we
draw a frame we usually already hold reports from *after* that frame's
timestamp. That turns the hard problem (predicting where a ship will be) into
the easy one (interpolating between two known positions).

Dead reckoning is therefore the fallback, not the main path. It matters for
aircraft, whose reports can arrive after the frame they belong to, and for AIS
Class B, which can go 30 seconds or more between transmissions.
"""

from __future__ import annotations

import bisect
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

import numpy as np

from ..geodesy import dead_reckon
from ..logging_setup import get_logger
from ..util import lerp_angle, utcnow
from .model import Position, TrackKind, TrackReport

log = get_logger(__name__)


@dataclass
class MotionParams:
    """Per-kind interpolation settings, read from ``tracks.motion.<kind>``."""

    smoothing_tau_s: float = 0.0
    max_extrapolate_s: float = 30.0
    dead_reckon: bool = True

    @classmethod
    def from_config(cls, cfg, kind: TrackKind) -> "MotionParams":
        node = cfg.sub(f"tracks.motion.{kind.value}")
        return cls(
            smoothing_tau_s=float(node.get("smoothing_tau_s", 0.0)),
            max_extrapolate_s=float(node.get("max_extrapolate_s", 30.0)),
            dead_reckon=bool(node.get("dead_reckon", True)),
        )


class Track:
    """One target and everything we have heard about it."""

    __slots__ = ("id", "kind", "labels", "sources", "_reports", "_times",
                 "_smooth_state", "first_seen", "last_update")

    def __init__(self, track_id: str, kind: TrackKind):
        self.id = track_id
        self.kind = kind
        self.labels: dict = {}
        self.sources: set[str] = set()
        self._reports: list[TrackReport] = []
        self._times: list[float] = []          # epoch seconds, kept sorted
        self._smooth_state: Optional[tuple[float, float, float, float]] = None
        self.first_seen: Optional[datetime] = None
        self.last_update: Optional[datetime] = None

    # -- ingestion ----------------------------------------------------------
    def add(self, report: TrackReport) -> None:
        stamp = report.timestamp.timestamp()

        # Feeds repeat and occasionally reorder reports; keep the list sorted
        # and drop exact duplicates so interpolation brackets stay meaningful.
        index = bisect.bisect_left(self._times, stamp)
        if index < len(self._times) and abs(self._times[index] - stamp) < 1e-6:
            self._reports[index] = report
        else:
            self._times.insert(index, stamp)
            self._reports.insert(index, report)

        # Later reports win for descriptive fields, but a field that arrives
        # once (a ship's name, from a type-5 message) must not be erased by the
        # position reports that follow it.
        for key, value in report.labels.items():
            if value not in (None, ""):
                self.labels[key] = value
        if report.source:
            self.sources.add(report.source)

        if self.first_seen is None or report.timestamp < self.first_seen:
            self.first_seen = report.timestamp
        if self.last_update is None or report.timestamp > self.last_update:
            self.last_update = report.timestamp

    def trim(self, before: datetime) -> None:
        """Drop history older than ``before``, always keeping the newest report."""
        cutoff = before.timestamp()
        keep = bisect.bisect_left(self._times, cutoff)
        if keep <= 0:
            return
        keep = min(keep, len(self._times) - 1)
        del self._times[:keep]
        del self._reports[:keep]

    # -- queries ------------------------------------------------------------
    @property
    def latest(self) -> Optional[TrackReport]:
        return self._reports[-1] if self._reports else None

    @property
    def report_count(self) -> int:
        return len(self._reports)

    def age_s(self, now: Optional[datetime] = None) -> float:
        if self.last_update is None:
            return float("inf")
        return ((now or utcnow()) - self.last_update).total_seconds()

    def position_at(self, when: datetime, motion: MotionParams) -> Optional[Position]:
        """Resolve the track's position at ``when``, or None if not covered."""
        if not self._reports:
            return None

        target = when.timestamp()
        index = bisect.bisect_left(self._times, target)

        if index == 0 and target < self._times[0]:
            position = self._before_first(target, motion)
        elif index >= len(self._times):
            position = self._after_last(target, motion)
        elif abs(self._times[index] - target) < 1e-9:
            position = self._exact(index)
        else:
            position = self._interpolate(index - 1, index, target)

        if position is None:
            return None
        return self._smooth(position, target, motion)

    # -- internals ----------------------------------------------------------
    def _exact(self, index: int) -> Position:
        report = self._reports[index]
        return Position(lat=report.lat, lon=report.lon,
                        alt_m=report.alt_m if report.alt_m is not None else 0.0,
                        speed_mps=report.speed_mps, course_deg=report.course_deg,
                        heading_deg=report.heading_deg, mode="report", age_s=0.0)

    def _interpolate(self, i: int, j: int, target: float) -> Position:
        a, b = self._reports[i], self._reports[j]
        ta, tb = self._times[i], self._times[j]
        span = tb - ta
        frac = 0.0 if span <= 0 else (target - ta) / span

        alt_a = a.alt_m if a.alt_m is not None else (b.alt_m or 0.0)
        alt_b = b.alt_m if b.alt_m is not None else alt_a

        return Position(
            lat=a.lat + (b.lat - a.lat) * frac,
            lon=a.lon + (b.lon - a.lon) * frac,
            alt_m=alt_a + (alt_b - alt_a) * frac,
            speed_mps=_lerp_optional(a.speed_mps, b.speed_mps, frac),
            course_deg=_lerp_optional_angle(a.course_deg, b.course_deg, frac),
            heading_deg=_lerp_optional_angle(a.heading_deg, b.heading_deg, frac),
            mode="interpolated",
            age_s=min(target - ta, tb - target),
        )

    def _after_last(self, target: float, motion: MotionParams) -> Optional[Position]:
        report = self._reports[-1]
        gap = target - self._times[-1]
        if gap > motion.max_extrapolate_s:
            return None

        alt = report.alt_m if report.alt_m is not None else 0.0
        if not motion.dead_reckon or not report.speed_mps or report.course_deg is None:
            return Position(lat=report.lat, lon=report.lon, alt_m=alt,
                            speed_mps=report.speed_mps, course_deg=report.course_deg,
                            heading_deg=report.heading_deg, mode="held", age_s=gap)

        lat, lon = dead_reckon(report.lat, report.lon, report.course_deg,
                               report.speed_mps, gap)
        if report.vertical_rate_mps:
            alt = max(0.0, alt + report.vertical_rate_mps * gap)
        return Position(lat=lat, lon=lon, alt_m=alt, speed_mps=report.speed_mps,
                        course_deg=report.course_deg, heading_deg=report.heading_deg,
                        mode="dead_reckoned", age_s=gap)

    def _before_first(self, target: float, motion: MotionParams) -> Optional[Position]:
        """Frame time precedes our earliest report -- reckon backwards a little.

        This happens right after a track appears: the first report we hear may
        be newer than the frame currently being drawn.
        """
        report = self._reports[0]
        gap = self._times[0] - target
        if gap > motion.max_extrapolate_s:
            return None

        alt = report.alt_m if report.alt_m is not None else 0.0
        if not motion.dead_reckon or not report.speed_mps or report.course_deg is None:
            return Position(lat=report.lat, lon=report.lon, alt_m=alt,
                            speed_mps=report.speed_mps, course_deg=report.course_deg,
                            heading_deg=report.heading_deg, mode="held", age_s=gap)

        lat, lon = dead_reckon(report.lat, report.lon, report.course_deg,
                               report.speed_mps, -gap)
        if report.vertical_rate_mps:
            alt = max(0.0, alt - report.vertical_rate_mps * gap)
        return Position(lat=lat, lon=lon, alt_m=alt, speed_mps=report.speed_mps,
                        course_deg=report.course_deg, heading_deg=report.heading_deg,
                        mode="dead_reckoned", age_s=gap)

    def _smooth(self, position: Position, target: float,
                motion: MotionParams) -> Position:
        """Light exponential smoothing to take the step out of report updates.

        Smoothing costs lag of roughly one time constant, which is why the
        aircraft default is a few tenths of a second: enough to stop a label
        twitching on each ADS-B update, not enough to trail a moving target.
        """
        tau = motion.smoothing_tau_s
        if tau <= 0:
            return position

        state = self._smooth_state
        if state is None:
            self._smooth_state = (target, position.lat, position.lon, position.alt_m)
            return position

        prev_t, prev_lat, prev_lon, prev_alt = state
        dt = target - prev_t
        # Going backwards (a seek, or a reconnect that rewinds the clock) or a
        # long jump means the filter's state is meaningless; restart it.
        if dt <= 0 or dt > max(5.0, 10.0 * tau):
            self._smooth_state = (target, position.lat, position.lon, position.alt_m)
            return position

        alpha = 1.0 - float(np.exp(-dt / tau))
        lat = prev_lat + (position.lat - prev_lat) * alpha
        lon = prev_lon + (position.lon - prev_lon) * alpha
        alt = prev_alt + (position.alt_m - prev_alt) * alpha
        self._smooth_state = (target, lat, lon, alt)

        position.lat, position.lon, position.alt_m = lat, lon, alt
        return position


def _lerp_optional(a: Optional[float], b: Optional[float],
                   frac: float) -> Optional[float]:
    if a is None:
        return b
    if b is None:
        return a
    return a + (b - a) * frac


def _lerp_optional_angle(a: Optional[float], b: Optional[float],
                         frac: float) -> Optional[float]:
    if a is None:
        return b
    if b is None:
        return a
    return lerp_angle(a, b, frac)


@dataclass
class TrackState:
    """A track paired with its position at the queried instant."""

    track: Track
    position: Position

    @property
    def id(self) -> str:
        return self.track.id

    @property
    def kind(self) -> TrackKind:
        return self.track.kind

    @property
    def labels(self) -> dict:
        return self.track.labels


class TrackStore:
    """Thread-safe collection of tracks, fed by source threads, read by the renderer."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.history_s = float(cfg.get("tracks.history_s", 900.0))
        self.stale_timeout = {
            TrackKind.AIRCRAFT: float(cfg.get("tracks.stale_timeout_s.aircraft", 30.0)),
            TrackKind.SHIP: float(cfg.get("tracks.stale_timeout_s.ship", 600.0)),
        }
        self.motion = {
            TrackKind.AIRCRAFT: MotionParams.from_config(cfg, TrackKind.AIRCRAFT),
            TrackKind.SHIP: MotionParams.from_config(cfg, TrackKind.SHIP),
        }

        self._lock = threading.RLock()
        self._tracks: dict[str, Track] = {}
        self.total_reports = 0
        self.dropped_reports = 0

    # -- writes -------------------------------------------------------------
    def add_report(self, report: TrackReport) -> bool:
        if not report.is_valid():
            self.dropped_reports += 1
            return False
        with self._lock:
            track = self._tracks.get(report.track_id)
            if track is None:
                track = Track(report.track_id, report.kind)
                self._tracks[report.track_id] = track
            track.add(report)
            self.total_reports += 1
        return True

    def add_reports(self, reports: Iterable[TrackReport]) -> int:
        return sum(1 for report in reports if self.add_report(report))

    def prune(self, now: Optional[datetime] = None) -> int:
        """Drop stale tracks and trim history. Returns how many tracks went away."""
        now = now or utcnow()
        cutoff = now - timedelta(seconds=self.history_s)
        removed = 0
        with self._lock:
            for track_id in list(self._tracks):
                track = self._tracks[track_id]
                timeout = self.stale_timeout.get(track.kind, 300.0)
                if track.age_s(now) > timeout:
                    del self._tracks[track_id]
                    removed += 1
                else:
                    track.trim(cutoff)
        return removed

    # -- reads --------------------------------------------------------------
    def snapshot_at(self, when: datetime) -> list[TrackState]:
        """Every track that has a usable position at ``when``."""
        out: list[TrackState] = []
        with self._lock:
            for track in self._tracks.values():
                motion = self.motion.get(track.kind)
                if motion is None:
                    continue
                position = track.position_at(when, motion)
                if position is not None:
                    out.append(TrackState(track=track, position=position))
        return out

    def get(self, track_id: str) -> Optional[Track]:
        with self._lock:
            return self._tracks.get(track_id)

    def counts(self) -> dict:
        with self._lock:
            aircraft = sum(1 for t in self._tracks.values()
                           if t.kind is TrackKind.AIRCRAFT)
            ships = sum(1 for t in self._tracks.values() if t.kind is TrackKind.SHIP)
        return {"aircraft": aircraft, "ships": ships,
                "total": aircraft + ships, "reports": self.total_reports}

    def __len__(self) -> int:
        with self._lock:
            return len(self._tracks)
