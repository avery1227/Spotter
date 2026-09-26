"""Offline test mode: a recorded clip plus recorded track data.

Lets the overlay be tuned -- label layout, colours, ``encoder_delay_s``,
calibration -- without depending on the live stream, and makes runs repeatable,
which the live stream never is.

The clip's first frame is mapped to a wall-clock instant (``offline.start_time``,
or the first recorded report's timestamp). Every frame after that gets a
timestamp derived from its position in the clip, so the exact same code path
that resolves track positions for live frames resolves them here.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .ingest.types import PIXEL_FORMAT, StreamInfo, VideoFrameEvent
from .logging_setup import get_logger
from .tracks.model import TrackKind, TrackReport
from .tracks.store import TrackStore

log = get_logger(__name__)


def load_recorded_tracks(path: str | Path) -> list[TrackReport]:
    """Read a JSONL track recording written by ``tools/record_tracks.py``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"recorded track file not found: {path}")

    reports: list[TrackReport] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
                timestamp = row["timestamp"]
                when = (datetime.fromtimestamp(timestamp, tz=timezone.utc)
                        if isinstance(timestamp, (int, float))
                        else datetime.fromisoformat(
                            str(timestamp).replace("Z", "+00:00")))
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                reports.append(TrackReport(
                    track_id=row["track_id"],
                    kind=TrackKind(row.get("kind", "aircraft")),
                    timestamp=when.astimezone(timezone.utc),
                    lat=float(row["lat"]), lon=float(row["lon"]),
                    alt_m=row.get("alt_m"),
                    speed_mps=row.get("speed_mps"),
                    course_deg=row.get("course_deg"),
                    heading_deg=row.get("heading_deg"),
                    vertical_rate_mps=row.get("vertical_rate_mps"),
                    labels=row.get("labels") or {},
                    source=row.get("source", "recording"),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                log.warning("skipping malformed recorded report",
                            extra={"line": lineno, "error": str(exc)})

    reports.sort(key=lambda r: r.timestamp)
    log.info("loaded recorded tracks", extra={
        "path": str(path), "reports": len(reports),
        "span_s": (round((reports[-1].timestamp - reports[0].timestamp)
                         .total_seconds(), 1) if len(reports) > 1 else 0)})
    return reports


def dump_report(report: TrackReport) -> str:
    """Serialise a report as one JSONL line."""
    return json.dumps({
        "track_id": report.track_id,
        "kind": report.kind.value,
        "timestamp": report.timestamp.timestamp(),
        "lat": report.lat, "lon": report.lon, "alt_m": report.alt_m,
        "speed_mps": report.speed_mps, "course_deg": report.course_deg,
        "heading_deg": report.heading_deg,
        "vertical_rate_mps": report.vertical_rate_mps,
        "labels": report.labels, "source": report.source,
    }, default=str)


class OfflineTrackFeeder:
    """Replays recorded reports into the store as the clip's clock passes them.

    Reports are released only once the frame clock reaches them, so the store
    contains exactly what it would have held live -- including the gaps. Feeding
    everything up front would make interpolation look better than it really is.
    """

    def __init__(self, reports: list[TrackReport], store: TrackStore,
                 lookahead_s: float = 30.0, rewind_threshold_s: float = 1.0):
        self.reports = reports
        self.store = store
        # Frames are rendered for a time in the past, so the store legitimately
        # holds reports newer than the frame being drawn. Mirror that here.
        self.lookahead_s = lookahead_s
        self.rewind_threshold_s = rewind_threshold_s
        self._index = 0
        self._last_when: Optional[datetime] = None

    def advance_to(self, when: datetime) -> int:
        # The clip loops back to its start, so the clock jumps backwards. When
        # it does, rewind and replay the same reports rather than sitting at
        # the end of the recording with nothing left to release.
        if self._last_when is not None and when < self._last_when - timedelta(
                seconds=self.rewind_threshold_s):
            log.info("offline clock rewound; replaying recorded tracks",
                     extra={"from": self._last_when.isoformat(),
                            "to": when.isoformat()})
            self.reset()
        self._last_when = when

        cutoff = when + timedelta(seconds=self.lookahead_s)
        released = 0
        while (self._index < len(self.reports)
               and self.reports[self._index].timestamp <= cutoff):
            self.store.add_report(self.reports[self._index])
            self._index += 1
            released += 1
        return released

    def reset(self) -> None:
        self._index = 0
        self._last_when = None

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self.reports)


class OfflinePlayer:
    """Decodes a local clip, stamping frames onto a synthetic wall clock."""

    def __init__(self, cfg, stop_event: Optional[threading.Event] = None):
        self.cfg = cfg
        self.stop_event = stop_event or threading.Event()
        self.path = cfg.path("offline.video", "./data/offline/clip.mp4")
        self.loop = bool(cfg.get("offline.loop", True))
        self.speed = float(cfg.get("offline.speed", 1.0))
        # On loop, restart the synthetic clock at start_time so the same window
        # replays with the same tracks. Advancing it instead would run the clip
        # past the end of the recorded track data and the overlay would empty
        # out -- the opposite of what a repeatable tuning loop is for.
        self.advance_clock_on_loop = bool(
            cfg.get("offline.advance_clock_on_loop", False))
        self.encoder_delay_s = float(cfg.get("stream.encoder_delay_s", 0.0))
        self.info = StreamInfo()

        start = cfg.get("offline.start_time")
        self.start_time: Optional[datetime] = None
        if start:
            parsed = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
            self.start_time = (parsed.replace(tzinfo=timezone.utc)
                               if parsed.tzinfo is None else parsed).astimezone(
                                   timezone.utc)

    def set_start_time(self, when: datetime) -> None:
        self.start_time = when

    def frames(self) -> Iterator[VideoFrameEvent]:
        """Yield frames, looping if configured, paced to ``offline.speed``."""
        import av

        if not self.path or not Path(self.path).is_file():
            raise FileNotFoundError(f"offline clip not found: {self.path}")
        if self.start_time is None:
            self.start_time = datetime.now(timezone.utc)

        pass_index = 0
        clip_duration = 0.0

        while not self.stop_event.is_set():
            wall_start = time.monotonic()
            with av.open(str(self.path)) as container:
                if not container.streams.video:
                    raise ValueError(f"{self.path} has no video stream")
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                self._update_info(stream, container)

                offset_base = (pass_index * clip_duration
                               if self.advance_clock_on_loop else 0.0)
                last_offset = 0.0
                first = True

                for frame in container.decode(stream):
                    if self.stop_event.is_set():
                        return
                    offset = float(frame.pts * stream.time_base) if frame.pts \
                        is not None else last_offset
                    last_offset = offset

                    if self.speed > 0:
                        # Play at wall-clock pace so stalls, fades and the
                        # drift timer behave the way they will in production.
                        target = wall_start + offset / self.speed
                        delay = target - time.monotonic()
                        if delay > 0 and self.stop_event.wait(delay):
                            return

                    image = frame.to_ndarray(format=PIXEL_FORMAT)
                    if not image.flags.writeable or not image.flags.c_contiguous:
                        image = np.ascontiguousarray(image)

                    yield VideoFrameEvent(
                        image=image,
                        wall_time=(self.start_time
                                   + timedelta(seconds=offset_base + offset)
                                   - timedelta(seconds=self.encoder_delay_s)),
                        offset_s=offset,
                        sequence=pass_index,
                        time_is_authoritative=False,
                        discontinuity=first,
                    )
                    first = False

                clip_duration = max(clip_duration, last_offset + 1.0 / max(
                    self.info.fps, 1.0))

            if not self.loop:
                return
            pass_index += 1
            log.info("offline clip looped", extra={"pass": pass_index,
                                                   "duration_s": round(clip_duration, 2)})

    def _update_info(self, stream, container) -> None:
        if self.info.is_ready():
            return
        fps = 0.0
        for candidate in (stream.average_rate, stream.guessed_rate, stream.base_rate):
            if candidate:
                fps = float(candidate)
                break
        audio = container.streams.audio[0] if container.streams.audio else None
        self.info = StreamInfo(
            width=int(stream.codec_context.width),
            height=int(stream.codec_context.height),
            fps=fps or 30.0,
            codec=stream.codec_context.name,
            has_audio=audio is not None,
            audio_codec=audio.codec_context.name if audio is not None else "",
            audio_rate=int(audio.codec_context.sample_rate) if audio else 0,
        )
        log.info("offline clip parameters", extra={
            "width": self.info.width, "height": self.info.height,
            "fps": round(self.info.fps, 3), "path": str(self.path)})

    def stop(self) -> None:
        self.stop_event.set()
