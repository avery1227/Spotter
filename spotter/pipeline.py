"""The pipeline: ingest -> tracks -> projection -> overlay -> output.

One frame's journey:

1. A segment is downloaded and decoded; each frame carries the wall-clock time
   it was captured (PDT + offset - ``encoder_delay_s``).
2. The track store is asked where every target was *at that instant* -- usually
   an interpolation between two real reports, because the pipeline runs behind
   real time.
3. Targets are projected through the calibrated camera model and culled.
4. Skia draws markers and labels straight into the decoded buffer.
5. The frame is handed to the encoder's paced writer.

The loop is deliberately single-threaded from decode to submit. Back-pressure
from the encoder's bounded queue is what keeps the whole thing in step and
memory flat; adding threads between these stages would only add buffering.
"""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator, Optional

from .calib.model import CameraModel
from .drift import DriftDetector
from .ingest.decoder import SegmentDecodeError, SegmentDecoder
from .ingest.hls import HLSReader
from .ingest.types import StreamInfo, VideoFrameEvent
from .logging_setup import get_logger
from .output.encoder import AUDIO_PROBE_BYTES, RTMPOutput
from .projection import TargetProjector
from .render.overlay import OverlayRenderer, OverlayStatus
from .sky.layers import SkyLayers
from .tracks.manager import TrackManager
from .util import utcnow

log = get_logger(__name__)

STATUS_INTERVAL_S = 30.0

#: Upper bound on buffered priming audio, so a stream whose video never decodes
#: cannot grow this without limit.
AUDIO_PRIME_LIMIT_BYTES = 256 * 1024

#: Frames discarded while waiting for enough audio to prime ffmpeg's probe
#: before we give up waiting and start video-only.
MAX_FRAMES_AWAITING_AUDIO = 240


@dataclass
class PipelineStats:
    frames: int = 0
    started_at: float = field(default_factory=time.monotonic)
    decode_errors: int = 0
    last_frame_time: Optional[datetime] = None
    render_ms_ewma: float = 0.0

    @property
    def uptime_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def fps(self) -> float:
        return self.frames / self.uptime_s if self.uptime_s > 0 else 0.0


class Pipeline:
    """Wires every stage together and runs the frame loop."""

    def __init__(self, cfg, stop_event: Optional[threading.Event] = None,
                 frame_hub=None):
        self.cfg = cfg
        #: When the web UI is attached, every composited frame is published
        #: here. Publishing is a reference copy, so it costs nothing when
        #: nobody is watching; JPEG encoding happens lazily on request.
        self.frame_hub = frame_hub
        self.stop_event = stop_event or threading.Event()
        self.offline = bool(cfg.get("offline.enabled", False))

        self.model: Optional[CameraModel] = None
        self.projector: Optional[TargetProjector] = None
        self.renderer: Optional[OverlayRenderer] = None
        self.output: Optional[RTMPOutput] = None

        self.tracks = TrackManager(cfg)
        self.sky = SkyLayers(cfg)
        self.drift = DriftDetector(cfg)
        self.stats = PipelineStats()

        self._reader: Optional[HLSReader] = None
        self._decoder: Optional[SegmentDecoder] = None
        self._player = None
        self._feeder = None
        self._stall_base = None   # clean copy of the held frame during a stall
        self._last_status_log = 0.0
        self._hide_labels_on_drift = bool(cfg.get("drift.hide_labels_on_drift", False))
        self._audio_prime: list[bytes] = []
        self._audio_primed_bytes = 0
        self._frames_before_start = 0
        #: True while the web UI is up but there is no calibration to project
        #: through yet. The monitor page shows this instead of empty stats.
        self.waiting_for_calibration = False

    # -- setup --------------------------------------------------------------
    def _load_calibration(self, width: int, height: int) -> CameraModel:
        path = self.cfg.path("calibration.path", "./calibration.json")
        if path is None or not path.is_file():
            raise FileNotFoundError(
                f"calibration not found at {path}. Run 'python calibrate.py solve' "
                f"first -- without it there is nothing to project through.")
        model = CameraModel.load(path)
        if (model.width, model.height) != (width, height):
            log.info("rescaling calibration to the live frame size", extra={
                "calibrated": f"{model.width}x{model.height}",
                "stream": f"{width}x{height}"})
            model = model.scaled_to(width, height)
        log.info("calibration loaded", extra={
            "path": str(path), "yaw_deg": round(model.yaw_deg, 3),
            "pitch_deg": round(model.pitch_deg, 3),
            "hfov_deg": round(model.hfov_deg, 3),
            "height_m": round(model.height_m, 2),
            "rms_px": model.meta.get("rms_px")})
        return model

    def _build_stages(self, info: StreamInfo) -> None:
        width = int(self.cfg.get("video.width") or info.width)
        height = int(self.cfg.get("video.height") or info.height)
        fps = float(self.cfg.get("video.fps") or info.fps or 30.0)

        self.model = self._load_calibration(width, height)
        self.projector = TargetProjector(self.cfg, self.model)
        self.renderer = OverlayRenderer(self.cfg, self.model)

        # An AAC-LC frame is 1024 samples, so the primed frame count converts
        # directly to the seconds of audio that precede the first video frame.
        audio_lead_s = 0.0
        if self._decoder is not None and info.audio_rate:
            audio_lead_s = (self._decoder.audio_packets * 1024.0) / info.audio_rate

        self.output = RTMPOutput(
            self.cfg, width, height, fps,
            audio_codec=info.audio_codec, audio_rate=info.audio_rate,
            audio_channels=info.audio_channels,
            audio_lead_s=audio_lead_s).start()

        # Hand over the audio buffered while ffmpeg was starting, then point
        # the decoder's remuxer straight at the pipe.
        if self.output.audio_enabled and self._audio_prime:
            primed = b"".join(self._audio_prime)
            log.info("priming audio pipe", extra={
                "bytes": len(primed), "lead_s": round(audio_lead_s, 3)})
            self.output.write_audio(primed)
        self._audio_prime = []
        if self._decoder is not None:
            self._decoder.set_audio_sink(
                self.output.write_audio if self.output.audio_enabled
                else (lambda _data: None))

        log.info("pipeline ready", extra={
            "size": f"{width}x{height}", "fps": round(fps, 3),
            "horizon_km": round(self.projector.horizon_km, 2),
            "encoder_delay_s": self.cfg.get("stream.encoder_delay_s"),
            "drift": self.drift.active})

    # -- frame sources ------------------------------------------------------
    def _live_frames(self) -> Iterator[VideoFrameEvent]:
        self._reader = HLSReader(self.cfg, self.stop_event).start()
        self._decoder = SegmentDecoder(self.cfg, audio_sink=self._on_audio)

        for payload in self._reader.segments(timeout=1.0):
            if self.stop_event.is_set():
                return
            if payload is None:
                # Idle tick: nothing decoded. The writer thread keeps emitting
                # the held frame; keep its badge and the web preview current.
                self._on_input_idle()
                continue
            try:
                for event in self._decoder.decode(payload):
                    yield event
                    if self.stop_event.is_set():
                        return
            except SegmentDecodeError as exc:
                self.stats.decode_errors += 1
                log.warning("segment decode failed", extra={"error": str(exc)})
            except Exception:
                self.stats.decode_errors += 1
                log.exception("unexpected decode error")

    def _offline_frames(self) -> Iterator[VideoFrameEvent]:
        from .offline import OfflinePlayer, OfflineTrackFeeder, load_recorded_tracks

        self._player = OfflinePlayer(self.cfg, self.stop_event)

        tracks_path = self.cfg.path("offline.tracks")
        if tracks_path and tracks_path.is_file():
            reports = load_recorded_tracks(tracks_path)
            if reports and self._player.start_time is None:
                self._player.set_start_time(reports[0].timestamp)
            self._feeder = OfflineTrackFeeder(reports, self.tracks.store)
        else:
            log.warning("offline mode without recorded tracks; the overlay will "
                        "be empty unless live sources are also enabled",
                        extra={"path": str(tracks_path)})

        yield from self._player.frames()

    def _on_audio(self, data: bytes) -> None:
        """Receive remuxed ADTS bytes from the decoder.

        Before the encoder exists these are buffered, then flushed the instant
        it starts. ffmpeg opens its inputs in order and will not read a video
        frame until it has probed the audio pipe -- and the only thing producing
        audio is the same decode loop that is blocked submitting video. Starting
        ffmpeg with an empty audio pipe deadlocks the two against each other.
        """
        if self.output is None:
            if self._audio_primed_bytes < AUDIO_PRIME_LIMIT_BYTES:
                self._audio_prime.append(data)
                self._audio_primed_bytes += len(data)
            return
        self.output.write_audio(data)

    def _ready_to_start_output(self, event: VideoFrameEvent) -> bool:
        """Whether enough audio is buffered for ffmpeg's probe to succeed."""
        info = self._source_info()
        if self.offline or not info.has_audio:
            return True
        if str(self.cfg.get("video.audio", "passthrough")) != "passthrough":
            return True
        if self._audio_primed_bytes >= AUDIO_PROBE_BYTES * 1.5:
            return True

        # Safety valve: never wait forever on a stream whose audio never
        # materialises. A few seconds of dropped frames beats not starting.
        self._frames_before_start += 1
        if self._frames_before_start > MAX_FRAMES_AWAITING_AUDIO:
            log.warning("starting encoder without a full audio prime", extra={
                "primed_bytes": self._audio_primed_bytes,
                "frames_discarded": self._frames_before_start})
            return True
        return False

    def _source_info(self) -> StreamInfo:
        """Block until the frame source reveals its resolution and rate."""
        if self.offline:
            return self._player.info if self._player else StreamInfo()
        return self._decoder.info if self._decoder else StreamInfo()

    # -- main loop ----------------------------------------------------------
    def _wait_for_calibration(self) -> bool:
        """Block until calibration.json exists. Returns False if we should stop.

        Without a calibration there is nothing to project through, so the
        pipeline cannot run. But exiting is the wrong answer when the web UI is
        attached: the UI is how you *make* a calibration, and under a panel or a
        restart policy an exit becomes a crash loop that takes the UI down with
        it every few seconds. So with the UI up we stay alive, and start the
        moment a calibration is saved.
        """
        path = self.cfg.path("calibration.path", "./calibration.json")
        if path is not None and path.is_file():
            return True

        if self.frame_hub is None:
            log.error("no calibration; nothing to project through. Run with "
                      "--web and calibrate in the browser, or use calibrate.py",
                      extra={"path": str(path)})
            return False

        self.waiting_for_calibration = True
        log.warning("no calibration yet -- the web UI is up. Open it, calibrate, "
                    "and the pipeline will start by itself",
                    extra={"path": str(path)})
        last_reminder = time.monotonic()
        while not self.stop_event.wait(2.0):
            if path.is_file():
                log.info("calibration found; starting the pipeline",
                         extra={"path": str(path)})
                self.waiting_for_calibration = False
                return True
            if time.monotonic() - last_reminder > 120:
                last_reminder = time.monotonic()
                log.info("still waiting for a calibration", extra={"path": str(path)})
        return False

    def run(self) -> int:
        if not self._wait_for_calibration():
            stopped = self.stop_event.is_set()
            self.shutdown()
            # Asked to stop while waiting is a clean exit; no calibration and no
            # UI to fix it with is a configuration error.
            return 0 if stopped else 2

        log.info("starting pipeline", extra={
            "mode": "offline" if self.offline else "live",
            "url": self.cfg.get("stream.url") if not self.offline
            else str(self.cfg.path("offline.video"))})

        if not self.offline or self.cfg.get("offline.live_tracks", False):
            self.tracks.start()
        # Satellites are computed from the frame's own time, so they work in
        # offline replays too; lightning simply stays empty there.
        self.sky.start()

        frames = self._offline_frames() if self.offline else self._live_frames()
        exit_code = 0
        try:
            for event in frames:
                if self.stop_event.is_set():
                    break
                if self.output is None:
                    # The decoder learns the stream's parameters as it opens the
                    # first segment; fall back to the frame itself if it did not.
                    info = self._source_info()
                    if not info.is_ready():
                        info = StreamInfo(width=event.width, height=event.height,
                                          fps=30.0)
                    if not self._ready_to_start_output(event):
                        continue    # discard this frame; keep buffering audio
                    self._build_stages(info)
                self._process_frame(event)
                self._periodic(event)
        except KeyboardInterrupt:
            log.info("interrupted")
        except FileNotFoundError as exc:
            log.error("cannot start", extra={"error": str(exc)})
            exit_code = 2
        except Exception:
            log.exception("pipeline failed")
            exit_code = 1
        finally:
            self.shutdown()
        return exit_code

    def _on_input_idle(self) -> None:
        """Redraw the held frame with a live RECONNECTING badge during a stall.

        Without this the badge never appears: it is drawn by _process_frame,
        which is exactly what stops running when the source stalls. The web
        preview would also freeze, and the wall display would keep tearing
        down its stream thinking the connection had died.
        """
        if self.output is None or not self.output.should_show_badge():
            return
        if self._stall_base is None:
            held = self.output.held_frame()
            if held is None:
                return
            self._stall_base = held.copy()   # clean, badge-free original

        frame = self._stall_base.copy()
        self.renderer.render_badges(frame, OverlayStatus(
            reconnecting=True,
            reconnecting_since_s=self.output.starved_for_s(),
            drift_flagged=self.drift.flagged,
            drift_shift_px=self.drift.shift_px,
            labels_hidden=self.drift.flagged and self._hide_labels_on_drift,
        ))
        self.output.hold_frame(frame)
        if self.frame_hub is not None:
            last = self.stats.last_frame_time
            self.frame_hub.publish(frame, {
                "frame_time": last.isoformat() if last else None,
                "visible": 0,
                "stalled": True,
            })

    def _process_frame(self, event: VideoFrameEvent) -> None:
        started = time.perf_counter()
        self._stall_base = None

        if self._feeder is not None:
            self._feeder.advance_to(event.wall_time)

        states = self.tracks.store.snapshot_at(event.wall_time)
        targets = self.projector.project(states)
        sky_targets, effects = self.sky.project(event.wall_time, self.model)
        targets = targets + sky_targets

        report = self.drift.check(event.image) if self.drift.active else None
        if report is not None and report.flagged:
            log.debug("drift check result", extra=report.to_dict())

        status = OverlayStatus(
            reconnecting=self.output.should_show_badge(),
            reconnecting_since_s=self.output.starved_for_s(),
            drift_flagged=self.drift.flagged,
            drift_shift_px=self.drift.shift_px,
            labels_hidden=self.drift.flagged and self._hide_labels_on_drift,
            attributions=self.tracks.attributions() + self.sky.attributions(),
            track_counts=self.tracks.store.counts(),
        )

        self.renderer.render(event.image, targets, event.wall_time, status,
                             effects=effects)

        if self.frame_hub is not None:
            self.frame_hub.publish(event.image, {
                "frame_time": event.wall_time.isoformat(),
                "visible": len(targets),
            })

        # Stop the clock before submitting: submit() blocks on the encoder's
        # bounded queue by design, so including it would just measure the
        # frame interval and hide whether rendering is actually keeping up.
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.output.submit(event.image)

        self.stats.frames += 1
        self.stats.last_frame_time = event.wall_time
        self.stats.render_ms_ewma = (0.98 * self.stats.render_ms_ewma
                                     + 0.02 * elapsed_ms)

    def _periodic(self, event: VideoFrameEvent) -> None:
        now = time.monotonic()
        if now - self._last_status_log < STATUS_INTERVAL_S:
            return
        self._last_status_log = now

        lag = (utcnow() - event.wall_time).total_seconds()
        payload = {
            "frames": self.stats.frames,
            "pipeline_fps": round(self.stats.fps, 2),
            "render_ms": round(self.stats.render_ms_ewma, 2),
            "lag_s": round(lag, 2),
            "visible": self.projector.last_stats.visible,
            "cull": self.projector.last_stats.as_dict(),
            "tracks": self.tracks.store.counts(),
            "output": self.output.status(),
            "drift": self.drift.status(),
            "decode_errors": self.stats.decode_errors,
            "sky": self.sky.status(),
        }
        if self._reader is not None:
            payload["ingest"] = self._reader.status()
        log.info("status", extra=payload)

        if self.output is not None and not self.output.alive:
            log.error("encoder process died; stopping so the supervisor restarts us")
            self.stop_event.set()

    def web_status(self) -> dict:
        """Snapshot for the monitor page. Safe to call from another thread."""
        last = self.stats.last_frame_time
        payload = {
            "uptime_s": round(self.stats.uptime_s, 1),
            "frames": self.stats.frames,
            "pipeline_fps": round(self.stats.fps, 2),
            "render_ms": round(self.stats.render_ms_ewma, 2),
            "decode_errors": self.stats.decode_errors,
            "frame_time": last.isoformat() if last else None,
            "lag_s": round((utcnow() - last).total_seconds(), 2) if last else None,
            "tracks": self.tracks.store.counts(),
            "attributions": self.tracks.attributions() + self.sky.attributions(),
            "groups": [g.status() for g in self.tracks.groups],
            "sky": self.sky.status(),
            "drift": self.drift.status(),
            "offline": self.offline,
            "waiting_for_calibration": self.waiting_for_calibration,
        }
        if self.projector is not None:
            payload["cull"] = self.projector.last_stats.as_dict()
            payload["visible"] = self.projector.last_stats.visible
        if self.output is not None:
            payload["output"] = self.output.status()
            payload["target_fps"] = self.output.fps
        if self._reader is not None:
            payload["ingest"] = self._reader.status()
        return payload

    # -- teardown -----------------------------------------------------------
    def shutdown(self) -> None:
        log.info("shutting down")
        self.stop_event.set()
        for name, closer in (("reader", self._reader), ("player", self._player),
                             ("tracks", self.tracks), ("sky", self.sky),
                             ("output", self.output)):
            if closer is None:
                continue
            try:
                closer.stop()
            except Exception:
                log.exception("error stopping component", extra={"component": name})
        log.info("stopped", extra={
            "frames": self.stats.frames,
            "uptime_s": round(self.stats.uptime_s, 1),
            "decode_errors": self.stats.decode_errors})


def install_signal_handlers(stop_event: threading.Event) -> None:
    """Translate SIGTERM/SIGINT into a clean stop, so Docker and systemd work."""
    def handler(signum, _frame):
        log.info("signal received", extra={"signal": signal.Signals(signum).name})
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError, AttributeError):
            pass  # not on the main thread, or not supported on this platform
