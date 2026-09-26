"""Source stalls: the playlist keeps answering but no new segments arrive.

Seen in production as a ~30s freeze with nothing in the log, no RECONNECTING
badge on the output, and the wall display repeatedly reconnecting because the
preview stopped advancing.
"""

import threading
import time

import numpy as np

from spotter.config import Config
from spotter.ingest.hls import HLSReader
from spotter.ingest.playlist import MediaPlaylist
from spotter.ingest.resolver import ResolvedStream
from spotter.pipeline import Pipeline
from spotter.render.overlay import OverlayRenderer, OverlayStatus
from spotter.web.frames import FrameHub


def make_reader(tmp_path) -> HLSReader:
    cfg = Config({"stream": {"url": "https://example.invalid/watch",
                             "stall_warn_s": 1.0, "stall_reresolve_s": 3.0}},
                 root=tmp_path)
    return HLSReader(cfg, threading.Event())


def empty_playlist() -> MediaPlaylist:
    return MediaPlaylist(segments=[], target_duration=2.0, media_sequence=100)


def test_stall_is_reported_then_triggers_a_reresolve(tmp_path):
    reader = make_reader(tmp_path)
    now = time.monotonic()
    reader._resolved = ResolvedStream("https://x/p.m3u8", resolved_at=now - 600)

    reader.last_segment_at = now - 0.5
    assert reader._check_stall(empty_playlist()) is False
    assert reader.status()["stalled"] is False

    reader.last_segment_at = now - 2.0
    assert reader._check_stall(empty_playlist()) is False
    assert reader.status()["stalled"] is True
    assert reader.status()["stalls"] == 1

    reader.last_segment_at = now - 4.0
    assert reader._check_stall(empty_playlist()) is True
    assert reader._resolved is None
    assert reader.status()["stalls"] == 1, "one stall, not one per poll"


def test_a_fresh_resolve_is_not_immediately_repeated(tmp_path):
    reader = make_reader(tmp_path)
    reader._resolved = ResolvedStream("https://x/p.m3u8")   # resolved just now
    reader.last_segment_at = time.monotonic() - 60
    assert reader._check_stall(empty_playlist()) is False
    assert reader._resolved is not None


class FakeOutput:
    def __init__(self, frame):
        self._held = frame
        self.starved = True

    def should_show_badge(self):
        return self.starved

    def starved_for_s(self):
        return 7.0

    def held_frame(self):
        return self._held

    def hold_frame(self, frame):
        self._held = frame


def make_stalled_pipeline(tmp_path):
    cfg = Config({
        "camera": {"lat": 41.27, "lon": -72.5, "height_m": 15.0},
        "tracks": {"adsb": {"enabled": False}, "ais": {"enabled": False}},
        "drift": {"enabled": False},
    }, root=tmp_path)
    hub = FrameHub()
    pipeline = Pipeline(cfg, stop_event=threading.Event(), frame_hub=hub)
    pipeline.renderer = OverlayRenderer(cfg)
    clean = np.zeros((360, 640, 4), dtype=np.uint8)
    clean[..., 3] = 255
    pipeline.output = FakeOutput(clean)
    return pipeline, hub, clean


def test_idle_ticks_redraw_the_badge_and_keep_the_preview_moving(tmp_path):
    pipeline, hub, clean = make_stalled_pipeline(tmp_path)
    original = clean.copy()

    pipeline._on_input_idle()
    first = hub.sequence
    held = pipeline.output.held_frame()
    assert held is not clean
    assert (held[..., :3] != 0).any(), "badge was not drawn"
    assert np.array_equal(clean, original), "must not draw on the original"

    pipeline._on_input_idle()
    assert hub.sequence > first, "preview must keep advancing during a stall"

    # Each redraw starts from the clean frame, so badges never stack up.
    assert np.array_equal(pipeline._stall_base, original)


def test_no_redraw_before_the_badge_is_due(tmp_path):
    pipeline, hub, clean = make_stalled_pipeline(tmp_path)
    pipeline.output.starved = False
    before = hub.sequence
    pipeline._on_input_idle()
    assert hub.sequence == before
    assert pipeline.output.held_frame() is clean


def test_render_badges_draws_only_the_badge(tmp_path):
    renderer = OverlayRenderer(Config({}, root=tmp_path))
    frame = np.zeros((360, 640, 4), dtype=np.uint8)
    renderer.render_badges(frame, OverlayStatus())
    assert not frame.any(), "no status, nothing to draw"
    renderer.render_badges(frame, OverlayStatus(reconnecting=True,
                                                reconnecting_since_s=4))
    ys, xs = np.nonzero(frame[..., :3].any(axis=-1))
    assert xs.min() > 640 / 2 and ys.max() < 360 / 2, "badge sits top-right"
