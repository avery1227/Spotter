"""Grab a single frame from the live stream (or an offline clip).

Used by ``calibrate.py`` to get a still to click on, and by the drift tooling to
refresh reference patches.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..logging_setup import get_logger
from .decoder import SegmentDecoder
from .hls import HLSReader
from .types import VideoFrameEvent

log = get_logger(__name__)


def grab_frame(cfg, timeout_s: float = 120.0,
               skip_frames: int = 0) -> Optional[VideoFrameEvent]:
    """Pull one frame from the configured live stream.

    ``skip_frames`` discards frames before returning one, which is useful when
    the very first decoded frame of a segment happens to catch a glitch.
    """
    reader = HLSReader(cfg, queue_size=2).start()
    decoder = SegmentDecoder(cfg, audio_sink=None)
    deadline = time.monotonic() + timeout_s
    skipped = 0
    try:
        for payload in reader.segments(timeout=2.0):
            if time.monotonic() > deadline:
                log.error("timed out waiting for a frame",
                          extra={"timeout_s": timeout_s})
                return None
            if payload is None:
                continue
            for event in decoder.decode(payload):
                if skipped < skip_frames:
                    skipped += 1
                    continue
                return event
    finally:
        reader.stop()
    return None


def grab_frame_from_file(path: str | Path,
                         at_seconds: float = 0.0) -> Optional[VideoFrameEvent]:
    """Grab a frame from a local video file, for offline calibration."""
    import av

    from ..util import utcnow
    from .types import PIXEL_FORMAT

    with av.open(str(path)) as container:
        if not container.streams.video:
            return None
        stream = container.streams.video[0]
        if at_seconds > 0 and stream.time_base:
            container.seek(int(at_seconds / float(stream.time_base)), stream=stream)
        for frame in container.decode(stream):
            image = frame.to_ndarray(format=PIXEL_FORMAT)
            if not image.flags.writeable or not image.flags.c_contiguous:
                image = np.ascontiguousarray(image)
            return VideoFrameEvent(image=image, wall_time=utcnow(), offset_s=0.0,
                                   sequence=-1, time_is_authoritative=False)
    return None


def save_frame(image: np.ndarray, path: str | Path) -> Path:
    """Write a BGRA frame to disk as PNG/JPEG."""
    import cv2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image[:, :, :3])
    return path


def load_frame(path: str | Path) -> np.ndarray:
    """Read an image from disk as a writable BGRA array."""
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2BGRA))
