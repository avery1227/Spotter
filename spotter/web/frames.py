"""Shares the pipeline's most recent composited frame with the web UI.

The pipeline publishes every frame it renders; the web server encodes one to
JPEG only when a browser actually asks for it. That asymmetry matters: encoding
1080p JPEG at 30 fps would cost more CPU than the H.264 encode does, for a
preview nobody may be watching.

The other rule here is that **watching must never slow down streaming**. The
frame reference is swapped without holding a lock -- an attribute assignment is
atomic under the GIL -- and the condition variable is used only to wake MJPEG
readers. Resizing and encoding happen entirely outside any lock, so a viewer
requesting a 1080p JPEG cannot stall the render loop behind an 8 MB memcpy.

The cost of that choice is that an encode can race the decoder overwriting the
buffer, so a preview frame may occasionally tear. For a monitoring view that is
a much better trade than dropping real output frames.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from ..logging_setup import get_logger

log = get_logger(__name__)


class FrameHub:
    """Holds the latest frame for the web preview, with lazy JPEG encoding."""

    def __init__(self, max_width: int = 960, quality: int = 70):
        self.max_width = int(max_width)
        self.quality = int(quality)

        # Only guards the wait/notify handshake, never the pixel work.
        self._waiters = threading.Condition()

        self._frame: Optional[np.ndarray] = None
        self._meta: dict = {}
        self._sequence = 0
        self._published_at = 0.0

        self._cache_lock = threading.Lock()
        self._cache_sequence = -1
        self._cache_key: tuple = ()
        self._cache: Optional[bytes] = None

    # -- producer -----------------------------------------------------------
    def publish(self, image: np.ndarray, meta: Optional[dict] = None) -> None:
        """Called from the render loop for every composited frame.

        Deliberately cheap: three attribute assignments and a notify. No lock
        is taken around the frame itself, so the render loop can never be made
        to wait on a viewer's JPEG encode.
        """
        self._frame = image
        self._meta = meta or {}
        self._sequence += 1
        self._published_at = time.monotonic()
        with self._waiters:
            self._waiters.notify_all()

    # -- consumers ----------------------------------------------------------
    @property
    def sequence(self) -> int:
        return self._sequence

    def age_s(self) -> Optional[float]:
        published = self._published_at
        return (time.monotonic() - published) if published else None

    def wait_for_frame(self, after: int, timeout: float = 5.0) -> int:
        """Block until a frame newer than ``after`` arrives. Returns the sequence."""
        deadline = time.monotonic() + timeout
        with self._waiters:
            while self._sequence <= after:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return self._sequence
                self._waiters.wait(remaining)
            return self._sequence

    def jpeg(self, max_width: Optional[int] = None,
             quality: Optional[int] = None) -> Optional[tuple[bytes, int]]:
        """Encode the latest frame. Returns ``(bytes, sequence)`` or None."""
        import cv2

        width = int(max_width or self.max_width)
        qual = int(quality or self.quality)
        key = (width, qual)

        # Several viewers at the same size share one encode.
        with self._cache_lock:
            if self._sequence == self._cache_sequence and key == self._cache_key:
                return self._cache, self._cache_sequence

        frame = self._frame            # atomic read of the current reference
        sequence = self._sequence
        if frame is None:
            return None

        if frame.shape[1] > width:
            scale = width / float(frame.shape[1])
            small = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame.copy()

        ok, buffer = cv2.imencode(".jpg", small[:, :, :3],
                                  [int(cv2.IMWRITE_JPEG_QUALITY), qual])
        if not ok:
            return None
        data = buffer.tobytes()

        with self._cache_lock:
            self._cache_sequence = sequence
            self._cache_key = key
            self._cache = data
        return data, sequence

    def snapshot_meta(self) -> dict:
        return dict(self._meta)
