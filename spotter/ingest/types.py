"""Types shared between ingest and the rest of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

#: Pixel format carried end to end. BGRA keeps Skia's kBGRA_8888 raster surface
#: able to draw straight into the decoded buffer with no conversion, and ffmpeg
#: accepts it verbatim as rawvideo input.
PIXEL_FORMAT = "bgra"
BYTES_PER_PIXEL = 4

@dataclass
class VideoFrameEvent:
    """One decoded frame, tagged with the wall-clock time it was captured."""

    #: (H, W, 4) uint8 BGRA. Writable: the renderer draws into it in place.
    image: np.ndarray
    #: Capture time, already corrected by ``stream.encoder_delay_s``.
    wall_time: datetime
    #: Presentation time within the segment, seconds from the segment's start.
    offset_s: float
    #: Media sequence number of the segment this frame came from.
    sequence: int
    #: False when the timestamp was derived from local clock minus an assumed
    #: latency rather than from a real PROGRAM-DATE-TIME tag.
    time_is_authoritative: bool = True
    #: True for the first frame after a gap or reconnect.
    discontinuity: bool = False

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

@dataclass
class StreamInfo:
    """Video parameters discovered from the first decoded segment."""

    width: int = 0
    height: int = 0
    fps: float = 0.0
    codec: str = ""
    has_audio: bool = False
    audio_codec: str = ""
    audio_rate: int = 0
    audio_channels: int = 0
    extra: dict = field(default_factory=dict)

    def is_ready(self) -> bool:
        return self.width > 0 and self.height > 0 and self.fps > 0
