"""Decode HLS segments into wall-clock-stamped frames.

Each segment is decoded on its own, in memory. That is what makes the timing
exact: a frame's capture time is

    segment PDT + (frame PTS - segment's earliest PTS) - encoder_delay_s

The subtraction of the segment's *earliest* PTS matters. With B-frames the first
frame out of the decoder is not necessarily the earliest in presentation order,
so we make a cheap demux-only pass over the (already in-memory) segment first to
find the true minimum. That pass costs microseconds and removes a
reorder-delay-sized bias from every timestamp.
"""

from __future__ import annotations

import io
from datetime import timedelta
from typing import Callable, Iterator, Optional

import av
import numpy as np

from ..logging_setup import get_logger
from ..util import utcnow
from .audio import ADTSRemuxer
from .hls import SegmentPayload
from .types import PIXEL_FORMAT, StreamInfo, VideoFrameEvent

log = get_logger(__name__)

#: Receives ready-to-write ADTS bytes. Muxing happens here, in the decode
#: thread, because it needs the source container's audio stream as a template
#: and that is only valid while the segment is open.
AudioSink = Callable[[bytes], None]


class SegmentDecodeError(Exception):
    pass


class SegmentDecoder:
    """Turns :class:`SegmentPayload` objects into :class:`VideoFrameEvent`s."""

    def __init__(self, cfg, audio_sink: Optional[AudioSink] = None):
        self.cfg = cfg
        self.audio_sink = audio_sink
        self.encoder_delay_s = float(cfg.get("stream.encoder_delay_s", 0.0))
        self.pdt_enabled = bool(cfg.get("stream.pdt.enabled", True))
        self.max_skew_s = float(cfg.get("stream.pdt.max_skew_s", 900.0))
        self.assumed_latency_s = float(cfg.get("stream.pdt.assumed_latency_s", 30.0))
        self.pass_audio = str(cfg.get("video.audio", "passthrough")) == "passthrough"

        self.info = StreamInfo()
        self._warned_pdt = False
        self._pending_discontinuity = True
        self._remuxer = ADTSRemuxer(audio_sink) if audio_sink is not None else None

    # -- timing -------------------------------------------------------------
    def _segment_start(self, payload: SegmentPayload):
        """Wall-clock time of the segment's first sample, and whether it is real."""
        pdt = payload.pdt
        if self.pdt_enabled and pdt is not None:
            skew = abs((utcnow() - pdt).total_seconds())
            if skew <= self.max_skew_s:
                return pdt, True
            if not self._warned_pdt:
                self._warned_pdt = True
                log.warning(
                    "PROGRAM-DATE-TIME is implausible; falling back to local clock",
                    extra={"pdt": pdt.isoformat(), "skew_s": round(skew, 1),
                           "max_skew_s": self.max_skew_s})
        elif pdt is None and not self._warned_pdt:
            self._warned_pdt = True
            log.warning("playlist has no PROGRAM-DATE-TIME; using local clock",
                        extra={"assumed_latency_s": self.assumed_latency_s})

        # Fallback: assume the segment's start is one assumed-latency behind now,
        # minus the segment duration we are about to play out.
        return (utcnow()
                - timedelta(seconds=self.assumed_latency_s)
                - timedelta(seconds=payload.segment.duration)), False

    @staticmethod
    def _min_video_pts(data: bytes) -> Optional[int]:
        """Cheapest possible pass: demux only, find the smallest video PTS."""
        try:
            with av.open(io.BytesIO(data), mode="r") as container:
                if not container.streams.video:
                    return None
                stream = container.streams.video[0]
                best: Optional[int] = None
                for packet in container.demux(stream):
                    if packet.pts is None:
                        continue
                    if best is None or packet.pts < best:
                        best = packet.pts
                return best
        except av.FFmpegError:
            return None

    # -- decode -------------------------------------------------------------
    def decode(self, payload: SegmentPayload) -> Iterator[VideoFrameEvent]:
        """Decode one segment, yielding frames in presentation order."""
        segment_start, authoritative = self._segment_start(payload)
        base_pts = self._min_video_pts(payload.data)

        discontinuity = self._pending_discontinuity or payload.gap
        self._pending_discontinuity = False

        try:
            container = av.open(io.BytesIO(payload.data), mode="r")
        except av.FFmpegError as exc:
            self._pending_discontinuity = True
            raise SegmentDecodeError(
                f"could not open segment {payload.segment.sequence}: {exc}") from exc

        with container:
            if not container.streams.video:
                raise SegmentDecodeError(
                    f"segment {payload.segment.sequence} has no video stream")

            video = container.streams.video[0]
            # Let PyAV spread decoding across cores; at 1080p30 this is the
            # difference between keeping up and not on a modest CPU.
            video.thread_type = "AUTO"
            audio = container.streams.audio[0] if container.streams.audio else None
            self._update_info(video, audio)

            time_base = float(video.time_base) if video.time_base else 1.0 / 90000.0
            if base_pts is None:
                base_pts = video.start_time if video.start_time is not None else 0

            streams = [video] + ([audio] if audio is not None and self.pass_audio else [])

            for packet in container.demux(*streams):
                if audio is not None and packet.stream is audio:
                    # The trailing empty packet only exists to flush a decoder;
                    # there is nothing to pass through for a copied stream.
                    if packet.pts is None and packet.dts is None:
                        continue
                    if self._remuxer is not None and not self._remuxer.failed:
                        self._remuxer.feed(packet, audio)
                    continue

                # Do NOT skip the empty trailing packet here: decoding it is what
                # drains the frame-threaded decoder's reorder queue. Skipping it
                # silently loses one thread-depth of frames off the end of every
                # segment, which shows up as a stutter at each segment boundary.
                for frame in packet.decode():
                    pts = frame.pts if frame.pts is not None else base_pts
                    offset_s = max(0.0, (pts - base_pts) * time_base)
                    wall = (segment_start
                            + timedelta(seconds=offset_s)
                            - timedelta(seconds=self.encoder_delay_s))

                    image = frame.to_ndarray(format=PIXEL_FORMAT)
                    # to_ndarray may hand back a read-only view; the renderer
                    # draws into this buffer in place, so guarantee it is ours.
                    if not image.flags.writeable or not image.flags.c_contiguous:
                        image = np.ascontiguousarray(image)

                    yield VideoFrameEvent(
                        image=image,
                        wall_time=wall,
                        offset_s=offset_s,
                        sequence=payload.segment.sequence,
                        time_is_authoritative=authoritative,
                        discontinuity=discontinuity,
                    )
                    discontinuity = False

    def _update_info(self, video, audio) -> None:
        if not self.info.is_ready():
            fps = 0.0
            for candidate in (video.average_rate, video.guessed_rate, video.base_rate):
                if candidate:
                    fps = float(candidate)
                    break
            self.info = StreamInfo(
                width=int(video.codec_context.width),
                height=int(video.codec_context.height),
                fps=fps or 30.0,
                codec=video.codec_context.name,
                has_audio=audio is not None,
                audio_codec=audio.codec_context.name if audio is not None else "",
                audio_rate=int(audio.codec_context.sample_rate) if audio is not None else 0,
                audio_channels=(getattr(audio.codec_context.layout, "nb_channels", 0)
                                if audio is not None else 0),
            )
            log.info("stream parameters", extra={
                "width": self.info.width, "height": self.info.height,
                "fps": round(self.info.fps, 3), "codec": self.info.codec,
                "audio": self.info.audio_codec or None,
            })

    @property
    def audio_packets(self) -> int:
        """AAC frames remuxed so far. Used to measure the priming lead."""
        return self._remuxer.packets if self._remuxer is not None else 0

    def set_audio_sink(self, sink: AudioSink) -> None:
        """Redirect remuxed audio bytes, e.g. from the prime buffer to the pipe."""
        if self._remuxer is not None:
            self._remuxer.set_sink(sink)

    def close(self) -> None:
        if self._remuxer is not None:
            self._remuxer.close()

    def note_discontinuity(self) -> None:
        """Tell the decoder the next segment follows a break in the stream."""
        self._pending_discontinuity = True
