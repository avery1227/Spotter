"""Remux demuxed AAC packets into a raw ADTS byte stream.

libav strips the ADTS header when demuxing AAC out of MPEG-TS, so the packets
cannot simply be concatenated -- they have to be re-wrapped before ffmpeg can
read them back as ``-f aac``.

Two details matter, and getting either wrong produces a stream that *looks*
plausible and decodes to noise:

* The output stream must be created with
  ``add_stream_from_template(source_stream)``. Creating it by name
  (``add_stream("aac", rate=...)``) leaves the codec parameters at their
  defaults, and the ADTS headers then encode the wrong sample-rate index and
  channel configuration. ffmpeg reports ``sample rate not set`` and
  ``channel element 2.8 is not allocated``.

* Because of that, the remuxer must be created while the *source* container is
  still open. It is therefore owned by the decoder, not by the output stage,
  and the bytes it produces are handed onwards through a sink callback.
"""

from __future__ import annotations

from typing import Callable, Optional

import av

from ..logging_setup import get_logger

log = get_logger(__name__)

ByteSink = Callable[[bytes], None]


class _SinkFile:
    """Minimal writable file-like object that forwards to a callback."""

    def __init__(self, sink: ByteSink):
        self._sink = sink
        self.closed = False

    def write(self, data) -> int:
        payload = bytes(data)
        if payload:
            self._sink(payload)
        return len(payload)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    # PyAV probes for these; ADTS is a streaming format and never seeks.
    def seekable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False


class ADTSRemuxer:
    """Wraps AAC packets back into ADTS frames, emitting bytes to a sink."""

    def __init__(self, sink: ByteSink):
        self.sink = sink
        self._container = None
        self._stream = None
        self._handle: Optional[_SinkFile] = None
        self.failed = False
        self.packets = 0

    @property
    def ready(self) -> bool:
        return self._container is not None and not self.failed

    def feed(self, packet, template) -> bool:
        """Remux one packet. ``template`` is the source container's audio stream."""
        if self.failed:
            return False
        if self._container is None and not self._open(template):
            return False
        try:
            packet.stream = self._stream
            self._container.mux(packet)
            self.packets += 1
            return True
        except Exception as exc:
            log.warning("ADTS remux failed; disabling audio passthrough",
                        extra={"error": f"{type(exc).__name__}: {exc}"})
            self.failed = True
            return False

    def _open(self, template) -> bool:
        try:
            handle = _SinkFile(self.sink)
            container = av.open(handle, mode="w", format="adts")
            # Copies the source's codec parameters, which is what makes the
            # generated ADTS headers describe the actual audio.
            stream = container.add_stream_from_template(template)
            self._container, self._stream, self._handle = container, stream, handle
            log.info("ADTS remuxer ready", extra={
                "codec": template.codec_context.name,
                "rate": template.codec_context.sample_rate,
                "profile": template.codec_context.profile})
            return True
        except Exception as exc:
            log.warning("could not open ADTS remuxer; audio will be dropped",
                        extra={"error": f"{type(exc).__name__}: {exc}"})
            self.failed = True
            return False

    def set_sink(self, sink: ByteSink) -> None:
        self.sink = sink
        # The file-like object holds its own reference to the old sink, so it
        # has to be rebound too or output would keep going to the buffer.
        if self._handle is not None:
            self._handle._sink = sink

    def close(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
            self._stream = None
            self._handle = None
