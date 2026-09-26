"""Live HLS segment reader.

We fetch segments ourselves rather than handing the playlist URL to libav. That
costs a little code but buys the thing this project is built around: we know
exactly which segment every frame came from, so a frame's wall-clock time is
``segment PDT + offset within segment`` rather than a guess anchored to however
many segments libav happened to buffer.

The reader runs on its own thread, keeps a small read-ahead queue, re-resolves
the signed playlist URL before it expires, and reconnects with backoff.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from queue import Empty, Full, Queue
from typing import Iterator, Optional

import requests

from ..logging_setup import get_logger
from ..util import Backoff
from .playlist import MediaPlaylist, Segment, parse_media_playlist
from .resolver import ResolvedStream, ResolveError, resolve_stream

log = get_logger(__name__)

#: Re-resolve this long before the signed URL's stated expiry.
EXPIRY_MARGIN_S = 600.0


@dataclass
class SegmentPayload:
    """A downloaded segment plus the metadata needed to time its frames."""

    data: bytes
    segment: Segment
    #: True when the reader had to skip forward because we fell out of the
    #: publisher's DVR window; downstream should reset PTS continuity.
    gap: bool = False

    @property
    def pdt(self) -> Optional[datetime]:
        return self.segment.pdt


class StreamGone(Exception):
    """The reader gave up: the broadcast ended or is permanently unreachable."""


class HLSReader:
    """Background thread that turns a watch URL into a stream of segment bytes."""

    def __init__(self, cfg, stop_event: Optional[threading.Event] = None,
                 queue_size: int = 4):
        self.cfg = cfg
        self.stop_event = stop_event or threading.Event()
        self.queue: "Queue[SegmentPayload]" = Queue(maxsize=queue_size)

        self.url = cfg.require("stream.url")
        self.resolver = cfg.get("stream.resolver", "streamlink")
        self.quality = cfg.get("stream.quality", "best")
        self.resolver_timeout_s = float(cfg.get("stream.resolver_timeout_s", 60.0))
        self.reresolve_interval_s = float(cfg.get("stream.reresolve_interval_s", 10800.0))
        self.read_timeout_s = float(cfg.get("stream.read_timeout_s", 20.0))
        self.playlist_poll_s = float(cfg.get("stream.pdt.playlist_poll_s", 4.0))
        # The playlist can keep answering 200 while the publisher stops adding
        # segments. Say so after stall_warn_s, and fetch a fresh URL after
        # stall_reresolve_s in case this one has quietly gone stale.
        self.stall_warn_s = float(cfg.get("stream.stall_warn_s", 10.0))
        self.stall_reresolve_s = float(cfg.get("stream.stall_reresolve_s", 30.0))

        self._backoff = Backoff.from_config(cfg.sub("stream.backoff"))
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
        self._resolved: Optional[ResolvedStream] = None
        self._last_sequence: Optional[int] = None
        self._thread: Optional[threading.Thread] = None

        # Observability for the status badge / logs.
        self.last_segment_at: float = 0.0
        self.connected: bool = False
        self.reconnects: int = 0
        self.stalls: int = 0
        self._stalled_since: Optional[float] = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "HLSReader":
        self._thread = threading.Thread(target=self._run, name="hls-reader", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        self._session.close()

    def segments(self, timeout: float = 1.0) -> Iterator[Optional[SegmentPayload]]:
        """Yield downloaded segments; yields ``None`` when idle so callers can tick."""
        while not self.stop_event.is_set():
            try:
                yield self.queue.get(timeout=timeout)
            except Empty:
                yield None

    # -- internals ----------------------------------------------------------
    def _needs_resolve(self) -> bool:
        if self._resolved is None:
            return True
        if self._resolved.age_s() > self.reresolve_interval_s:
            return True
        remaining = self._resolved.seconds_until_expiry()
        return remaining is not None and remaining < EXPIRY_MARGIN_S

    def _resolve(self) -> None:
        self._resolved = resolve_stream(
            self.url,
            resolver=self.resolver,
            quality=self.quality,
            timeout_s=self.resolver_timeout_s,
        )
        # A fresh playlist URL means fresh sequence numbering may not line up
        # with what we saw before; re-anchor at the live edge.
        self._last_sequence = None

    def _fetch_playlist(self) -> MediaPlaylist:
        assert self._resolved is not None
        resp = self._session.get(self._resolved.playlist_url, timeout=self.read_timeout_s)
        resp.raise_for_status()
        playlist = parse_media_playlist(resp.text, url=resp.url)
        if not playlist.segments:
            raise StreamGone("playlist contained no segments")
        return playlist

    def _download(self, segment: Segment) -> bytes:
        resp = self._session.get(segment.uri, timeout=self.read_timeout_s)
        resp.raise_for_status()
        return resp.content

    def _pick_new_segments(self, playlist: MediaPlaylist) -> tuple[list[Segment], bool]:
        """Choose which segments to download next, and whether we skipped a gap."""
        segments = playlist.segments
        if self._last_sequence is None:
            # Cold start: begin one segment back from the live edge. Starting on
            # the very last segment risks racing a partially-published file;
            # one back is still under ~10s of latency at a 5s target duration.
            start_index = max(0, len(segments) - 2)
            chosen = segments[start_index:]
            log.info("anchoring at live edge", extra={
                "sequence": chosen[0].sequence,
                "window_s": playlist.window_duration(),
                "pdt": chosen[0].pdt.isoformat() if chosen[0].pdt else None,
            })
            return chosen, False

        expected = self._last_sequence + 1
        fresh = [s for s in segments if s.sequence >= expected]
        gap = bool(fresh) and fresh[0].sequence > expected
        if gap:
            log.warning("fell behind the DVR window; skipping ahead", extra={
                "expected": expected,
                "got": fresh[0].sequence,
                "skipped": fresh[0].sequence - expected,
            })
        return fresh, gap

    def _run(self) -> None:
        log.info("hls reader starting", extra={"url": self.url})
        while not self.stop_event.is_set():
            try:
                if self._needs_resolve():
                    self._resolve()
                self._pump()
            except StreamGone as exc:
                log.error("stream gone; will retry", extra={"error": str(exc)})
                self._resolved = None
            except ResolveError as exc:
                log.error("resolve failed", extra={"error": str(exc)})
                self._resolved = None
            except requests.RequestException as exc:
                log.warning("network error", extra={"error": str(exc)})
                # A 403 on the playlist means the signature expired early.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (403, 404, 410):
                    self._resolved = None
            except Exception:  # pragma: no cover - defensive
                log.exception("unexpected reader error")
                self._resolved = None

            if self.stop_event.is_set():
                break
            self.connected = False
            self.reconnects += 1
            delay = self._backoff.sleep(self.stop_event)
            log.info("reconnecting", extra={"after_s": round(delay, 2),
                                            "attempt": self._backoff.attempt})
        log.info("hls reader stopped")

    def _pump(self) -> None:
        """Poll the playlist and download segments until something goes wrong."""
        next_poll = 0.0
        while not self.stop_event.is_set():
            if self._needs_resolve():
                return  # bounce back out to _run so it re-resolves cleanly

            now = time.monotonic()
            if now < next_poll:
                self.stop_event.wait(min(0.25, next_poll - now))
                continue

            playlist = self._fetch_playlist()
            self.connected = True
            self._backoff.reset()

            fresh, gap = self._pick_new_segments(playlist)
            if not fresh:
                if self._check_stall(playlist):
                    return  # bounce back out to _run, which re-resolves
                # Nothing new yet; poll again at roughly half the target duration.
                next_poll = time.monotonic() + min(self.playlist_poll_s,
                                                   playlist.target_duration / 2)
                continue

            for index, segment in enumerate(fresh):
                if self.stop_event.is_set():
                    return
                try:
                    data = self._download(segment)
                except requests.RequestException as exc:
                    # One bad segment should not tear down the connection; skip
                    # it and let PTS continuity handling deal with the hole.
                    log.warning("segment download failed", extra={
                        "sequence": segment.sequence, "error": str(exc)})
                    self._last_sequence = segment.sequence
                    continue

                payload = SegmentPayload(data=data, segment=segment,
                                         gap=gap and index == 0)
                self._put(payload)
                self._last_sequence = segment.sequence
                self.last_segment_at = time.monotonic()
                if self._stalled_since is not None:
                    log.info("segments resumed", extra={
                        "stalled_for_s": round(self.last_segment_at
                                               - self._stalled_since, 1),
                        "sequence": segment.sequence})
                    self._stalled_since = None

            next_poll = time.monotonic() + min(self.playlist_poll_s,
                                               playlist.target_duration / 2)

    def _check_stall(self, playlist: MediaPlaylist) -> bool:
        """Log a publisher stall; return True when it is time to re-resolve."""
        if not self.last_segment_at:
            return False
        now = time.monotonic()
        idle = now - self.last_segment_at
        if idle < self.stall_warn_s:
            return False

        if self._stalled_since is None:
            self._stalled_since = self.last_segment_at
            self.stalls += 1
            newest = playlist.segments[-1].sequence if playlist.segments else None
            log.warning("no new segments from the source", extra={
                "idle_s": round(idle, 1),
                "last_sequence": self._last_sequence,
                "playlist_newest": newest,
                "target_duration_s": playlist.target_duration,
                "endlist": playlist.endlist,
            })

        # Measure from whichever is later, the last segment or the last
        # resolve, so a stall that outlives one re-resolve does not trigger
        # another on every poll.
        since_resolve = self._resolved.age_s() if self._resolved else idle
        if min(idle, since_resolve) >= self.stall_reresolve_s:
            log.warning("source still stalled; re-resolving the stream URL",
                        extra={"idle_s": round(idle, 1)})
            self._resolved = None
            return True
        return False

    def _put(self, payload: SegmentPayload) -> None:
        """Enqueue a segment, dropping the oldest if the consumer is starved.

        Dropping beats blocking: if rendering has fallen behind, the useful thing
        is to stay near the live edge rather than accumulate an ever-growing lag.
        """
        while not self.stop_event.is_set():
            try:
                self.queue.put(payload, timeout=0.5)
                return
            except Full:
                try:
                    dropped = self.queue.get_nowait()
                    log.warning("consumer behind; dropping segment", extra={
                        "sequence": dropped.segment.sequence})
                except Empty:
                    pass

    def status(self) -> dict:
        age = (time.monotonic() - self.last_segment_at) if self.last_segment_at else None
        return {
            "connected": self.connected,
            "reconnects": self.reconnects,
            "queue": self.queue.qsize(),
            "last_segment_age_s": round(age, 2) if age is not None else None,
            "stalled": self._stalled_since is not None,
            "stalls": self.stalls,
            "playlist_expires_in_s": (
                round(self._resolved.seconds_until_expiry())
                if self._resolved and self._resolved.expires_at else None),
        }
