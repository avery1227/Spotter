"""Minimal M3U8 parsing, focused on what we need for wall-clock timing.

We deliberately do not use a full HLS library: the only thing that matters here
is getting an accurate ``EXT-X-PROGRAM-DATE-TIME`` for every segment, and HLS's
PDT rules are short enough to implement exactly.

PDT semantics (RFC 8216 section 4.3.2.6): an ``EXT-X-PROGRAM-DATE-TIME`` tag
applies to the segment that follows it. Later segments without their own tag
inherit ``previous PDT + previous EXTINF duration``. YouTube emits exactly one
tag at the head of the window, so the accumulation path is the common one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional
from urllib.parse import urljoin

from ..logging_setup import get_logger

log = get_logger(__name__)

_ATTR_RE = re.compile(r'([A-Za-z0-9_-]+)=("[^"]*"|[^,]*)')
_QUALITY_RE = re.compile(r"(\d+)p")


@dataclass(frozen=True)
class Segment:
    """One media segment with the wall-clock time of its first sample."""

    uri: str
    sequence: int
    duration: float
    #: Program date-time of the segment's start, always timezone-aware UTC.
    pdt: Optional[datetime]
    discontinuity: bool = False
    #: True when ``pdt`` was accumulated from an earlier tag rather than read
    #: from a tag on this segment. Accumulated values drift if the publisher's
    #: EXTINF durations are approximate, so we surface the distinction.
    pdt_inferred: bool = False

    @property
    def end_pdt(self) -> Optional[datetime]:
        if self.pdt is None:
            return None
        return self.pdt + timedelta(seconds=self.duration)


@dataclass
class MediaPlaylist:
    """A parsed media playlist."""

    segments: list[Segment]
    target_duration: float
    media_sequence: int
    endlist: bool = False
    #: The URL the playlist was fetched from, used to resolve relative URIs.
    url: str = ""

    @property
    def live_edge(self) -> Optional[datetime]:
        """Wall-clock time of the newest sample the publisher has released."""
        for seg in reversed(self.segments):
            if seg.end_pdt is not None:
                return seg.end_pdt
        return None

    def window_duration(self) -> float:
        return sum(s.duration for s in self.segments)


def _parse_attrs(line: str) -> dict[str, str]:
    _, _, rest = line.partition(":")
    return {k: v.strip('"') for k, v in _ATTR_RE.findall(rest)}


def _parse_pdt(value: str) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp, normalising to timezone-aware UTC."""
    text = value.strip()
    # Python's fromisoformat handles "+00:00" but not a bare trailing "Z"
    # before 3.11, and never handles more than 6 fractional digits.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    text = re.sub(r"(\.\d{6})\d+", r"\1", text)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _looks_like_playlist(text: str) -> bool:
    """An M3U8 must open with #EXTM3U (RFC 8216 section 4.3.1.1)."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.startswith("#EXTM3U")
    return False


def is_master_playlist(text: str) -> bool:
    return "#EXT-X-STREAM-INF" in text


def select_variant(text: str, base_url: str, quality: str = "best") -> Optional[str]:
    """Pick a variant URL out of a master playlist.

    ``quality`` may be ``best``, ``worst``, or a height like ``720p``; a height
    selects the highest variant that does not exceed it.
    """
    variants: list[tuple[int, int, str]] = []  # (height, bandwidth, url)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("#EXT-X-STREAM-INF"):
            continue
        attrs = _parse_attrs(line)
        bandwidth = int(attrs.get("BANDWIDTH") or attrs.get("AVERAGE-BANDWIDTH") or 0)
        height = 0
        if "RESOLUTION" in attrs and "x" in attrs["RESOLUTION"]:
            try:
                height = int(attrs["RESOLUTION"].split("x")[1])
            except (ValueError, IndexError):
                height = 0
        for candidate in lines[i + 1:]:
            candidate = candidate.strip()
            if not candidate:
                continue
            if candidate.startswith("#"):
                break
            variants.append((height, bandwidth, urljoin(base_url, candidate)))
            break

    if not variants:
        return None

    variants.sort(key=lambda v: (v[0], v[1]))
    if quality == "worst":
        return variants[0][2]
    match = _QUALITY_RE.match(quality)
    if match:
        cap = int(match.group(1))
        capped = [v for v in variants if v[0] <= cap]
        if capped:
            return capped[-1][2]
    return variants[-1][2]


def parse_media_playlist(text: str, url: str = "") -> MediaPlaylist:
    """Parse a media playlist, assigning a PDT to every segment we can.

    A response that is not a playlist at all -- an HTML error page, a captive
    portal interstitial, a JSON error body -- yields an empty playlist rather
    than a list of nonsense "segments" the reader would then try to download.
    """
    if not _looks_like_playlist(text):
        preview = " ".join(text.split())[:120]
        log.warning("response is not an M3U8 playlist",
                    extra={"url": url, "preview": preview})
        return MediaPlaylist(segments=[], target_duration=6.0,
                             media_sequence=0, url=url)

    target_duration = 6.0
    media_sequence = 0
    endlist = False

    segments: list[Segment] = []
    pending_duration: Optional[float] = None
    pending_pdt: Optional[datetime] = None
    pending_discontinuity = False
    # Running clock used when a segment has no PDT tag of its own.
    next_pdt: Optional[datetime] = None
    sequence: Optional[int] = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith("#"):
            if line.startswith("#EXTINF"):
                _, _, rest = line.partition(":")
                try:
                    pending_duration = float(rest.split(",")[0])
                except (ValueError, IndexError):
                    pending_duration = target_duration
            elif line.startswith("#EXT-X-PROGRAM-DATE-TIME"):
                _, _, rest = line.partition(":")
                pending_pdt = _parse_pdt(rest)
            elif line.startswith("#EXT-X-TARGETDURATION"):
                _, _, rest = line.partition(":")
                try:
                    target_duration = float(rest)
                except ValueError:
                    pass
            elif line.startswith("#EXT-X-MEDIA-SEQUENCE"):
                _, _, rest = line.partition(":")
                try:
                    media_sequence = int(rest)
                except ValueError:
                    pass
            elif line.startswith("#EXT-X-DISCONTINUITY") and \
                    not line.startswith("#EXT-X-DISCONTINUITY-SEQUENCE"):
                pending_discontinuity = True
                # A discontinuity invalidates the accumulated clock; we must
                # wait for a fresh PDT tag rather than keep counting.
                next_pdt = None
            elif line.startswith("#EXT-X-ENDLIST"):
                endlist = True
            continue

        # A non-comment line is a segment URI.
        if sequence is None:
            sequence = media_sequence
        duration = pending_duration if pending_duration is not None else target_duration

        inferred = False
        if pending_pdt is not None:
            pdt = pending_pdt
        elif next_pdt is not None:
            pdt = next_pdt
            inferred = True
        else:
            pdt = None

        segments.append(Segment(
            uri=urljoin(url, line) if url else line,
            sequence=sequence,
            duration=duration,
            pdt=pdt,
            discontinuity=pending_discontinuity,
            pdt_inferred=inferred,
        ))

        next_pdt = pdt + timedelta(seconds=duration) if pdt is not None else None
        sequence += 1
        pending_duration = None
        pending_pdt = None
        pending_discontinuity = False

    return MediaPlaylist(
        segments=segments,
        target_duration=target_duration,
        media_sequence=media_sequence,
        endlist=endlist,
        url=url,
    )


def newer_than(segments: Iterable[Segment], last_sequence: Optional[int]) -> list[Segment]:
    """Segments strictly newer than ``last_sequence`` (all of them if it is None)."""
    if last_sequence is None:
        return list(segments)
    return [s for s in segments if s.sequence > last_sequence]
