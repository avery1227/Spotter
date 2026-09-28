"""Resolve a YouTube watch URL into a playable HLS media-playlist URL.

Two backends are supported. ``streamlink`` handles YouTube's quirks best and is
the default; ``yt_dlp`` is the fallback for when a YouTube change breaks one but
not the other. Both return the URL of a *media* playlist (the one containing
``#EXTINF`` segments); if a backend hands back a master playlist we pick the
variant ourselves.

Signed googlevideo URLs carry an ``expire`` parameter, so the caller is expected
to re-resolve periodically (``stream.reresolve_interval_s``) and on failure.
"""

from __future__ import annotations

import os
import re
import time
from http.cookiejar import LoadError, MozillaCookieJar
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import parse_qs, urlparse

import requests

from ..compat import ensure_urllib3_percent_re
from ..logging_setup import get_logger

log = get_logger(__name__)

_QUALITY_RANK = re.compile(r"(\d+)p")

#: Picked up automatically when ``stream.cookies_file`` is unset, so a panel
#: user can fix bot-checks by dropping a file into the server directory.
DEFAULT_COOKIES_FILE = "cookies.txt"

_BOT_CHECK = re.compile(r"not a bot|LOGIN_REQUIRED|Sign in to confirm", re.I)


class ResolveError(Exception):
    """The watch URL could not be turned into a playable stream."""


@dataclass
class ResolvedStream:
    """A resolved HLS media playlist, plus enough metadata to know when it ages out."""

    playlist_url: str
    resolved_at: float = field(default_factory=time.monotonic)
    quality: str = "best"
    backend: str = "streamlink"
    #: Unix epoch at which the signed URL stops working, if we could read it.
    expires_at: Optional[float] = None

    def age_s(self) -> float:
        return time.monotonic() - self.resolved_at

    def seconds_until_expiry(self) -> Optional[float]:
        if self.expires_at is None:
            return None
        return self.expires_at - time.time()


def _expiry_from_url(url: str) -> Optional[float]:
    """googlevideo URLs encode expiry either as /expire/<epoch>/ or ?expire=<epoch>."""
    match = re.search(r"/expire/(\d{9,12})", url)
    if match:
        return float(match.group(1))
    query = parse_qs(urlparse(url).query)
    if "expire" in query:
        try:
            return float(query["expire"][0])
        except (TypeError, ValueError):
            return None
    return None


def find_cookies_file(configured: Optional[str]) -> Optional[str]:
    """Return the Netscape cookies file to send to YouTube, or None.

    An explicitly configured path must exist; an unset one falls back to
    ``cookies.txt`` in the working directory if there is one.
    """
    if configured:
        path = os.path.expanduser(configured)
        if not os.path.isfile(path):
            raise ResolveError(f"stream.cookies_file {configured!r} does not exist")
        return path
    return DEFAULT_COOKIES_FILE if os.path.isfile(DEFAULT_COOKIES_FILE) else None


def _load_cookie_jar(path: str) -> MozillaCookieJar:
    jar = MozillaCookieJar(path)
    try:
        jar.load(ignore_discard=True, ignore_expires=True)
    except (LoadError, OSError) as exc:
        raise ResolveError(
            f"could not read cookies file {path!r} (needs Netscape/cookies.txt format): {exc}"
        ) from exc
    return jar


def _resolve_streamlink(url: str, quality: str, timeout_s: float,
                        cookies_file: Optional[str] = None) -> str:
    try:
        import streamlink
        from streamlink.stream.hls import HLSStream
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ResolveError("streamlink is not installed") from exc

    # Importing streamlink patches a private urllib3 global; make sure the
    # result is still usable by everything else in the process.
    ensure_urllib3_percent_re()

    session = streamlink.Streamlink()
    session.set_option("stream-timeout", timeout_s)
    session.set_option("http-timeout", timeout_s)
    if cookies_file:
        session.http.cookies.update(_load_cookie_jar(cookies_file))

    try:
        streams = session.streams(url)
    except Exception as exc:
        raise ResolveError(f"streamlink failed to list streams: {exc}") from exc

    if not streams:
        raise ResolveError("streamlink found no streams (is the broadcast live?)")

    stream = streams.get(quality) or streams.get("best")
    if stream is None:
        raise ResolveError(f"quality {quality!r} unavailable; have {sorted(streams)}")
    if not isinstance(stream, HLSStream):
        raise ResolveError(f"expected an HLS stream, got {type(stream).__name__}")
    return stream.url


def _resolve_ytdlp(url: str, quality: str, timeout_s: float,
                   cookies_file: Optional[str] = None) -> str:
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise ResolveError("yt-dlp is not installed") from exc

    # yt-dlp replaces urllib3's _PERCENT_RE with a shim that lacks .sub, which
    # breaks every subsequent HTTP request in the process -- track feeds
    # included -- and never recovers. We only get here as a fallback, so this
    # would otherwise turn one transient resolver failure into a dead service.
    ensure_urllib3_percent_re()

    # Prefer a muxed HLS rendition so audio rides along in the same segments.
    fmt = "best[protocol^=m3u8]/best"
    if quality not in ("best", "worst") and _QUALITY_RANK.match(quality):
        height = _QUALITY_RANK.match(quality).group(1)
        fmt = f"best[protocol^=m3u8][height<={height}]/{fmt}"

    opts = {
        "quiet": True,
        "no_warnings": True,
        "format": fmt,
        "socket_timeout": timeout_s,
        "noplaylist": True,
    }
    if cookies_file:
        # Validate up front so a bad file reads as a cookies problem rather
        # than an opaque extractor error.
        _load_cookie_jar(cookies_file)
        opts["cookiefile"] = cookies_file
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:
        raise ResolveError(f"yt-dlp failed: {exc}") from exc

    if not info.get("is_live", True):
        log.warning("yt-dlp reports the source is not live", extra={"url": url})

    manifest = info.get("manifest_url") or info.get("url")
    if not manifest:
        raise ResolveError("yt-dlp returned no manifest URL")
    return manifest


def _ensure_media_playlist(url: str, timeout_s: float, quality: str) -> str:
    """If ``url`` is a master playlist, choose a variant and return its URL."""
    from .playlist import is_master_playlist, select_variant

    try:
        resp = requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise ResolveError(f"could not fetch playlist: {exc}") from exc

    if not is_master_playlist(resp.text):
        return url

    variant = select_variant(resp.text, base_url=resp.url, quality=quality)
    if variant is None:
        raise ResolveError("master playlist contained no usable variant")
    log.info("selected variant from master playlist",
             extra={"quality": quality, "variant": variant[:120]})
    return variant


def resolve_stream(url: str,
                   resolver: str = "streamlink",
                   quality: str = "best",
                   timeout_s: float = 60.0,
                   cookies_file: Optional[str] = None) -> ResolvedStream:
    """Resolve ``url`` to an HLS media playlist, trying the other backend on failure.

    ``cookies_file`` is a Netscape-format cookies.txt sent with the YouTube
    requests; YouTube demands one from most datacenter IPs ("Sign in to
    confirm you're not a bot").
    """
    order = ["streamlink", "yt_dlp"] if resolver == "streamlink" else ["yt_dlp", "streamlink"]
    errors: list[str] = []

    for backend in order:
        started = time.monotonic()
        try:
            raw = (_resolve_streamlink(url, quality, timeout_s, cookies_file)
                   if backend == "streamlink"
                   else _resolve_ytdlp(url, quality, timeout_s, cookies_file))
            playlist = _ensure_media_playlist(raw, timeout_s, quality)
        except ResolveError as exc:
            errors.append(f"{backend}: {exc}")
            log.warning("resolver backend failed",
                        extra={"backend": backend, "error": str(exc)})
            continue

        resolved = ResolvedStream(
            playlist_url=playlist,
            quality=quality,
            backend=backend,
            expires_at=_expiry_from_url(playlist),
        )
        log.info("resolved stream", extra={
            "backend": backend,
            "quality": quality,
            "took_s": round(time.monotonic() - started, 2),
            "expires_in_s": (round(resolved.seconds_until_expiry())
                             if resolved.expires_at else None),
        })
        return resolved

    if any(_BOT_CHECK.search(e) for e in errors):
        # Logged on its own because the console formatter truncates the
        # combined error below well before any advice would show.
        if cookies_file:
            log.error("youtube rejected the cookies; export a fresh cookies.txt",
                      extra={"cookies_file": cookies_file})
        else:
            log.error("youtube wants a signed-in session from this IP; "
                      "put a cookies.txt in the working directory",
                      extra={"setting": "stream.cookies_file"})
    message = "all resolver backends failed: " + "; ".join(errors)
    raise ResolveError(message)
