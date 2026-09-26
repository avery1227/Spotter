"""Repairs for third-party monkeypatches that break other libraries.

Nothing here is nice. It exists because two of our dependencies patch the same
private urllib3 global in incompatible ways, and the loser is every HTTP call
the process makes afterwards.
"""

from __future__ import annotations

import re

from .logging_setup import get_logger

log = get_logger(__name__)

#: urllib3's own definition, which is what everything else expects to find.
_PERCENT_PATTERN = re.compile(r"%[a-fA-F0-9]{2}")


class _PercentRe:
    """A percent-encoding regex that satisfies both urllib3 and its patchers.

    ``subn`` is delegated to whatever override is currently installed, because
    streamlink and yt-dlp both replace it deliberately: they return the string
    unchanged so that urllib3 does not re-encode sequences that are already
    percent-encoded. Everything else goes to a real compiled pattern, which is
    the part they forget to provide.
    """

    __slots__ = ("_delegate",)

    def __init__(self, delegate=None):
        self._delegate = delegate

    def subn(self, repl, string, count=0):
        delegate_subn = getattr(self._delegate, "subn", None)
        if delegate_subn is not None:
            try:
                return delegate_subn(repl, string, count)
            except TypeError:
                # yt-dlp's takes (repl, string) only.
                return delegate_subn(repl, string)
        return _PERCENT_PATTERN.subn(repl, string, count)

    def __getattr__(self, item):
        return getattr(_PERCENT_PATTERN, item)


def ensure_urllib3_percent_re() -> bool:
    """Make sure ``urllib3.util.url._PERCENT_RE`` is usable. Returns True if repaired.

    streamlink replaces this global with a shim that proxies unknown attributes
    to the original compiled pattern. yt-dlp replaces it with one that does
    *not* -- it provides only ``subn``. Importing yt-dlp therefore leaves
    ``_PERCENT_RE.sub`` missing, and urllib3 raises

        AttributeError: 'Urllib3PercentREOverride' object has no attribute 'sub'

    on essentially every request from then on. We import yt-dlp only as a
    fallback when streamlink fails to resolve the stream, so a single transient
    resolver failure would otherwise poison every HTTP call -- track feeds
    included -- for the remaining life of the process. It does not recover on
    its own.

    Call this after importing either library.
    """
    try:
        import urllib3.util.url as url_module
    except Exception:                                   # pragma: no cover
        return False

    current = getattr(url_module, "_PERCENT_RE", None)
    if current is not None and hasattr(current, "sub") and hasattr(current, "subn"):
        return False

    url_module._PERCENT_RE = _PercentRe(current)
    log.warning("repaired urllib3._PERCENT_RE after a third-party patch left it "
                "incomplete; HTTP would otherwise fail process-wide",
                extra={"was": type(current).__name__})
    return True


def apply_all() -> None:
    """Every repair, applied once at startup."""
    ensure_urllib3_percent_re()
