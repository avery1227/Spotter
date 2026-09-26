"""Repairs for third-party monkeypatches.

This exists because of a real outage: a single transient stream-resolve
failure made the process import yt-dlp as a fallback, which replaced urllib3's
``_PERCENT_RE`` with a shim missing ``.sub``. Every HTTP call afterwards --
the ADS-B and AIS feeds included -- failed, and the process never recovered.
It sat five and a half hours behind live until it was restarted.
"""

import re

import pytest

from spotter.compat import _PERCENT_PATTERN, ensure_urllib3_percent_re


class BrokenOverride:
    """What yt-dlp installs: subn only, no sub."""

    def subn(self, repl, string, count=0):
        return string, 0


class ProxyingOverride:
    """What streamlink installs: proxies everything else to the real pattern."""

    _inner = re.compile(r"%[a-fA-F0-9]{2}")

    def __getattr__(self, item):
        return getattr(self._inner, item)

    def subn(self, repl, string, count=0):
        return string, len(self._inner.findall(string))


@pytest.fixture
def url_module():
    import urllib3.util.url as module

    original = module._PERCENT_RE
    yield module
    module._PERCENT_RE = original


def test_untouched_pattern_is_left_alone(url_module):
    url_module._PERCENT_RE = _PERCENT_PATTERN
    assert ensure_urllib3_percent_re() is False
    assert url_module._PERCENT_RE is _PERCENT_PATTERN


def test_a_proxying_override_is_left_alone(url_module):
    """streamlink's shim works; replacing it would be gratuitous."""
    override = ProxyingOverride()
    url_module._PERCENT_RE = override
    assert ensure_urllib3_percent_re() is False
    assert url_module._PERCENT_RE is override


def test_an_override_missing_sub_is_repaired(url_module):
    url_module._PERCENT_RE = BrokenOverride()
    assert not hasattr(url_module._PERCENT_RE, "sub")

    assert ensure_urllib3_percent_re() is True

    repaired = url_module._PERCENT_RE
    assert hasattr(repaired, "sub")
    assert hasattr(repaired, "subn")
    # .sub behaves like the real thing again.
    assert repaired.sub(lambda m: m.group(0).upper(), "a%2fb") == "a%2Fb"


def test_repair_preserves_the_patcher_s_intent(url_module):
    """The overrides exist for a reason: stop urllib3 re-encoding.

    Both streamlink and yt-dlp replace subn so that an already percent-encoded
    string is returned untouched. The repair must keep that, or we fix one bug
    by reintroducing the one they were working around.
    """
    url_module._PERCENT_RE = BrokenOverride()
    ensure_urllib3_percent_re()
    text, _count = url_module._PERCENT_RE.subn(
        lambda m: m.group(0).upper(), "already%2fencoded")
    assert text == "already%2fencoded"


def test_repair_is_idempotent(url_module):
    url_module._PERCENT_RE = BrokenOverride()
    assert ensure_urllib3_percent_re() is True
    assert ensure_urllib3_percent_re() is False


def test_requests_survives_the_real_import_order(url_module):
    """The actual failure: import streamlink, then yt-dlp, then use requests."""
    # importorskip performs the import; that is the whole point of the test,
    # so the modules are deliberately not bound to names.
    pytest.importorskip("streamlink")
    pytest.importorskip("yt_dlp")

    ensure_urllib3_percent_re()

    # urllib3 reaches for .sub while parsing a URL; without the repair this
    # raises AttributeError rather than making a request.
    from urllib3.util.url import parse_url
    assert parse_url("https://example.com/a%2fb").host == "example.com"
    assert hasattr(url_module._PERCENT_RE, "sub")
