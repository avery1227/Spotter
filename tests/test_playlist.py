"""HLS playlist parsing -- the part every frame timestamp depends on."""

from datetime import datetime, timezone

import pytest

from spotter.ingest.playlist import (is_master_playlist, newer_than,
                                     parse_media_playlist, select_variant)

MEDIA = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:5
#EXT-X-MEDIA-SEQUENCE:825253
#EXT-X-PROGRAM-DATE-TIME:2026-09-25T09:53:20.014+00:00
#EXTINF:5.0,
https://example.com/seg0.ts
#EXTINF:5.0,
https://example.com/seg1.ts
#EXTINF:4.0,
https://example.com/seg2.ts
#EXTINF:5.0,
https://example.com/seg3.ts
"""

MASTER = """#EXTM3U
#EXT-X-STREAM-INF:BANDWIDTH=800000,RESOLUTION=640x360
low.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=3000000,RESOLUTION=1280x720
mid.m3u8
#EXT-X-STREAM-INF:BANDWIDTH=6000000,RESOLUTION=1920x1080
high.m3u8
"""


def test_media_playlist_basics():
    pl = parse_media_playlist(MEDIA, url="https://example.com/index.m3u8")
    assert len(pl.segments) == 4
    assert pl.target_duration == 5.0
    assert pl.media_sequence == 825253
    assert not pl.endlist
    assert [s.sequence for s in pl.segments] == [825253, 825254, 825255, 825256]


def test_pdt_accumulates_using_actual_durations():
    """Only the first segment carries a PDT tag; the rest must be derived.

    The third segment is 4 seconds, not 5, so a naive "index * target_duration"
    would put every later segment a second off -- which is 30 frames of error.
    """
    pl = parse_media_playlist(MEDIA, url="https://example.com/index.m3u8")
    base = datetime(2026, 9, 25, 9, 53, 20, 14000, tzinfo=timezone.utc)

    assert pl.segments[0].pdt == base
    assert pl.segments[0].pdt_inferred is False
    for segment in pl.segments[1:]:
        assert segment.pdt_inferred is True

    offsets = [(s.pdt - base).total_seconds() for s in pl.segments]
    assert offsets == [0.0, 5.0, 10.0, 14.0]
    assert (pl.live_edge - base).total_seconds() == 19.0
    assert pl.window_duration() == 19.0


def test_discontinuity_resets_the_derived_clock():
    text = """#EXTM3U
#EXT-X-TARGETDURATION:4
#EXT-X-MEDIA-SEQUENCE:10
#EXT-X-PROGRAM-DATE-TIME:2026-01-01T00:00:00Z
#EXTINF:4.0,
a.ts
#EXT-X-DISCONTINUITY
#EXTINF:4.0,
b.ts
#EXT-X-PROGRAM-DATE-TIME:2026-01-01T01:00:00Z
#EXTINF:4.0,
c.ts
"""
    pl = parse_media_playlist(text)
    # After a discontinuity the accumulated clock is meaningless, so rather
    # than guess we report no PDT until a fresh tag arrives.
    assert pl.segments[0].pdt is not None
    assert pl.segments[1].discontinuity is True
    assert pl.segments[1].pdt is None
    assert pl.segments[2].pdt == datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("stamp,expected_hour", [
    ("2026-01-01T12:00:00Z", 12),
    ("2026-01-01T12:00:00+00:00", 12),
    ("2026-01-01T07:00:00-05:00", 12),
    ("2026-01-01T12:00:00.123456789Z", 12),   # more than 6 fractional digits
])
def test_pdt_timestamp_formats(stamp, expected_hour):
    text = (f"#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXT-X-MEDIA-SEQUENCE:1\n"
            f"#EXT-X-PROGRAM-DATE-TIME:{stamp}\n#EXTINF:4.0,\na.ts\n")
    pdt = parse_media_playlist(text).segments[0].pdt
    assert pdt is not None and pdt.tzinfo is not None
    assert pdt.astimezone(timezone.utc).hour == expected_hour


def test_master_playlist_variant_selection():
    assert is_master_playlist(MASTER)
    assert not is_master_playlist(MEDIA)
    base = "https://example.com/master.m3u8"
    assert select_variant(MASTER, base, "best").endswith("high.m3u8")
    assert select_variant(MASTER, base, "worst").endswith("low.m3u8")
    assert select_variant(MASTER, base, "720p").endswith("mid.m3u8")
    # A cap between variants picks the best one that fits under it.
    assert select_variant(MASTER, base, "1080p").endswith("high.m3u8")
    assert select_variant(MASTER, base, "480p").endswith("low.m3u8")


def test_newer_than_filters_by_sequence():
    pl = parse_media_playlist(MEDIA, url="https://example.com/index.m3u8")
    assert len(newer_than(pl.segments, None)) == 4
    assert len(newer_than(pl.segments, 825254)) == 2
    assert newer_than(pl.segments, 999999) == []


def test_relative_segment_uris_resolve_against_the_playlist():
    text = ("#EXTM3U\n#EXT-X-TARGETDURATION:4\n#EXT-X-MEDIA-SEQUENCE:1\n"
            "#EXTINF:4.0,\nchunk/0.ts\n")
    pl = parse_media_playlist(text, url="https://cdn.example.com/a/b/index.m3u8")
    assert pl.segments[0].uri == "https://cdn.example.com/a/b/chunk/0.ts"


def test_empty_and_garbage_playlists_do_not_raise():
    assert parse_media_playlist("").segments == []
    assert parse_media_playlist("not a playlist at all").segments == []
