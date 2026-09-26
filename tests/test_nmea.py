"""AIVDM decoding, checked against published test vectors."""

import pytest

from spotter.tracks.nmea import (SIXBIT_CHARS, NMEADecoder, decode_payload,
                                 sixbit_to_bits)


def checksum(body: str) -> str:
    value = 0
    for char in body:
        value ^= ord(char)
    return f"{value:02X}"


def sentence(payload: str, fill: int = 0, total: int = 1, index: int = 1,
             seq: str = "", channel: str = "A") -> str:
    body = f"AIVDM,{total},{index},{seq},{channel},{payload},{fill}"
    return f"!{body}*{checksum(body)}"


# ---------------------------------------------------------------------------
# Encoder used only by the tests, so message layouts can be checked field by
# field rather than against a handful of opaque published strings.
# ---------------------------------------------------------------------------

def bits_of(value: int, length: int) -> str:
    if value < 0:
        value += 1 << length
    return f"{value:0{length}b}"


def text_bits(text: str, chars: int) -> str:
    padded = (text.upper() + "@" * chars)[:chars]
    return "".join(f"{SIXBIT_CHARS.index(c):06b}" for c in padded)


def to_payload(bits: str) -> tuple[str, int]:
    pad = (-len(bits)) % 6
    bits += "0" * pad
    out = []
    for i in range(0, len(bits), 6):
        value = int(bits[i:i + 6], 2)
        out.append(chr(value + 48 + (8 if value > 39 else 0)))
    return "".join(out), pad


def class_b_position(mmsi=338123456, lat=41.1234, lon=-72.6543,
                     sog=7.3, cog=143.2, heading=145, msg_type=18) -> str:
    bits = bits_of(msg_type, 6) + bits_of(0, 2) + bits_of(mmsi, 30)
    bits += bits_of(0, 8)
    bits += bits_of(round(sog * 10), 10)
    bits += bits_of(1, 1)
    bits += bits_of(round(lon * 600000), 28)
    bits += bits_of(round(lat * 600000), 27)
    bits += bits_of(round(cog * 10), 12)
    bits += bits_of(heading, 9)
    bits += bits_of(42, 6)
    return bits


# ---------------------------------------------------------------------------
# Published vectors
# ---------------------------------------------------------------------------

def test_type_1_position_report():
    decoder = NMEADecoder()
    messages = decoder.feed_line("!AIVDM,1,1,,A,13HOI:0P0000VOHLCnHQKwvL05Ip,0*23")
    assert len(messages) == 1
    message = messages[0]
    assert message.msg_type == 1
    assert message.mmsi == 227006760
    # Le Havre approaches.
    assert message.lat == pytest.approx(49.47558, abs=1e-4)
    assert message.lon == pytest.approx(0.13138, abs=1e-4)
    assert message.nav_status == "under way using engine"


def test_type_5_static_and_voyage_spans_two_sentences():
    decoder = NMEADecoder()
    first = decoder.feed_line(
        "!AIVDM,2,1,1,A,55?MbV02;H;s<HtKR20EHE:0@T4@Dn2222222216L961O5Gf0NSQEp6ClRp8,0*1C",
        now=1.0)
    assert first == [], "a half-received message must not be emitted"

    second = decoder.feed_line("!AIVDM,2,2,1,A,88888888880,2*25", now=1.0)
    assert len(second) == 1
    message = second[0]
    assert message.msg_type == 5
    assert message.mmsi == 351759000
    assert message.name == "EVER DIADEM"
    assert message.callsign == "3FOF8"
    assert message.destination == "NEW YORK"
    assert message.imo == 9134270
    assert message.ship_type == 70
    assert message.length_m == 295.0
    assert message.beam_m == 32.0
    assert message.draught_m == pytest.approx(12.2)


def test_type_24_part_a_carries_the_name():
    decoder = NMEADecoder()
    messages = decoder.feed_line("!AIVDM,1,1,,A,H42O55i18tMET00000000000000,2*6D")
    assert len(messages) == 1
    assert messages[0].msg_type == 24
    assert messages[0].part_number == 0
    assert messages[0].name == "PROGUY"


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------

def test_type_18_round_trip():
    payload, fill = to_payload(class_b_position())
    message = decode_payload(payload, fill)
    assert message.msg_type == 18
    assert message.mmsi == 338123456
    assert message.lat == pytest.approx(41.1234, abs=1e-5)
    assert message.lon == pytest.approx(-72.6543, abs=1e-5)
    assert message.sog_kn == pytest.approx(7.3)
    assert message.cog_deg == pytest.approx(143.2)
    assert message.heading_deg == 145


def test_type_19_carries_position_and_name():
    bits = class_b_position(msg_type=19)
    bits += bits_of(0, 4) + text_bits("MISS MARIE", 20) + bits_of(37, 8)
    bits += bits_of(12, 9) + bits_of(4, 9) + bits_of(2, 6) + bits_of(3, 6)
    message = decode_payload(*to_payload(bits))
    assert message.msg_type == 19
    assert message.name == "MISS MARIE"
    assert message.ship_type == 37
    assert message.length_m == 16.0
    assert message.beam_m == 5.0
    assert message.lat == pytest.approx(41.1234, abs=1e-5)


def test_type_24_part_b_carries_type_and_callsign():
    bits = bits_of(24, 6) + bits_of(0, 2) + bits_of(338123456, 30) + bits_of(1, 2)
    bits += bits_of(36, 8) + text_bits("VENDOR", 7) + text_bits("WDF1234", 7)
    bits += bits_of(8, 9) + bits_of(4, 9) + bits_of(2, 6) + bits_of(2, 6)
    message = decode_payload(*to_payload(bits))
    assert message.part_number == 1
    assert message.callsign == "WDF1234"
    assert message.ship_type == 36
    assert message.length_m == 12.0


def test_not_available_sentinels_become_none():
    bits = class_b_position(sog=102.3, cog=360.0, heading=511)
    message = decode_payload(*to_payload(bits))
    assert message.sog_kn is None
    assert message.cog_deg is None
    assert message.heading_deg is None


# ---------------------------------------------------------------------------
# Framing robustness
# ---------------------------------------------------------------------------

def test_bad_checksum_is_rejected():
    decoder = NMEADecoder()
    assert decoder.feed_line("!AIVDM,1,1,,A,13HOI:0P0000VOHLCnHQKwvL05Ip,0*FF") == []
    assert decoder.errors == 1


def test_non_ais_sentences_are_ignored_silently():
    decoder = NMEADecoder()
    assert decoder.feed_line("$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,,,,,,*47") == []
    assert decoder.feed_line("") == []
    assert decoder.errors == 0


def test_tag_block_prefix_is_stripped():
    decoder = NMEADecoder()
    line = ("\\s:SHORESTATION,c:1700000000*00\\"
            "!AIVDM,1,1,,A,13HOI:0P0000VOHLCnHQKwvL05Ip,0*23")
    assert len(decoder.feed_line(line)) == 1


def test_interleaved_fragments_on_different_channels():
    """Two multipart messages in flight at once must not be spliced together."""
    decoder = NMEADecoder()
    payload_a, fill_a = to_payload(class_b_position(mmsi=111111111))
    payload_b, fill_b = to_payload(class_b_position(mmsi=222222222))

    half_a1, half_a2 = payload_a[:14], payload_a[14:]
    half_b1, half_b2 = payload_b[:14], payload_b[14:]

    assert decoder.feed_line(sentence(half_a1, 0, 2, 1, "1", "A"), now=1.0) == []
    assert decoder.feed_line(sentence(half_b1, 0, 2, 1, "2", "B"), now=1.0) == []

    out_a = decoder.feed_line(sentence(half_a2, fill_a, 2, 2, "1", "A"), now=1.0)
    out_b = decoder.feed_line(sentence(half_b2, fill_b, 2, 2, "2", "B"), now=1.0)
    assert [m.mmsi for m in out_a] == [111111111]
    assert [m.mmsi for m in out_b] == [222222222]


def test_stale_fragments_are_expired():
    decoder = NMEADecoder(fragment_timeout_s=10.0)
    decoder.feed_line(sentence("55?MbV02;H;s", 0, 2, 1, "1"), now=1.0)
    assert len(decoder._fragments) == 1
    # A later incomplete message sweeps the abandoned one out.
    decoder.feed_line(sentence("55?MbV02;H;s", 0, 2, 1, "2"), now=100.0)
    assert ("A", "1") not in decoder._fragments


def test_sixbit_fill_bits_are_trimmed():
    assert len(sixbit_to_bits("000", 0)) == 18
    assert len(sixbit_to_bits("000", 2)) == 16
