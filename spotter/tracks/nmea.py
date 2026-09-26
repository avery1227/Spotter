"""AIVDM/AIVDO decoding for raw NMEA 0183 AIS input.

Implements the subset of ITU-R M.1371 that a shore-side camera overlay actually
needs:

===== ===============================================================
Type  Content
===== ===============================================================
1-3   Class A position report (the bulk of commercial traffic)
5     Class A static and voyage data -- name, callsign, type, destination
18    Class B position report (most recreational traffic)
19    Class B extended position report (position *and* name)
24    Class B static data, sent in two parts (A: name, B: type/callsign)
===== ===============================================================

Everything else is parsed far enough to identify and then ignored.

Payloads are packed six bits per character, so the decoder unpacks to a bit
array once and then slices fields out of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..logging_setup import get_logger

log = get_logger(__name__)

#: The AIS six-bit character set (ITU-R M.1371 Table 47).
SIXBIT_CHARS = ("@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_ !\"#$%&'()*+,-./0123456789:;<=>?")

#: Sentinel meaning "not available" for each field that has one.
SOG_NA = 1023
COG_NA = 3600
HEADING_NA = 511
LON_NA = 0x6791AC0
LAT_NA = 0x3412140

NAV_STATUS = {
    0: "under way using engine", 1: "at anchor", 2: "not under command",
    3: "restricted manoeuvrability", 4: "constrained by draught", 5: "moored",
    6: "aground", 7: "engaged in fishing", 8: "under way sailing",
    11: "towing astern", 12: "pushing ahead", 14: "AIS-SART", 15: "undefined",
}


class AISDecodeError(Exception):
    pass


def sixbit_to_bits(payload: str, fill_bits: int = 0) -> str:
    """Unpack an AIVDM payload into a string of '0'/'1'.

    Each character carries six bits: subtract 48, and subtract a further 8 if
    the result exceeds 40, which skips the unused ASCII range.
    """
    bits = []
    for char in payload:
        value = ord(char) - 48
        if value > 40:
            value -= 8
        if value < 0 or value > 63:
            raise AISDecodeError(f"invalid payload character {char!r}")
        bits.append(f"{value:06b}")
    out = "".join(bits)
    if fill_bits:
        out = out[:len(out) - fill_bits] if fill_bits < len(out) else ""
    return out


def _uint(bits: str, start: int, length: int) -> Optional[int]:
    chunk = bits[start:start + length]
    if len(chunk) < length:
        return None
    return int(chunk, 2)


def _int(bits: str, start: int, length: int) -> Optional[int]:
    """Two's-complement signed field."""
    value = _uint(bits, start, length)
    if value is None:
        return None
    if value & (1 << (length - 1)):
        value -= 1 << length
    return value


def _text(bits: str, start: int, length: int) -> Optional[str]:
    chunk = bits[start:start + length]
    usable = len(chunk) - (len(chunk) % 6)
    if usable <= 0:
        return None
    out = []
    for i in range(0, usable, 6):
        index = int(chunk[i:i + 6], 2)
        out.append(SIXBIT_CHARS[index])
    # '@' pads the field; text ends at the first one.
    text = "".join(out).split("@")[0].strip()
    return text or None


@dataclass
class AISMessage:
    """A decoded AIS message, with only the fields we care about populated."""

    msg_type: int
    mmsi: int
    lat: Optional[float] = None
    lon: Optional[float] = None
    sog_kn: Optional[float] = None
    cog_deg: Optional[float] = None
    heading_deg: Optional[float] = None
    nav_status: Optional[str] = None
    turn_rate: Optional[float] = None
    name: Optional[str] = None
    callsign: Optional[str] = None
    ship_type: Optional[int] = None
    destination: Optional[str] = None
    imo: Optional[int] = None
    draught_m: Optional[float] = None
    length_m: Optional[float] = None
    beam_m: Optional[float] = None
    part_number: Optional[int] = None
    raw_timestamp: Optional[int] = None

    @property
    def has_position(self) -> bool:
        return self.lat is not None and self.lon is not None


def _decode_position_common(bits: str, msg: AISMessage, sog_at: int, lon_at: int,
                            lat_at: int, cog_at: int, heading_at: int,
                            stamp_at: int) -> None:
    """Shared field layout between types 1-3, 18 and 19 (offsets differ)."""
    sog = _uint(bits, sog_at, 10)
    if sog is not None and sog != SOG_NA:
        msg.sog_kn = sog / 10.0

    lon = _int(bits, lon_at, 28)
    lat = _int(bits, lat_at, 27)
    if lon is not None and lat is not None and lon != LON_NA and lat != LAT_NA:
        # Stored in 1/10000 minutes of arc.
        candidate_lon = lon / 600000.0
        candidate_lat = lat / 600000.0
        if -180.0 <= candidate_lon <= 180.0 and -90.0 <= candidate_lat <= 90.0:
            msg.lon, msg.lat = candidate_lon, candidate_lat

    cog = _uint(bits, cog_at, 12)
    if cog is not None and cog != COG_NA:
        msg.cog_deg = (cog / 10.0) % 360.0

    heading = _uint(bits, heading_at, 9)
    if heading is not None and heading != HEADING_NA:
        msg.heading_deg = float(heading % 360)

    msg.raw_timestamp = _uint(bits, stamp_at, 6)


def _dimensions(bits: str, bow_at: int, bow_len: int = 9,
                port_len: int = 6) -> tuple[Optional[float], Optional[float]]:
    """Reconstruct length and beam from the four reference-point distances."""
    to_bow = _uint(bits, bow_at, bow_len)
    to_stern = _uint(bits, bow_at + bow_len, bow_len)
    to_port = _uint(bits, bow_at + 2 * bow_len, port_len)
    to_starboard = _uint(bits, bow_at + 2 * bow_len + port_len, port_len)
    length = (to_bow + to_stern) if (to_bow is not None and to_stern is not None) else None
    beam = (to_port + to_starboard) if (to_port is not None and
                                        to_starboard is not None) else None
    return (float(length) if length else None, float(beam) if beam else None)


def decode_payload(payload: str, fill_bits: int = 0) -> Optional[AISMessage]:
    """Decode one complete (reassembled) AIVDM payload."""
    bits = sixbit_to_bits(payload, fill_bits)
    if len(bits) < 38:
        return None

    msg_type = _uint(bits, 0, 6)
    mmsi = _uint(bits, 8, 30)
    if msg_type is None or mmsi is None:
        return None
    msg = AISMessage(msg_type=msg_type, mmsi=mmsi)

    if msg_type in (1, 2, 3):
        if len(bits) < 143:
            return None
        status = _uint(bits, 38, 4)
        msg.nav_status = NAV_STATUS.get(status)
        rot = _int(bits, 42, 8)
        if rot is not None and rot != -128:
            # Encoded as 4.733 * sqrt(deg/min), signed.
            msg.turn_rate = (rot / 4.733) ** 2 * (1 if rot >= 0 else -1)
        _decode_position_common(bits, msg, 50, 61, 89, 116, 128, 137)

    elif msg_type == 18:
        if len(bits) < 139:
            return None
        _decode_position_common(bits, msg, 46, 57, 85, 112, 124, 133)

    elif msg_type == 19:
        if len(bits) < 139:
            return None
        _decode_position_common(bits, msg, 46, 57, 85, 112, 124, 133)
        msg.name = _text(bits, 143, 120)
        msg.ship_type = _uint(bits, 263, 8)
        msg.length_m, msg.beam_m = _dimensions(bits, 271)

    elif msg_type == 5:
        if len(bits) < 302:
            return None
        msg.imo = _uint(bits, 40, 30) or None
        msg.callsign = _text(bits, 70, 42)
        msg.name = _text(bits, 112, 120)
        msg.ship_type = _uint(bits, 232, 8)
        msg.length_m, msg.beam_m = _dimensions(bits, 240)
        draught = _uint(bits, 294, 8)
        if draught:
            msg.draught_m = draught / 10.0
        msg.destination = _text(bits, 302, 120)

    elif msg_type == 24:
        part = _uint(bits, 38, 2)
        msg.part_number = part
        if part == 0:
            msg.name = _text(bits, 40, 120)
        elif part == 1:
            msg.ship_type = _uint(bits, 40, 8)
            msg.callsign = _text(bits, 90, 42)
            msg.length_m, msg.beam_m = _dimensions(bits, 132)

    else:
        # Types 4, 9, 21, 27 and friends: recognised, not useful here.
        return msg

    return msg


@dataclass
class _Fragment:
    parts: dict[int, str] = field(default_factory=dict)
    total: int = 0
    fill_bits: int = 0
    last_seen: float = 0.0


class NMEADecoder:
    """Stateful decoder handling multi-sentence AIVDM reassembly.

    Type 5 messages do not fit in one sentence, so they arrive split across two
    with a shared sequential message ID. Fragments are keyed by
    ``(channel, sequence id)`` and abandoned if the rest never turns up.
    """

    def __init__(self, fragment_timeout_s: float = 30.0):
        self.fragment_timeout_s = fragment_timeout_s
        self._fragments: dict[tuple[str, str], _Fragment] = {}
        self.decoded = 0
        self.errors = 0

    @staticmethod
    def _checksum_ok(sentence: str) -> bool:
        if "*" not in sentence:
            return True  # some feeds omit it; accept rather than drop data
        body, _, checksum = sentence.rpartition("*")
        body = body.lstrip("!$")
        try:
            expected = int(checksum[:2], 16)
        except ValueError:
            return False
        actual = 0
        for char in body:
            actual ^= ord(char)
        return actual == expected

    def feed_line(self, line: str, now: float = 0.0) -> list[AISMessage]:
        """Feed one NMEA sentence; returns any messages it completed."""
        line = line.strip()
        if not line or ("AIVDM" not in line and "AIVDO" not in line):
            return []

        # Some feeds prefix a tag block or timestamp before the '!'.
        start = line.find("!")
        if start > 0:
            line = line[start:]

        if not self._checksum_ok(line):
            self.errors += 1
            log.debug("NMEA checksum failure", extra={"line": line[:80]})
            return []

        fields = line.split("*")[0].split(",")
        if len(fields) < 6:
            self.errors += 1
            return []

        try:
            total = int(fields[1])
            index = int(fields[2])
        except ValueError:
            self.errors += 1
            return []

        seq_id = fields[3]
        channel = fields[4] or "A"
        payload = fields[5]
        try:
            fill_bits = int(fields[6]) if len(fields) > 6 and fields[6] else 0
        except ValueError:
            fill_bits = 0

        if total <= 1:
            return self._decode(payload, fill_bits)

        key = (channel, seq_id)
        entry = self._fragments.get(key)
        if entry is None or entry.total != total:
            entry = _Fragment(total=total)
            self._fragments[key] = entry
        entry.parts[index] = payload
        entry.last_seen = now
        if index == total:
            entry.fill_bits = fill_bits

        if len(entry.parts) == total:
            del self._fragments[key]
            joined = "".join(entry.parts[i] for i in sorted(entry.parts))
            return self._decode(joined, entry.fill_bits)

        self._expire(now)
        return []

    def _decode(self, payload: str, fill_bits: int) -> list[AISMessage]:
        try:
            message = decode_payload(payload, fill_bits)
        except (AISDecodeError, ValueError, IndexError) as exc:
            self.errors += 1
            log.debug("AIS decode failed", extra={"error": str(exc)})
            return []
        if message is None:
            return []
        self.decoded += 1
        return [message]

    def _expire(self, now: float) -> None:
        if not now:
            return
        stale = [k for k, v in self._fragments.items()
                 if v.last_seen and now - v.last_seen > self.fragment_timeout_s]
        for key in stale:
            del self._fragments[key]
