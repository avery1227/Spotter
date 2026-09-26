"""AIS sources: AIS-catcher over HTTP, raw NMEA over UDP, and aisstream.io.

All three end up producing the same :class:`TrackReport`. The interesting part
is that AIS splits a vessel across message types: position comes in types 1-3
and 18, while the name, type and destination arrive separately in types 5, 19
and 24. The track store merges labels across reports, so a vessel picks up its
name as soon as a static message turns up, without losing it afterwards.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Sequence

import requests

from ..logging_setup import get_logger
from ..util import as_float, as_int, clean_str, utcnow
from .model import KNOTS_TO_MPS, TrackKind, TrackReport
from .nmea import AISMessage, NMEADecoder
from .sources import Emit, PollingSource, TrackSource, check_rate_limit

log = get_logger(__name__)

#: Speeds at or above this are AIS's "not available" encoding, not real motion.
SOG_SENTINEL_KN = 102.3


def _track_id(mmsi) -> Optional[str]:
    value = as_int(mmsi)
    if not value or value <= 0:
        return None
    return f"mmsi:{value}"


def _labels_from(name=None, callsign=None, ship_type=None, destination=None,
                 imo=None, mmsi=None, length_m=None, beam_m=None,
                 draught_m=None, nav_status=None) -> dict:
    labels = {
        "mmsi": as_int(mmsi),
        "name": clean_str(name),
        "callsign": clean_str(callsign),
        "ship_type": as_int(ship_type),
        "destination": clean_str(destination),
        "imo": as_int(imo) or None,
        "length_m": as_float(length_m),
        "beam_m": as_float(beam_m),
        "draught_m": as_float(draught_m),
        "nav_status": clean_str(nav_status),
    }
    return {k: v for k, v in labels.items() if v not in (None, "", 0)}


def report_from_ais_message(message: AISMessage, source: str,
                            when: Optional[datetime] = None
                            ) -> Optional[TrackReport]:
    """Turn a decoded NMEA message into a report, if it carries a position.

    Static-only messages (types 5 and 24) have no position of their own. They
    are still valuable -- they carry the vessel's name -- so they are attached
    to the vessel's last known position rather than dropped. The caller handles
    that by way of :class:`AISAssembler`.
    """
    track_id = _track_id(message.mmsi)
    if track_id is None or not message.has_position:
        return None

    speed = None
    if message.sog_kn is not None and message.sog_kn < SOG_SENTINEL_KN:
        speed = message.sog_kn * KNOTS_TO_MPS

    return TrackReport(
        track_id=track_id,
        kind=TrackKind.SHIP,
        timestamp=when or utcnow(),
        lat=message.lat, lon=message.lon, alt_m=0.0,
        speed_mps=speed,
        course_deg=message.cog_deg,
        heading_deg=message.heading_deg,
        labels=_labels_from(name=message.name, callsign=message.callsign,
                            ship_type=message.ship_type,
                            destination=message.destination, imo=message.imo,
                            mmsi=message.mmsi, length_m=message.length_m,
                            beam_m=message.beam_m, draught_m=message.draught_m,
                            nav_status=message.nav_status),
        source=source,
    )


class AISAssembler:
    """Joins static AIS data to position reports for the same vessel.

    A type 5 or 24 message names a vessel but says nothing about where it is.
    Rather than discard it, we remember the static fields and attach them to
    that MMSI's next position report -- and, if we have already seen a position,
    emit a synthetic report immediately so the name appears without waiting for
    the vessel's next transmission.
    """

    def __init__(self, max_vessels: int = 5000):
        self.max_vessels = max_vessels
        self._static: dict[str, dict] = {}
        self._last_position: dict[str, TrackReport] = {}

    def feed(self, message: AISMessage, source: str,
             when: Optional[datetime] = None) -> list[TrackReport]:
        track_id = _track_id(message.mmsi)
        if track_id is None:
            return []

        static = _labels_from(
            name=message.name, callsign=message.callsign,
            ship_type=message.ship_type, destination=message.destination,
            imo=message.imo, mmsi=message.mmsi, length_m=message.length_m,
            beam_m=message.beam_m, draught_m=message.draught_m)

        if message.has_position:
            report = report_from_ais_message(message, source, when)
            if report is None:
                return []
            merged = dict(self._static.get(track_id, {}))
            merged.update(report.labels)
            report.labels = merged
            self._last_position[track_id] = report
            if static:
                self._remember(track_id, static)
            self._evict()
            return [report]

        # Static-only message.
        if not static:
            return []
        changed = self._remember(track_id, static)
        previous = self._last_position.get(track_id)
        if not changed or previous is None:
            return []

        # Re-emit the last known position carrying the new descriptive fields,
        # timestamped when that position was measured so it interpolates
        # identically and does not look like fresh movement.
        refreshed = TrackReport(
            track_id=previous.track_id, kind=previous.kind,
            timestamp=previous.timestamp, lat=previous.lat, lon=previous.lon,
            alt_m=previous.alt_m, speed_mps=previous.speed_mps,
            course_deg=previous.course_deg, heading_deg=previous.heading_deg,
            labels={**previous.labels, **self._static[track_id]},
            source=source)
        self._last_position[track_id] = refreshed
        return [refreshed]

    def _remember(self, track_id: str, static: dict) -> bool:
        existing = self._static.setdefault(track_id, {})
        changed = any(existing.get(k) != v for k, v in static.items())
        existing.update(static)
        return changed

    def _evict(self) -> None:
        if len(self._last_position) <= self.max_vessels:
            return
        oldest = sorted(self._last_position.items(),
                        key=lambda kv: kv[1].timestamp)[:len(self._last_position) // 4]
        for track_id, _ in oldest:
            self._last_position.pop(track_id, None)
            self._static.pop(track_id, None)


# ---------------------------------------------------------------------------
# AIS-catcher over HTTP
# ---------------------------------------------------------------------------

def _first(entry: dict, *keys, default=None):
    for key in keys:
        if key in entry and entry[key] not in (None, ""):
            return entry[key]
    return default


def parse_aiscatcher_vessel(entry: dict, source: str,
                            now_epoch: float) -> Optional[TrackReport]:
    """Map one AIS-catcher vessel record. Key names vary by version, so be broad."""
    track_id = _track_id(_first(entry, "mmsi", "MMSI", "userid"))
    if track_id is None:
        return None

    lat = as_float(_first(entry, "lat", "latitude", "LAT"))
    lon = as_float(_first(entry, "lon", "longitude", "LON"))
    if lat is None or lon is None:
        return None

    speed_kn = as_float(_first(entry, "speed", "sog", "SOG"))
    if speed_kn is not None and speed_kn >= SOG_SENTINEL_KN:
        speed_kn = None

    heading = as_float(_first(entry, "heading", "true_heading", "hdg"))
    if heading is not None and heading >= 511:
        heading = None

    course = as_float(_first(entry, "cog", "course", "COG"))
    if course is not None and course >= 360:
        course = None

    # AIS-catcher reports how long ago the vessel was last heard, in seconds.
    age = as_float(_first(entry, "last_signal", "seen", "age"), 0.0) or 0.0
    timestamp = datetime.fromtimestamp(now_epoch - age, tz=timezone.utc)

    to_bow = as_float(_first(entry, "to_bow", "dim_bow"))
    to_stern = as_float(_first(entry, "to_stern", "dim_stern"))
    length = (to_bow + to_stern) if (to_bow is not None and to_stern is not None) else \
        as_float(_first(entry, "length", "length_m"))

    return TrackReport(
        track_id=track_id, kind=TrackKind.SHIP, timestamp=timestamp,
        lat=lat, lon=lon, alt_m=0.0,
        speed_mps=None if speed_kn is None else speed_kn * KNOTS_TO_MPS,
        course_deg=course, heading_deg=heading,
        labels=_labels_from(
            name=_first(entry, "shipname", "name", "NAME"),
            callsign=_first(entry, "callsign", "call_sign"),
            ship_type=_first(entry, "shiptype", "ship_type", "type"),
            destination=_first(entry, "destination", "dest"),
            imo=_first(entry, "imo", "IMO"),
            mmsi=_first(entry, "mmsi", "MMSI", "userid"),
            length_m=length,
            draught_m=_first(entry, "draught", "draft"),
            nav_status=_first(entry, "status_text", "nav_status")),
        source=source,
    )


class AISCatcherHTTPSource(PollingSource):
    """Polls an AIS-catcher instance's JSON endpoint."""

    def __init__(self, cfg: dict, region=None):
        super().__init__(cfg, TrackKind.SHIP, region)
        self.url = str(cfg.get("url", "http://127.0.0.1:8100/api/vessels.json"))
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "spotter-overlay/1.0"

    def poll(self) -> Sequence[TrackReport]:
        response = self.session.get(self.url, timeout=self.timeout_s)
        check_rate_limit(response)
        response.raise_for_status()
        payload = response.json()

        if isinstance(payload, list):
            entries = payload
            now_epoch = time.time()
        elif isinstance(payload, dict):
            entries = (payload.get("values") or payload.get("vessels")
                       or payload.get("ships") or payload.get("data") or [])
            now_epoch = as_float(payload.get("now"), time.time()) or time.time()
            if now_epoch > 1e11:
                now_epoch /= 1000.0
        else:
            raise ValueError(f"unexpected payload type {type(payload).__name__}")

        reports = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            report = parse_aiscatcher_vessel(entry, self.name, now_epoch)
            if report is not None and report.is_valid():
                reports.append(report)
        return reports

    def attribution(self) -> str:
        return str(self.cfg.get("attribution") or "AIS: local receiver")


# ---------------------------------------------------------------------------
# Raw NMEA over UDP
# ---------------------------------------------------------------------------

class NMEAUDPSource(TrackSource):
    """Listens for AIVDM sentences on a UDP port (AIS-catcher ``-u``, rtl-ais...)."""

    def __init__(self, cfg: dict, region=None):
        super().__init__(cfg, TrackKind.SHIP, region)
        self.host = str(cfg.get("bind_host", "0.0.0.0"))
        self.port = int(cfg.get("bind_port", 10110))
        self.decoder = NMEADecoder()
        self.assembler = AISAssembler()
        #: A quiet AIS band is normal, so silence alone must not look like a
        #: failure. Only a genuinely dead socket should trigger failover.
        self.idle_warn_s = float(cfg.get("idle_warn_s", 300.0))

    def run(self, emit: Emit, stop: threading.Event) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        try:
            sock.bind((self.host, self.port))
        except OSError as exc:
            self.health.record_failure(f"bind failed: {exc}")
            log.error("could not bind UDP port", extra={
                "source": self.name, "host": self.host, "port": self.port,
                "error": str(exc)})
            return

        log.info("listening for NMEA", extra={"source": self.name,
                                              "host": self.host, "port": self.port})
        last_data = time.monotonic()
        warned_idle = False
        try:
            while not stop.is_set():
                try:
                    data, _addr = sock.recvfrom(8192)
                except socket.timeout:
                    idle = time.monotonic() - last_data
                    if idle > self.idle_warn_s and not warned_idle:
                        warned_idle = True
                        log.warning("no NMEA received", extra={
                            "source": self.name, "idle_s": round(idle)})
                    continue
                except OSError as exc:
                    self.health.record_failure(str(exc))
                    log.warning("UDP receive error",
                                extra={"source": self.name, "error": str(exc)})
                    if self.health.failed:
                        return
                    continue

                last_data = time.monotonic()
                warned_idle = False
                count = 0
                now = time.monotonic()
                for line in data.decode("ascii", errors="ignore").splitlines():
                    for message in self.decoder.feed_line(line, now=now):
                        for report in self.assembler.feed(message, self.name):
                            if report.is_valid():
                                emit(report)
                                count += 1
                self.health.record_success(count)
        finally:
            sock.close()

    def attribution(self) -> str:
        return str(self.cfg.get("attribution") or "AIS: local receiver")


# ---------------------------------------------------------------------------
# aisstream.io websocket
# ---------------------------------------------------------------------------

AISSTREAM_POSITION_TYPES = {
    "PositionReport", "StandardClassBPositionReport",
    "ExtendedClassBPositionReport",
}
AISSTREAM_STATIC_TYPES = {"ShipStaticData", "StaticDataReport"}


def _parse_aisstream_time(value) -> Optional[datetime]:
    """aisstream stamps look like '2024-05-01 12:34:56.789 +0000 UTC'."""
    if not value:
        return None
    text = str(value).replace(" UTC", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f %z", "%Y-%m-%d %H:%M:%S %z"):
        try:
            return datetime.strptime(text, fmt).astimezone(timezone.utc)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def parse_aisstream_message(payload: dict, source: str) -> Optional[TrackReport]:
    """Map one aisstream.io websocket frame to a report."""
    msg_type = payload.get("MessageType")
    meta = payload.get("MetaData") or {}
    body = (payload.get("Message") or {}).get(msg_type) or {}

    mmsi = _first(meta, "MMSI", "mmsi") or body.get("UserID")
    track_id = _track_id(mmsi)
    if track_id is None:
        return None

    lat = as_float(_first(body, "Latitude", "latitude"))
    lon = as_float(_first(body, "Longitude", "longitude"))
    if lat is None or lon is None:
        lat = as_float(_first(meta, "latitude", "Latitude"))
        lon = as_float(_first(meta, "longitude", "Longitude"))
    if lat is None or lon is None:
        return None

    speed_kn = as_float(_first(body, "Sog", "SOG"))
    if speed_kn is not None and speed_kn >= SOG_SENTINEL_KN:
        speed_kn = None
    course = as_float(_first(body, "Cog", "COG"))
    if course is not None and course >= 360:
        course = None
    heading = as_float(_first(body, "TrueHeading", "Heading"))
    if heading is not None and heading >= 511:
        heading = None

    dimension = body.get("Dimension") or {}
    length = None
    if dimension:
        bow, stern = as_float(dimension.get("A")), as_float(dimension.get("B"))
        if bow is not None and stern is not None:
            length = bow + stern

    return TrackReport(
        track_id=track_id, kind=TrackKind.SHIP,
        timestamp=_parse_aisstream_time(meta.get("time_utc")) or utcnow(),
        lat=lat, lon=lon, alt_m=0.0,
        speed_mps=None if speed_kn is None else speed_kn * KNOTS_TO_MPS,
        course_deg=course, heading_deg=heading,
        labels=_labels_from(
            name=_first(body, "Name", "ShipName") or meta.get("ShipName"),
            callsign=body.get("CallSign"),
            ship_type=_first(body, "Type", "ShipType"),
            destination=body.get("Destination"),
            imo=body.get("ImoNumber"),
            mmsi=mmsi, length_m=length,
            draught_m=body.get("MaximumStaticDraught")),
        source=source,
    )


class AISStreamSource(TrackSource):
    """Subscribes to aisstream.io over a websocket, filtered to a bounding box."""

    def __init__(self, cfg: dict, region=None):
        super().__init__(cfg, TrackKind.SHIP, region)
        self.url = str(cfg.get("url", "wss://stream.aisstream.io/v0/stream"))
        import os
        self.api_key = str(cfg.get("api_key") or os.environ.get("AISSTREAM_API_KEY", ""))
        self.reconnect_s = float(cfg.get("reconnect_s", 5.0))
        self.assembler = AISAssembler()
        self._last_position: dict[str, TrackReport] = {}

    def _subscription(self) -> dict:
        region = self.region or {}
        box = [[
            [float(region.get("lat_min", -90)), float(region.get("lon_min", -180))],
            [float(region.get("lat_max", 90)), float(region.get("lon_max", 180))],
        ]]
        return {"APIKey": self.api_key, "BoundingBoxes": box}

    def run(self, emit: Emit, stop: threading.Event) -> None:
        if not self.api_key:
            self.health.record_failure("no API key configured")
            log.error("aisstream needs an API key; set tracks.ais.sources[].api_key "
                      "or $AISSTREAM_API_KEY", extra={"source": self.name})
            # Fail fast and hard so the group demotes us rather than spinning.
            self.health.consecutive_failures = self.health.failover_after
            return

        try:
            from websockets.sync.client import connect
        except ImportError:
            self.health.record_failure("websockets package too old for sync client")
            log.error("aisstream needs websockets>=12", extra={"source": self.name})
            self.health.consecutive_failures = self.health.failover_after
            return

        while not stop.is_set():
            try:
                with connect(self.url, open_timeout=20, close_timeout=5) as socket_:
                    socket_.send(json.dumps(self._subscription()))
                    log.info("aisstream subscribed",
                             extra={"source": self.name, "box": self._subscription()
                                    ["BoundingBoxes"]})
                    while not stop.is_set():
                        try:
                            raw = socket_.recv(timeout=30)
                        except TimeoutError:
                            continue
                        self._handle(raw, emit)
            except Exception as exc:
                self.health.record_failure(f"{type(exc).__name__}: {exc}")
                log.warning("aisstream connection failed", extra={
                    "source": self.name, "error": str(exc),
                    "consecutive": self.health.consecutive_failures})
                if self.health.failed:
                    return
                stop.wait(self.reconnect_s)

    def _handle(self, raw, emit: Emit) -> None:
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        if payload.get("error") or payload.get("Error"):
            raise RuntimeError(str(payload.get("error") or payload.get("Error")))

        msg_type = payload.get("MessageType")
        if msg_type not in AISSTREAM_POSITION_TYPES | AISSTREAM_STATIC_TYPES:
            return

        report = parse_aisstream_message(payload, self.name)
        if report is None or not report.is_valid():
            return

        # Static frames carry no position of their own; aisstream fills in the
        # vessel's last known one, so merge rather than treating it as movement.
        if msg_type in AISSTREAM_STATIC_TYPES:
            previous = self._last_position.get(report.track_id)
            if previous is None:
                return
            previous.labels.update(report.labels)
            report = TrackReport(
                track_id=previous.track_id, kind=previous.kind,
                timestamp=previous.timestamp, lat=previous.lat, lon=previous.lon,
                alt_m=0.0, speed_mps=previous.speed_mps,
                course_deg=previous.course_deg, heading_deg=previous.heading_deg,
                labels=dict(previous.labels), source=self.name)
        else:
            self._last_position[report.track_id] = report

        emit(report)
        self.health.record_success(1)

    def attribution(self) -> str:
        return str(self.cfg.get("attribution") or "AIS: aisstream.io")


def build_ais_source(cfg: dict, region=None) -> Optional[TrackSource]:
    """Construct one AIS source from its config entry."""
    kind = str(cfg.get("type", "")).lower()
    if kind in ("aiscatcher_http", "ais_catcher", "aiscatcher", "http"):
        return AISCatcherHTTPSource(cfg, region)
    if kind in ("nmea_udp", "udp", "aivdm_udp"):
        return NMEAUDPSource(cfg, region)
    if kind in ("aisstream", "aisstream.io", "websocket"):
        return AISStreamSource(cfg, region)

    log.warning("unknown AIS source type",
                extra={"type": kind, "name": cfg.get("name")})
    return None
