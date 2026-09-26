"""ADS-B sources.

All three supported feeds speak the readsb/dump1090 ``aircraft.json`` dialect,
because the public aggregators are themselves readsb-derived. So there is one
field mapper and three thin wrappers that differ only in how the request is
formed and where the aircraft array lives.

Units in that dialect are aviation units -- feet, knots, feet per minute -- and
are converted to SI here so nothing downstream has to think about it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Sequence

import requests

from ..logging_setup import get_logger
from ..util import as_float, clean_str
from .model import FEET_TO_M, FPM_TO_MPS, KNOTS_TO_MPS, TrackKind, TrackReport
from .sources import PollingSource, check_rate_limit

log = get_logger(__name__)

#: dbFlags bit 0 marks military airframes in the adsb.lol / airplanes.live feeds.
DBFLAG_MILITARY = 1


def _timestamp(now_epoch: float, seen_pos: Optional[float]) -> datetime:
    """When the position was actually measured, not when we fetched it."""
    age = seen_pos if seen_pos is not None else 0.0
    return datetime.fromtimestamp(now_epoch - age, tz=timezone.utc)


def _normalise_now(value) -> float:
    """``now`` is seconds in readsb and milliseconds in the public APIs."""
    now = as_float(value)
    if now is None:
        import time
        return time.time()
    # Anything past ~1973 in milliseconds is absurd as a seconds value.
    return now / 1000.0 if now > 1e11 else now


def parse_aircraft(entry: dict, now_epoch: float, source: str) -> Optional[TrackReport]:
    """Map one readsb-dialect aircraft record to a :class:`TrackReport`."""
    lat = as_float(entry.get("lat"))
    lon = as_float(entry.get("lon"))
    if lat is None or lon is None:
        return None  # no position yet: the airframe is heard but not located

    icao = clean_str(entry.get("hex")) or clean_str(entry.get("icao"))
    if not icao:
        return None
    icao = icao.lower().lstrip("~")

    # alt_baro is the string "ground" for aircraft on the airport surface.
    raw_alt = entry.get("alt_geom")
    if raw_alt is None:
        raw_alt = entry.get("alt_baro")
    on_ground = isinstance(raw_alt, str) and raw_alt.strip().lower() == "ground"
    alt_ft = 0.0 if on_ground else as_float(raw_alt)
    alt_m = None if alt_ft is None else alt_ft * FEET_TO_M

    speed_kt = as_float(entry.get("gs"))
    if speed_kt is None:
        speed_kt = as_float(entry.get("tas")) or as_float(entry.get("ias"))

    vertical_fpm = as_float(entry.get("geom_rate"))
    if vertical_fpm is None:
        vertical_fpm = as_float(entry.get("baro_rate"))

    db_flags = as_float(entry.get("dbFlags")) or 0.0
    labels = {
        "icao": icao.upper(),
        "callsign": clean_str(entry.get("flight")),
        "registration": clean_str(entry.get("r")),
        "type": clean_str(entry.get("t")),
        "description": clean_str(entry.get("desc")),
        "category": clean_str(entry.get("category")),
        "squawk": clean_str(entry.get("squawk")),
        "operator": clean_str(entry.get("ownOp")),
        "on_ground": on_ground,
        "military": bool(int(db_flags) & DBFLAG_MILITARY),
        "mlat": bool(entry.get("mlat")),
        "emergency": clean_str(entry.get("emergency")) not in (None, "none"),
    }

    return TrackReport(
        track_id=f"icao:{icao}",
        kind=TrackKind.AIRCRAFT,
        timestamp=_timestamp(now_epoch, as_float(entry.get("seen_pos"))),
        lat=lat, lon=lon, alt_m=alt_m,
        speed_mps=None if speed_kt is None else speed_kt * KNOTS_TO_MPS,
        course_deg=as_float(entry.get("track")),
        heading_deg=(as_float(entry.get("true_heading"))
                     or as_float(entry.get("mag_heading"))),
        vertical_rate_mps=(None if vertical_fpm is None
                           else vertical_fpm * FPM_TO_MPS),
        labels={k: v for k, v in labels.items() if v not in (None, "")},
        source=source,
    )


def parse_aircraft_payload(payload: dict, source: str) -> list[TrackReport]:
    """Parse a whole aircraft.json-style document."""
    now_epoch = _normalise_now(payload.get("now"))
    entries = payload.get("aircraft")
    if entries is None:
        entries = payload.get("ac") or []
    reports = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        report = parse_aircraft(entry, now_epoch, source)
        if report is not None and report.is_valid():
            reports.append(report)
    return reports


class _HttpAircraftSource(PollingSource):
    """Shared plumbing for the three HTTP ADS-B feeds."""

    def __init__(self, cfg: dict, region=None):
        super().__init__(cfg, TrackKind.AIRCRAFT, region)
        self.session = requests.Session()
        self.session.headers["User-Agent"] = cfg.get(
            "user_agent", "spotter-overlay/1.0 (+AIS/ADS-B camera overlay)")
        api_key = cfg.get("api_key")
        if api_key:
            self.session.headers["auth"] = str(api_key)
            self.session.headers["X-API-Key"] = str(api_key)

    def _url(self) -> str:
        raise NotImplementedError

    def poll(self) -> Sequence[TrackReport]:
        response = self.session.get(self._url(), timeout=self.timeout_s)
        check_rate_limit(response)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"unexpected payload type {type(payload).__name__}")
        if payload.get("msg") and payload.get("msg") != "No error":
            raise ValueError(f"API error: {payload['msg']}")
        return parse_aircraft_payload(payload, self.name)


class ReadsbSource(_HttpAircraftSource):
    """A local readsb / dump1090-fa ``aircraft.json``."""

    def __init__(self, cfg: dict, region=None):
        super().__init__(cfg, region)
        self.url = str(cfg.get("url", "http://127.0.0.1:8080/data/aircraft.json"))

    def _url(self) -> str:
        return self.url

    def attribution(self) -> str:
        return str(self.cfg.get("attribution") or "ADS-B: local receiver")


class PointApiSource(_HttpAircraftSource):
    """adsb.lol / airplanes.live, which serve a radius around a point.

    Measured behaviour of adsb.lol (there is nothing about this in its docs):

    * It returns **no rate-limit headers at all** -- no ``X-RateLimit-*``, and
      no ``Retry-After`` on a 429. So there is nothing to obey; the backoff in
      :class:`PollingSource` is all we have.
    * The 429 comes from **nginx**, not the application: the body is an HTML
      error page, not JSON. Do not try to parse it.
    * The limit behaves like roughly one request per second with a small burst
      -- two back-to-back requests pass, a third within half a second does not.

    A steady poll of a few seconds is therefore fine; what actually trips it is
    *bursts*, which is why source failover is careful not to restart this
    source repeatedly.
    """

    def __init__(self, cfg: dict, region=None, attribution: str = "ADS-B"):
        super().__init__(cfg, region)
        self.template = str(cfg.get("url", ""))
        self.radius_nm = float(cfg.get("radius_nm", 60))
        self.lat = float(cfg.get("lat", 0.0))
        self.lon = float(cfg.get("lon", 0.0))
        self._attribution = attribution

    def _url(self) -> str:
        return self.template.format(lat=self.lat, lon=self.lon,
                                    radius_nm=int(round(self.radius_nm)),
                                    radius=int(round(self.radius_nm)),
                                    dist=int(round(self.radius_nm)))

    def attribution(self) -> str:
        return str(self.cfg.get("attribution") or self._attribution)


def build_adsb_source(cfg: dict, camera_lat: float, camera_lon: float,
                      region=None) -> Optional[PollingSource]:
    """Construct one ADS-B source from its config entry."""
    kind = str(cfg.get("type", "")).lower()
    merged = dict(cfg)
    merged.setdefault("lat", camera_lat)
    merged.setdefault("lon", camera_lon)

    if kind in ("readsb", "dump1090", "tar1090", "local"):
        return ReadsbSource(merged, region)
    if kind in ("adsb_fi", "adsb.fi", "opendata_adsb_fi"):
        merged.setdefault(
            "url", "https://opendata.adsb.fi/api/v2/lat/{lat}/lon/{lon}/dist/{radius_nm}")
        return PointApiSource(merged, region, attribution="ADS-B: adsb.fi")
    if kind in ("adsb_lol", "adsb.lol"):
        merged.setdefault("url", "https://api.adsb.lol/v2/point/{lat}/{lon}/{radius_nm}")
        return PointApiSource(merged, region, attribution="ADS-B: adsb.lol")
    if kind in ("airplanes_live", "airplanes.live"):
        merged.setdefault("url",
                          "https://api.airplanes.live/v2/point/{lat}/{lon}/{radius_nm}")
        return PointApiSource(merged, region, attribution="ADS-B: airplanes.live")

    log.warning("unknown ADS-B source type", extra={"type": kind,
                                                    "name": cfg.get("name")})
    return None
