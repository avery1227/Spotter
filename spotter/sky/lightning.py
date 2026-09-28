"""Lightning strikes from the Blitzortung.org community network.

Blitzortung publishes every located strike worldwide over a websocket, about
15 seconds after it happens (the network waits for enough stations to report
before it solves for a location). The pipeline renders frames 20-30 seconds
behind real time, so a strike usually arrives *before* the frame that shows
it, and its flash can be drawn on the frame of the actual strike, at its real
bearing and distance.

A strike that arrives after its frame has already gone out is still drawn: it
flashes on the first frame after it arrives, and its label says how long ago
it really happened. ``late`` in the status counts these; if it climbs, raise
``stream.encoder_delay_s`` to give the feed more headroom.

Blitzortung's data is free for private, non-commercial use, with attribution.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from ..geodesy import geodetic_to_enu
from ..logging_setup import get_logger
from ..projection import ProjectedTarget
from ..tracks.model import Position, TrackKind
from ..tracks.store import Track, TrackState

log = get_logger(__name__)

DEFAULT_HOSTS = ("wss://ws1.blitzortung.org/", "wss://ws7.blitzortung.org/",
                 "wss://ws8.blitzortung.org/")

EARTH_RADIUS_KM = 6371.0088


def lzw_decode(data: str) -> str:
    """Undo the LZW-style compression Blitzortung applies to each message.

    A direct port of the decoder in Blitzortung's own map page: codes below
    256 are literal characters, anything above indexes a dictionary built as
    the message is read.
    """
    if not data:
        return ""
    dictionary: dict[int, str] = {}
    current = data[0]
    previous = current
    out = [current]
    next_code = 256
    for char in data[1:]:
        code = ord(char)
        if code < 256:
            entry = char
        elif code in dictionary:
            entry = dictionary[code]
        else:
            entry = previous + current
        out.append(entry)
        current = entry[0]
        dictionary[next_code] = previous + current
        next_code += 1
        previous = entry
    return "".join(out)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


@dataclass
class Strike:
    """One located strike."""

    time: float                 # epoch seconds, when it happened
    lat: float
    lon: float
    received: float             # epoch seconds, when we heard about it
    distance_km: float
    #: Frame time it was first drawn at; drives the flash animation.
    first_frame: Optional[float] = None

    @property
    def key(self) -> str:
        return f"ltg:{self.time:.3f}:{self.lat:.3f}:{self.lon:.3f}"


def parse_strike(message: str, camera_lat: float, camera_lon: float,
                 max_range_km: float, received: float) -> Optional[Strike]:
    """Turn one raw websocket message into a Strike, if it is near enough."""
    try:
        data = json.loads(lzw_decode(message))
        stamp = float(data["time"]) / 1e9
        lat = float(data["lat"])
        lon = float(data["lon"])
    except (ValueError, KeyError, TypeError):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    distance = haversine_km(camera_lat, camera_lon, lat, lon)
    if distance > max_range_km:
        return None
    return Strike(time=stamp, lat=lat, lon=lon, received=received,
                  distance_km=distance)


@dataclass
class BoltEffect:
    """A flash to draw: the channel from the ground point up into the cloud."""

    ground_u: float
    ground_v: float
    top_u: float
    top_v: float
    intensity: float            # 0..1
    seed: int


class LightningLayer:
    """Listens to Blitzortung and turns recent strikes into flashes and labels."""

    def __init__(self, cfg):
        node = cfg.sub("sky.lightning")
        self.enabled = bool(node.get("enabled", True))
        self.hosts = list(node.get("hosts", DEFAULT_HOSTS) or DEFAULT_HOSTS)
        self.max_range_km = float(node.get("max_range_km", 300.0))
        self.flash_s = float(node.get("flash_s", 1.2))
        self.label_s = float(node.get("label_s", 20.0))
        self.max_labels = int(node.get("max_labels", 3))
        self.bolt_top_m = float(node.get("bolt_top_m", 6000.0))
        self.anchor_alt_m = float(node.get("anchor_alt_m", 1500.0))
        self.margin_px = float(cfg.get("projection.frame_margin_px", 80))
        self.camera_lat = float(cfg.get("camera.lat", 0.0))
        self.camera_lon = float(cfg.get("camera.lon", 0.0))

        self._lock = threading.Lock()
        self._strikes: dict[str, Strike] = {}
        self._delays: deque[float] = deque(maxlen=200)
        self._tracks: dict[str, Track] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.connected = False
        self.host: Optional[str] = None
        self.messages = 0
        self.strikes_in_range = 0
        self.late = 0
        self.in_view = 0
        self.last_error: Optional[str] = None

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "LightningLayer":
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._run, name="lightning",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        try:
            asyncio.run(self._listen_forever())
        except Exception:  # pragma: no cover - must never take the pipeline down
            log.exception("lightning listener crashed")

    async def _listen_forever(self) -> None:
        import websockets

        backoff = 5.0
        attempt = 0
        while not self._stop.is_set():
            host = self.hosts[attempt % len(self.hosts)]
            attempt += 1
            try:
                async with websockets.connect(host, open_timeout=15,
                                              ping_interval=30,
                                              max_size=2 ** 20) as ws:
                    await ws.send(json.dumps({"a": 111}))
                    self.connected, self.host, self.last_error = True, host, None
                    backoff = 5.0
                    log.info("lightning feed connected", extra={"host": host})
                    while not self._stop.is_set():
                        try:
                            message = await asyncio.wait_for(ws.recv(), 1.0)
                        except asyncio.TimeoutError:
                            continue
                        if isinstance(message, str):
                            self._on_message(message)
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                if self.connected:
                    log.warning("lightning feed dropped", extra={
                        "host": host, "error": self.last_error})
                else:
                    log.warning("lightning feed connect failed", extra={
                        "host": host, "error": self.last_error})
            finally:
                self.connected = False
            if self._stop.is_set():
                return
            await asyncio.sleep(backoff * (0.75 + random.random() * 0.5))
            backoff = min(backoff * 2.0, 300.0)

    def _on_message(self, message: str) -> None:
        self.messages += 1
        now = time.time()
        strike = parse_strike(message, self.camera_lat, self.camera_lon,
                              self.max_range_km, now)
        if strike is None:
            return
        self.add_strike(strike)

    def add_strike(self, strike: Strike) -> None:
        with self._lock:
            # The network re-publishes a strike when its solution improves.
            if strike.key in self._strikes:
                return
            for other in self._strikes.values():
                if (abs(other.time - strike.time) < 0.002
                        and abs(other.lat - strike.lat) < 0.02
                        and abs(other.lon - strike.lon) < 0.02):
                    return
            self._strikes[strike.key] = strike
            self._delays.append(strike.received - strike.time)
            self.strikes_in_range += 1
            # Anything older than the label window can never be drawn again.
            horizon = strike.received - self.label_s - 120.0
            for key in [k for k, s in self._strikes.items() if s.time < horizon]:
                del self._strikes[key]
                self._tracks.pop(key, None)

    # -- projection ---------------------------------------------------------
    def project(self, when: datetime,
                model) -> tuple[list[ProjectedTarget], list[BoltEffect]]:
        if not self.enabled or model is None:
            self.in_view = 0
            return [], []

        frame_t = when.timestamp()
        with self._lock:
            recent = [s for s in self._strikes.values()
                      if 0.0 <= frame_t - s.time <= self.label_s]
        if not recent:
            self.in_view = 0
            return [], []

        for strike in recent:
            if strike.first_frame is None or strike.first_frame > frame_t:
                if strike.first_frame is None and frame_t - strike.time > self.flash_s:
                    self.late += 1
                strike.first_frame = frame_t

        lat = np.array([s.lat for s in recent])
        lon = np.array([s.lon for s in recent])
        count = len(recent)
        points = np.concatenate([
            geodetic_to_enu(lat, lon, np.zeros(count), model.ref_lat, model.ref_lon, 0.0),
            geodetic_to_enu(lat, lon, np.full(count, self.bolt_top_m),
                            model.ref_lat, model.ref_lon, 0.0),
            geodetic_to_enu(lat, lon, np.full(count, self.anchor_alt_m),
                            model.ref_lat, model.ref_lon, 0.0),
        ])
        uv, in_front, _ = model.project_enu(points, refract=True)
        ground, top, anchor = uv[:count], uv[count:2 * count], uv[2 * count:]
        front = in_front[:count] & in_front[count:2 * count] & in_front[2 * count:]
        rel = model.relative_enu(points[2 * count:], refract=True)

        def inside(p: np.ndarray) -> bool:
            return (-self.margin_px <= p[0] <= model.width + self.margin_px
                    and -self.margin_px <= p[1] <= model.height + self.margin_px)

        effects: list[BoltEffect] = []
        visible: list[int] = []
        for i, strike in enumerate(recent):
            if not front[i] or not (inside(anchor[i]) or inside(top[i])):
                continue
            visible.append(i)
            since = frame_t - (strike.first_frame or frame_t)
            intensity = flash_intensity(since, self.flash_s)
            if intensity > 0.02:
                effects.append(BoltEffect(
                    ground_u=float(ground[i, 0]), ground_v=float(ground[i, 1]),
                    top_u=float(top[i, 0]), top_v=float(top[i, 1]),
                    intensity=intensity, seed=hash(strike.key) & 0xFFFFFFFF))

        newest = sorted(visible, key=lambda i: recent[i].time,
                        reverse=True)[:self.max_labels]
        targets: list[ProjectedTarget] = []
        for i in newest:
            strike = recent[i]
            track = self._tracks.get(strike.key)
            if track is None:
                track = Track(strike.key, TrackKind.LIGHTNING)
                self._tracks[strike.key] = track
            track.labels = {"age_s": frame_t - strike.time,
                            "distance_km": strike.distance_km}
            u, v = float(anchor[i, 0]), float(anchor[i, 1])
            bearing = math.degrees(math.atan2(rel[i, 0], rel[i, 1])) % 360.0
            elevation = math.degrees(math.atan2(rel[i, 2], math.hypot(rel[i, 0],
                                                                       rel[i, 1])))
            targets.append(ProjectedTarget(
                state=TrackState(track=track, position=Position(
                    lat=strike.lat, lon=strike.lon, alt_m=self.anchor_alt_m,
                    mode="strike")),
                u=u, v=v,
                range_m=float(np.linalg.norm(rel[i])),
                ground_range_m=strike.distance_km * 1000.0,
                bearing_deg=bearing, elevation_deg=elevation,
                category="lightning",
                on_screen=bool(0 <= u <= model.width and 0 <= v <= model.height),
            ))
        self.in_view = len(visible)
        return targets, effects

    def status(self) -> dict:
        with self._lock:
            delays = list(self._delays)
        return {
            "enabled": self.enabled,
            "connected": self.connected,
            "host": self.host,
            "messages": self.messages,
            "strikes_in_range": self.strikes_in_range,
            "feed_delay_s": round(statistics.median(delays), 1) if delays else None,
            "late": self.late,
            "in_view": self.in_view,
            "error": self.last_error,
        }

    def attribution(self) -> Optional[str]:
        return "Lightning: Blitzortung.org" if self.enabled and self.connected else None


def flash_intensity(since_s: float, flash_s: float) -> float:
    """Brightness of a flash ``since_s`` after it started, 0..1.

    Real lightning flickers: a flash is usually several return strokes tens of
    milliseconds apart along the same channel. Two quick re-brightenings on
    a decaying envelope read as lightning; a single fade reads as a lamp.
    """
    if since_s < 0.0 or since_s >= flash_s:
        return 0.0
    envelope = (1.0 - since_s / flash_s) ** 1.6
    strokes = max(math.exp(-since_s / 0.07),
                  0.85 * math.exp(-max(0.0, since_s - 0.12) / 0.06) * (since_s >= 0.12),
                  0.7 * math.exp(-max(0.0, since_s - 0.30) / 0.08) * (since_s >= 0.30))
    return max(0.0, min(1.0, 0.35 * envelope + 0.65 * strokes))
