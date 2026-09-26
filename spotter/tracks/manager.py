"""Builds and supervises every track source described in the config."""

from __future__ import annotations

import threading
import time
from typing import Optional

from ..logging_setup import get_logger
from ..util import utcnow
from .adsb import build_adsb_source
from .ais import build_ais_source
from .sources import SourceGroup
from .store import TrackStore

log = get_logger(__name__)

#: How often to sweep stale tracks out of the store.
PRUNE_INTERVAL_S = 5.0


class TrackManager:
    """Owns the store, the source groups, and the pruning thread."""

    def __init__(self, cfg, store: Optional[TrackStore] = None):
        self.cfg = cfg
        self.store = store if store is not None else TrackStore(cfg)
        self.groups: list[SourceGroup] = []
        self._stop = threading.Event()
        self._pruner: Optional[threading.Thread] = None

        region = cfg.sub("region").raw() if cfg.get("region") else {}
        camera_lat = float(cfg.get("camera.lat", 0.0))
        camera_lon = float(cfg.get("camera.lon", 0.0))

        if cfg.get("tracks.adsb.enabled", True):
            sources = []
            for entry in (cfg.get("tracks.adsb.sources") or []):
                entry = dict(entry)
                source = build_adsb_source(entry, camera_lat, camera_lon, region)
                if source is not None:
                    source.stale_timeout_s = float(
                        cfg.get("tracks.stale_timeout_s.aircraft", 30.0))
                    sources.append(source)
            if sources:
                self.groups.append(SourceGroup(
                    "adsb", sources, self.store,
                    failover_after=int(cfg.get("tracks.adsb.failover_after_failures", 5)),
                    recover_after=int(cfg.get("tracks.adsb.recover_after_successes", 3)),
                    recheck_primary_s=float(
                        cfg.get("tracks.adsb.recheck_primary_s", 120.0)),
                    all_failed_cooldown_s=float(
                        cfg.get("tracks.adsb.all_failed_cooldown_s", 30.0))))

        if cfg.get("tracks.ais.enabled", True):
            sources = []
            for entry in (cfg.get("tracks.ais.sources") or []):
                entry = dict(entry)
                source = build_ais_source(entry, region)
                if source is not None:
                    source.stale_timeout_s = float(
                        cfg.get("tracks.stale_timeout_s.ship", 600.0))
                    sources.append(source)
            if sources:
                self.groups.append(SourceGroup(
                    "ais", sources, self.store,
                    failover_after=int(cfg.get("tracks.ais.failover_after_failures", 5)),
                    recover_after=int(cfg.get("tracks.ais.recover_after_successes", 3)),
                    recheck_primary_s=float(
                        cfg.get("tracks.ais.recheck_primary_s", 120.0)),
                    all_failed_cooldown_s=float(
                        cfg.get("tracks.ais.all_failed_cooldown_s", 30.0))))

        log.info("track manager configured", extra={
            "groups": [{"name": g.name, "sources": [s.name for s in g.sources]}
                       for g in self.groups]})

    def start(self) -> "TrackManager":
        for group in self.groups:
            group.start()
        self._pruner = threading.Thread(target=self._prune_loop, name="track-pruner",
                                        daemon=True)
        self._pruner.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        for group in self.groups:
            group.stop()
        if self._pruner is not None:
            self._pruner.join(timeout=5.0)

    def _prune_loop(self) -> None:
        while not self._stop.wait(PRUNE_INTERVAL_S):
            try:
                removed = self.store.prune(utcnow())
                if removed:
                    log.debug("pruned stale tracks", extra={
                        "removed": removed, **self.store.counts()})
            except Exception:  # pragma: no cover - must never kill the thread
                log.exception("prune failed")

    def status(self) -> dict:
        return {
            "tracks": self.store.counts(),
            "groups": [group.status() for group in self.groups],
        }

    def attributions(self) -> list[str]:
        """Attribution strings for the currently active sources."""
        out = []
        for group in self.groups:
            text = group.attribution()
            if text and text not in out:
                out.append(text)
        return out

    def wait_for_data(self, timeout_s: float = 20.0, minimum: int = 1) -> bool:
        """Block until at least ``minimum`` tracks are known, or time out."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if len(self.store) >= minimum:
                return True
            time.sleep(0.25)
        return False
