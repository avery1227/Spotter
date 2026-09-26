"""Track source framework: polling, streaming, and priority failover.

Sources are declared in priority order in ``config.yaml``. Exactly one runs at a
time. When the active source fails repeatedly we demote to the next; a
background recheck periodically walks back up the list so a local receiver that
came back online takes over again from the public API fallback.

Running only one at a time is deliberate: the public endpoints are a courtesy,
and polling all of them in parallel just to throw away the results would be rude
and would burn rate limit we might need.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence

from ..logging_setup import get_logger
from ..util import Backoff, SourceHealth
from .model import TrackKind, TrackReport

log = get_logger(__name__)

Emit = Callable[[TrackReport], None]


class RateLimited(Exception):
    """The endpoint asked us to slow down (HTTP 429).

    Carried separately from ordinary failures so we can honour ``Retry-After``
    instead of retrying on our own schedule and digging the hole deeper.
    """

    def __init__(self, message: str, retry_after_s: Optional[float] = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


def check_rate_limit(response) -> None:
    """Raise :class:`RateLimited` for an HTTP 429, honouring ``Retry-After``."""
    if response.status_code != 429:
        return
    retry_after = None
    header = response.headers.get("Retry-After")
    if header:
        try:
            retry_after = float(header)
        except ValueError:
            retry_after = None   # HTTP-date form; fall back to our own backoff
    raise RateLimited(f"HTTP 429 from {response.url.split('?')[0]}", retry_after)


class TrackSource(ABC):
    """Base for anything that produces :class:`TrackReport` objects."""

    def __init__(self, cfg: dict, kind: TrackKind, region=None):
        self.cfg = dict(cfg)
        self.kind = kind
        self.region = region
        self.name = str(cfg.get("name") or cfg.get("type") or "source")
        self.type = str(cfg.get("type") or "")
        self.health = SourceHealth()
        #: How long a track of this kind survives without an update. Set by the
        #: manager; used to warn when the poll rate cannot keep up.
        self.stale_timeout_s: Optional[float] = None

    @abstractmethod
    def run(self, emit: Emit, stop: threading.Event) -> None:
        """Produce reports until ``stop`` is set. Must return when it is."""

    def describe(self) -> str:
        return f"{self.name} ({self.type})"

    # Shown in the on-screen attribution line.
    def attribution(self) -> str:
        return str(self.cfg.get("attribution") or self.name)


class PollingSource(TrackSource):
    """A source that is scraped on an interval."""

    def __init__(self, cfg: dict, kind: TrackKind, region=None):
        super().__init__(cfg, kind, region)
        self.poll_s = float(cfg.get("poll_s", 2.0))
        self.timeout_s = float(cfg.get("timeout_s", 5.0))
        self.max_poll_s = float(cfg.get("max_poll_s", 60.0))
        #: Effective interval. A public endpoint's rate limit is not something
        #: we can know in advance -- it varies by endpoint, by time of day and
        #: by who else shares our IP -- so back off when throttled and creep
        #: back down when forgiven, rather than hammering a fixed interval that
        #: happens to be slightly too fast.
        self._interval = self.poll_s
        #: Survives across restarts, because the source object does. Failover
        #: recreates the worker thread, and polling immediately on every
        #: restart is exactly the burst that gets a throttled endpoint to
        #: refuse us.
        self._last_request_at = 0.0
        self._warned_too_slow = False

    @abstractmethod
    def poll(self) -> Sequence[TrackReport]:
        """Fetch once. Raise on failure; return a (possibly empty) sequence."""

    def run(self, emit: Emit, stop: threading.Event) -> None:
        backoff = Backoff(initial_s=max(1.0, self.poll_s), max_s=30.0)

        # Pick up where the last stint left off rather than firing immediately.
        since = time.monotonic() - self._last_request_at
        if self._last_request_at and since < self._interval:
            if stop.wait(self._interval - since):
                return

        while not stop.is_set():
            started = time.monotonic()
            self._last_request_at = started
            try:
                reports = self.poll()
            except RateLimited as exc:
                # Being throttled is not the same as being broken. Wait out the
                # window the server asked for, and slow the steady-state poll so
                # we stop provoking it.
                self.health.record_failure(f"rate limited: {exc}")
                self._interval = min(self._interval * 2.0, self.max_poll_s)
                delay = exc.retry_after_s
                if delay is None:
                    delay = backoff.next_delay()
                delay = min(max(delay, self._interval), 120.0)
                log.warning("source rate limited; backing off", extra={
                    "source": self.name, "retry_after_s": round(delay, 1),
                    "poll_s": round(self._interval, 1),
                    "consecutive": self.health.consecutive_failures})
                if self.health.failed:
                    return
                stop.wait(delay)
                continue
            except Exception as exc:
                self.health.record_failure(f"{type(exc).__name__}: {exc}")
                status = getattr(getattr(exc, "response", None), "status_code", None)

                # A 4xx means we are asking the wrong thing of the wrong server:
                # a bad URL, a missing key, the wrong service on that port.
                # Retrying it five times before demoting just wastes requests
                # and, for the fallback we demote *to*, can trip its rate limit.
                if status is not None and 400 <= status < 500 and status not in (408, 429):
                    log.error(
                        "source looks misconfigured; giving up on it for now",
                        extra={"source": self.name, "status": status,
                               "error": str(exc)[:160],
                               "hint": "check this source's url in config.yaml"})
                    self.health.consecutive_failures = max(
                        self.health.consecutive_failures, self.health.failover_after)
                    return

                log.warning("source poll failed", extra={
                    "source": self.name, "error": str(exc),
                    "consecutive": self.health.consecutive_failures})
                if self.health.failed:
                    return  # let the group demote us
                backoff.sleep(stop)
                continue

            backoff.reset()
            count = 0
            for report in reports:
                emit(report)
                count += 1
            self.health.record_success(count)

            # Creep back toward the configured rate after a clean poll, so a
            # one-off throttle does not permanently slow us down.
            if self._interval > self.poll_s:
                self._interval = max(self.poll_s, self._interval * 0.8)
            self._warn_if_too_slow()

            log.debug("source poll", extra={"source": self.name, "reports": count,
                                            "poll_s": round(self._interval, 1)})

            elapsed = time.monotonic() - started
            stop.wait(max(0.0, self._interval - elapsed))


    def _warn_if_too_slow(self) -> None:
        """Say so when the poll rate cannot keep tracks alive.

        Tracks are dropped once their newest report is older than the stale
        timeout. If we are polling slower than that, every target expires just
        before its next update and the labels flicker instead of holding --
        which looks like a rendering bug rather than a feed problem.
        """
        if self.stale_timeout_s is None or self._warned_too_slow:
            return
        if self._interval < self.stale_timeout_s * 0.6:
            return
        self._warned_too_slow = True
        log.error(
            "polling slower than tracks survive; they will flicker or vanish",
            extra={"source": self.name,
                   "poll_s": round(self._interval, 1),
                   "stale_timeout_s": self.stale_timeout_s,
                   "hint": "this endpoint is throttling us -- prefer a local "
                           "receiver, or raise tracks.stale_timeout_s"})


class SourceGroup:
    """Runs one source from a priority list, failing over and recovering."""

    def __init__(self, name: str, sources: Sequence[TrackSource], store,
                 failover_after: int = 5, recover_after: int = 3,
                 recheck_primary_s: float = 120.0,
                 all_failed_cooldown_s: float = 30.0):
        self.name = name
        self.sources = list(sources)
        self.store = store
        self.recheck_primary_s = float(recheck_primary_s)
        self.all_failed_cooldown_s = float(all_failed_cooldown_s)
        for source in self.sources:
            source.health.failover_after = int(failover_after)
            source.health.recover_after = int(recover_after)

        self.active_index = 0
        self._stop = threading.Event()
        self._worker_stop: Optional[threading.Event] = None
        self._worker: Optional[threading.Thread] = None
        self._supervisor: Optional[threading.Thread] = None
        self._last_recheck = time.monotonic()
        #: Switches made without a single report arriving. Once this reaches the
        #: number of sources we have tried them all and should stop hammering.
        self._switches_since_success = 0

    @property
    def active(self) -> Optional[TrackSource]:
        if not self.sources:
            return None
        return self.sources[self.active_index % len(self.sources)]

    def start(self) -> "SourceGroup":
        if not self.sources:
            log.info("no sources configured", extra={"group": self.name})
            return self
        self._supervisor = threading.Thread(target=self._supervise,
                                            name=f"{self.name}-supervisor",
                                            daemon=True)
        self._supervisor.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._stop_worker()
        if self._supervisor is not None:
            self._supervisor.join(timeout=10.0)

    # -- internals ----------------------------------------------------------
    def _emit(self, report: TrackReport) -> None:
        self._switches_since_success = 0
        self.store.add_report(report)

    def _start_worker(self) -> None:
        source = self.active
        if source is None:
            return
        self._worker_stop = threading.Event()
        stop_event = self._worker_stop

        def target() -> None:
            log.info("track source active", extra={
                "group": self.name, "source": source.describe(),
                "priority": self.active_index})
            try:
                source.run(self._emit, stop_event)
            except Exception:
                log.exception("track source crashed",
                              extra={"group": self.name, "source": source.name})
                source.health.record_failure("crashed")
            finally:
                log.info("track source stopped",
                         extra={"group": self.name, "source": source.name})

        self._worker = threading.Thread(target=target, name=f"{self.name}-{source.name}",
                                        daemon=True)
        self._worker.start()

    def _stop_worker(self) -> None:
        if self._worker_stop is not None:
            self._worker_stop.set()
        if self._worker is not None:
            self._worker.join(timeout=10.0)
        self._worker = None
        self._worker_stop = None

    def _switch_to(self, index: int, reason: str) -> None:
        self._stop_worker()
        previous = self.active_index

        # If we have been all the way round the list without a single report,
        # every source is down. Cycling them at poll speed just hammers public
        # APIs (and is how you earn an HTTP 429), so wait before trying again.
        self._switches_since_success += 1
        if (self._switches_since_success >= len(self.sources)
                and self.all_failed_cooldown_s > 0):
            self._switches_since_success = 0
            log.warning("all track sources unavailable; cooling down", extra={
                "group": self.name,
                "sources": [s.name for s in self.sources],
                "cooldown_s": self.all_failed_cooldown_s})
            if self._stop.wait(self.all_failed_cooldown_s):
                return

        self.active_index = index % len(self.sources)
        # A demoted source starts its next stint with a clean slate, otherwise
        # old failures would demote it again on its first hiccup.
        self.sources[self.active_index].health = SourceHealth(
            self.sources[self.active_index].health.failover_after,
            self.sources[self.active_index].health.recover_after)
        log.info("switching track source", extra={
            "group": self.name, "from": self.sources[previous].name,
            "to": self.active.name, "reason": reason})
        self._start_worker()

    def _supervise(self) -> None:
        self._start_worker()
        while not self._stop.wait(1.0):
            # This thread is the only thing keeping failover alive, so nothing
            # it does may be allowed to kill it. An unhandled exception here
            # would silently strand the group on a dead source for the rest of
            # the process's life.
            try:
                self._supervise_once()
            except Exception:  # pragma: no cover - defensive
                log.exception("source supervisor iteration failed",
                              extra={"group": self.name})
                self._stop.wait(5.0)
        log.info("source supervisor stopped", extra={"group": self.name})

    def _supervise_once(self) -> None:
        source = self.active
        if source is None:
            return

        worker_dead = self._worker is None or not self._worker.is_alive()
        if worker_dead or source.health.failed:
            if len(self.sources) == 1:
                # Nothing to fail over to; restart the only source we have.
                self._switch_to(self.active_index, "restart (sole source)")
            else:
                self._switch_to(self.active_index + 1,
                                "exhausted" if source.health.failed else "exited")
            self._last_recheck = time.monotonic()
            return

        # Periodically walk back up toward the preferred source.
        if (self.active_index != 0
                and time.monotonic() - self._last_recheck > self.recheck_primary_s):
            self._last_recheck = time.monotonic()
            self._switch_to(0, "rechecking preferred source")

    def status(self) -> dict:
        source = self.active
        return {
            "group": self.name,
            "active": source.name if source else None,
            "priority": self.active_index,
            "health": source.health.as_dict() if source else {},
        }

    def attribution(self) -> Optional[str]:
        """Attribution for the on-screen credit line.

        Only a source that has actually delivered reports is credited. The
        active source is not proof of anything: the AIS UDP listener, for one,
        never "fails" -- it just sits there hearing nothing -- and crediting it
        would tell viewers the overlay has AIS data when it does not.
        """
        source = self.active
        if source is None or source.health.total_reports <= 0:
            return None
        return source.attribution()
