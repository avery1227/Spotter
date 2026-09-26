"""Small shared helpers: backoff, clocks, numeric coercion."""

from __future__ import annotations

import math
import random
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional


def utcnow() -> datetime:
    """Timezone-aware UTC now. Used everywhere so naive datetimes never leak in."""
    return datetime.now(timezone.utc)


def to_epoch(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class Backoff:
    """Exponential backoff with jitter.

    ``reset()`` on success; ``sleep()`` (or ``next_delay()``) on failure.
    """

    def __init__(self, initial_s: float = 2.0, max_s: float = 60.0,
                 factor: float = 2.0, jitter: float = 0.25):
        self.initial_s = float(initial_s)
        self.max_s = float(max_s)
        self.factor = float(factor)
        self.jitter = float(jitter)
        self._attempt = 0

    @classmethod
    def from_config(cls, cfg, prefix: str = "") -> "Backoff":
        key = f"{prefix}." if prefix else ""
        return cls(
            initial_s=cfg.get(f"{key}initial_s", 2.0),
            max_s=cfg.get(f"{key}max_s", 60.0),
            factor=cfg.get(f"{key}factor", 2.0),
            jitter=cfg.get(f"{key}jitter", 0.25),
        )

    @property
    def attempt(self) -> int:
        return self._attempt

    def reset(self) -> None:
        self._attempt = 0

    def next_delay(self) -> float:
        delay = min(self.initial_s * (self.factor ** self._attempt), self.max_s)
        self._attempt += 1
        if self.jitter:
            delay *= 1.0 + random.uniform(-self.jitter, self.jitter)
        return max(0.0, delay)

    def sleep(self, stop_event: Optional[threading.Event] = None) -> float:
        """Sleep for the next backoff interval, waking early if ``stop_event`` is set."""
        delay = self.next_delay()
        if stop_event is not None:
            stop_event.wait(delay)
        else:
            time.sleep(delay)
        return delay


class SourceHealth:
    """Tracks consecutive successes/failures so a source can be demoted or restored."""

    def __init__(self, failover_after: int = 5, recover_after: int = 3):
        self.failover_after = int(failover_after)
        self.recover_after = int(recover_after)
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_success: Optional[float] = None
        self.last_error: Optional[str] = None
        self.total_reports = 0

    def record_success(self, reports: int = 0) -> None:
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        self.last_success = time.monotonic()
        self.last_error = None
        self.total_reports += reports

    def record_failure(self, error: str) -> None:
        self.consecutive_successes = 0
        self.consecutive_failures += 1
        self.last_error = error

    @property
    def failed(self) -> bool:
        return self.consecutive_failures >= self.failover_after

    @property
    def healthy(self) -> bool:
        return self.consecutive_successes >= self.recover_after

    def as_dict(self) -> dict:
        return {
            "failures": self.consecutive_failures,
            "successes": self.consecutive_successes,
            "reports": self.total_reports,
            "last_error": self.last_error,
        }


def as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Coerce feed values to float, mapping junk and non-finite values to ``default``.

    Track feeds are loose with types: altitudes arrive as ``"ground"``, speeds as
    empty strings, positions occasionally as ``NaN``.
    """
    if value is None or value is True or value is False:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    out = as_float(value)
    return default if out is None else int(out)


def clean_str(value: Any) -> Optional[str]:
    """Strip a feed string, mapping blanks and AIS '@' padding to None."""
    if value is None:
        return None
    text = str(value).replace("@", " ").strip()
    return text or None


def wrap360(deg: float) -> float:
    return deg % 360.0


def angle_diff(a: float, b: float) -> float:
    """Smallest signed difference a - b, in (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def lerp_angle(a: float, b: float, t: float) -> float:
    """Interpolate between two bearings the short way around the compass."""
    return wrap360(a + angle_diff(b, a) * t)


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else (high if value > high else value)
