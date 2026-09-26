"""Structured logging.

Emits either line-delimited JSON (for journald/Loki/Docker log drivers) or a
readable console format. Extra keyword context is attached via the ``extra=``
dict and lands in the JSON payload as top-level fields.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, Mapping

# Attributes LogRecord always carries; anything else was added by the caller and
# belongs in the structured payload.
_STANDARD = frozenset(
    """args asctime created exc_info exc_text filename funcName levelname levelno
    lineno module msecs message msg name pathname process processName relativeCreated
    stack_info thread threadName taskName""".split()
)


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD or key.startswith("_"):
                continue
            payload[key] = _jsonable(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable single line: time, level, logger, message, then k=v pairs."""

    _COLORS = {
        "DEBUG": "\033[36m", "INFO": "\033[32m", "WARNING": "\033[33m",
        "ERROR": "\033[31m", "CRITICAL": "\033[1;31m",
    }
    _RESET = "\033[0m"

    def __init__(self, color: bool = True):
        super().__init__()
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        level = record.levelname
        if self.color:
            level = f"{self._COLORS.get(level, '')}{level:<7}{self._RESET}"
        else:
            level = f"{level:<7}"
        extras = " ".join(
            f"{k}={_compact(v)}"
            for k, v in record.__dict__.items()
            if k not in _STANDARD and not k.startswith("_")
        )
        line = f"{ts} {level} {record.name:<26} {record.getMessage()}"
        if extras:
            line += f"  {extras}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _compact(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    text = str(value)
    return text if len(text) <= 60 else text[:57] + "..."


def setup_logging(cfg=None, level: str | None = None, fmt: str | None = None) -> None:
    """Configure the root logger. Safe to call more than once."""
    if cfg is not None:
        level = level or cfg.get("logging.level", "INFO")
        fmt = fmt or cfg.get("logging.format", "json")
    level = (level or "INFO").upper()
    fmt = (fmt or "json").lower()

    handler = logging.StreamHandler(sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(ConsoleFormatter(color=sys.stderr.isatty()))

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    # Third-party libraries are chatty at DEBUG; keep them at INFO unless asked.
    for noisy in ("urllib3", "streamlink", "libav", "websockets", "PIL"):
        logging.getLogger(noisy).setLevel(max(logging.INFO, root.level))

    if cfg is not None:
        for name, lvl in (cfg.get("logging.levels", {}) or {}).items():
            logging.getLogger(name).setLevel(str(lvl).upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
