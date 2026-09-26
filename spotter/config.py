"""Configuration loading.

The whole pipeline is driven by ``config.yaml``. This module loads it, expands
``${ENV_VAR}`` references, resolves relative paths against the repo root, and
exposes dotted-path access so call sites can ask for ``cfg.get("render.colors")``
without defensive dictionary walking.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

REPO_ROOT = Path(__file__).resolve().parent.parent


class ConfigError(Exception):
    """Raised when the configuration file is missing or malformed."""


def _expand_env(value: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``${VAR:-default}`` inside strings."""
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return os.environ.get(name, default if default is not None else "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def deep_merge(base: dict, override: Mapping) -> dict:
    """Merge ``override`` into a copy of ``base``, recursing into dicts.

    Lists are replaced wholesale rather than concatenated: a config that lists
    track sources should be able to *replace* the defaults, not append to them.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class Config(Mapping):
    """Read-only view over the parsed config with dotted-path lookup."""

    __slots__ = ("_data", "_root", "_source")

    def __init__(self, data: Mapping, root: Path | None = None,
                 source: Path | None = None):
        self._data = dict(data)
        self._root = Path(root) if root is not None else REPO_ROOT
        #: The file this was loaded from, so the web UI can write back to the
        #: right one rather than guessing at repo_root/config.yaml.
        self._source = Path(source) if source is not None else None

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return (Config(value, self._root, self._source)
                if isinstance(value, dict) else value)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config({self._data!r})"

    # -- Convenience --------------------------------------------------------
    @property
    def root(self) -> Path:
        """Repo root that relative paths in this config resolve against."""
        return self._root

    @property
    def source_path(self) -> Path | None:
        """The config file this was loaded from, if it came from one."""
        return self._source

    def raw(self) -> dict:
        """The underlying plain dict (a copy, so callers cannot mutate us)."""
        return copy.deepcopy(self._data)

    def get(self, path: str, default: Any = None) -> Any:
        """Look up a dotted path, returning ``default`` if any segment is absent."""
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        if isinstance(node, dict):
            return Config(node, self._root, self._source)
        return node

    def require(self, path: str) -> Any:
        """Like :meth:`get`, but raise when the key is missing."""
        sentinel = object()
        value = self.get(path, sentinel)
        if value is sentinel:
            raise ConfigError(f"missing required config key: {path}")
        return value

    def path(self, dotted: str, default: str | None = None) -> Path | None:
        """Resolve a config value as a filesystem path, relative to the root."""
        value = self.get(dotted, default)
        if value in (None, ""):
            return None
        p = Path(str(value)).expanduser()
        return p if p.is_absolute() else (self._root / p).resolve()

    def sub(self, dotted: str) -> "Config":
        """Return a sub-config, or an empty one when the key is absent."""
        value = self.get(dotted)
        if isinstance(value, Config):
            return value
        if isinstance(value, Mapping):
            return Config(value, self._root, self._source)
        return Config({}, self._root, self._source)


def load_config(path: str | os.PathLike | None = None,
                overrides: Mapping | None = None) -> Config:
    """Load ``config.yaml`` (or ``$SPOTTER_CONFIG``) and return a :class:`Config`.

    ``overrides`` is deep-merged last, which is how the CLI applies ``--set``
    flags and how tests inject fixtures.
    """
    if path is None:
        path = os.environ.get("SPOTTER_CONFIG", REPO_ROOT / "config.yaml")
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise ConfigError(f"config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping, got {type(data).__name__}")

    data = _expand_env(data)
    if overrides:
        data = deep_merge(data, overrides)

    # Relative paths resolve against the config file's directory, which is the
    # repo root in the normal layout and the mount point inside Docker.
    return Config(data, root=cfg_path.parent, source=cfg_path)


def parse_set_overrides(pairs: list[str]) -> dict:
    """Turn ``["render.max_labels=10", "output.enabled=false"]`` into a dict.

    Values are parsed as YAML scalars so numbers, booleans and null work.
    """
    out: dict = {}
    for pair in pairs:
        if "=" not in pair:
            raise ConfigError(f"--set expects key=value, got {pair!r}")
        key, _, raw = pair.partition("=")
        value = yaml.safe_load(raw)
        node = out
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


# ---------------------------------------------------------------------------
# Writing values back
# ---------------------------------------------------------------------------

def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Enough precision for latitudes; strip the trailing zeros a plain
        # repr would leave behind.
        text = f"{value:.8f}".rstrip("0").rstrip(".")
        return text or "0"
    if isinstance(value, (int,)):
        return str(value)
    return quote_if_needed(str(value))


def quote_if_needed(text: str) -> str:
    """Quote a string only when YAML would otherwise misread it."""
    if text == "" or any(c in text for c in ":#{}[],&*?|<>=!%@`\"'\n"):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def set_yaml_value(text: str, dotted: str, value: Any) -> tuple[str, bool]:
    """Replace one scalar in YAML source, preserving comments and layout.

    A round trip through ``yaml.safe_load``/``yaml.dump`` would throw away every
    comment in ``config.yaml``, which is most of its value -- the file is as much
    documentation as configuration. So the edit is textual: find the key by
    walking indentation, and rewrite only the value on that one line.

    Returns ``(new_text, found)``.
    """
    parts = dotted.split(".")
    lines = text.split("\n")

    search_from, search_to = 0, len(lines)
    parent_indent = -1

    for depth, part in enumerate(parts):
        found = None
        for i in range(search_from, search_to):
            raw = lines[i]
            stripped = raw.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("- "):
                continue
            indent = len(raw) - len(raw.lstrip())
            if depth > 0 and indent <= parent_indent:
                search_to = i          # fell out of the parent's block
                break
            if ":" not in stripped:
                continue
            if stripped.split(":", 1)[0].strip() == part:
                found = i
                break

        if found is None:
            return text, False

        if depth == len(parts) - 1:
            key_part, _, rest = lines[found].partition(":")
            comment = ""
            if "#" in rest:
                _, _, comment_text = rest.partition("#")
                comment = "  #" + comment_text.rstrip()
            lines[found] = f"{key_part}: {_format_scalar(value)}{comment}"
            return "\n".join(lines), True

        parent_indent = len(lines[found]) - len(lines[found].lstrip())
        search_from, search_to = found + 1, search_to

    return text, False


def update_config_file(path: str | os.PathLike, updates: Mapping) -> dict:
    """Apply ``{dotted_key: value}`` to a config file, in place.

    Backs the file up first and verifies the result parses and contains the new
    values; if anything looks wrong the original is restored, because a
    half-written config would break the next start.
    """
    path = Path(path)
    original = path.read_text(encoding="utf-8")

    text = original
    applied: dict = {}
    missing: list[str] = []
    for key, value in updates.items():
        text, ok = set_yaml_value(text, key, value)
        if ok:
            # Record what was actually committed, which is the value as
            # formatted -- eight decimal places, about a millimetre of latitude.
            # Verifying against the caller's unrounded input instead would
            # reject a perfectly good write whenever formatting rounds it.
            applied[key] = yaml.safe_load(_format_scalar(value))
        else:
            missing.append(key)

    if missing:
        raise ConfigError(f"keys not found in {path.name}: {', '.join(missing)}")

    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")

    try:
        reloaded = yaml.safe_load(text) or {}
        for key, expected in applied.items():
            node = reloaded
            for part in key.split("."):
                node = node[part]
            # Both sides are the parse of the same literal, so this is exact.
            if node != expected:
                raise ValueError(
                    f"{key} read back as {node!r}, expected {expected!r}")
    except Exception as exc:
        path.write_text(original, encoding="utf-8")
        raise ConfigError(
            f"edit produced an invalid config; original restored ({exc})") from exc

    return applied
