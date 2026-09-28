"""Colours, fonts and label text composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import skia

from ..logging_setup import get_logger
from ..tracks.model import TrackKind

log = get_logger(__name__)

#: Tried in order. DejaVu ships in the Docker image; the rest cover a developer
#: running this straight on a desktop.
FONT_FALLBACKS = ("DejaVu Sans", "Liberation Sans", "Noto Sans", "Helvetica",
                  "Arial", "Segoe UI", "Roboto")

KNOTS_PER_MPS = 1.943844
FEET_PER_M = 3.280839895

#: Two-digit AIS ship type codes, collapsed to something worth reading on screen.
SHIP_TYPE_NAMES = {
    2: "WIG", 3: "Special", 4: "HSC", 5: "Special", 6: "Passenger",
    7: "Cargo", 8: "Tanker", 9: "Other",
}
SHIP_TYPE_EXACT = {
    30: "Fishing", 31: "Towing", 32: "Towing", 33: "Dredger", 34: "Diving",
    35: "Military", 36: "Sailing", 37: "Pleasure craft", 50: "Pilot",
    51: "SAR", 52: "Tug", 53: "Port tender", 55: "Law enforcement",
    58: "Medical", 60: "Passenger", 69: "Passenger", 70: "Cargo",
    80: "Tanker", 90: "Other",
}


#: Colours for categories added after many config.yaml files were written.
#: Anything in the config's ``render.colors`` still wins.
DEFAULT_COLORS = {
    "satellite": "#E1BEE7",
    "satellite_shadow": "#8E7CA0",
    "lightning": "#FFF59D",
}


def parse_color(value, default: int = 0xFFFFFFFF) -> int:
    """Parse ``#RRGGBB`` or ``#RRGGBBAA`` into a Skia ARGB integer."""
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return default
    text = value.strip().lstrip("#")
    try:
        if len(text) == 6:
            rgb = int(text, 16)
            return 0xFF000000 | rgb
        if len(text) == 8:
            rgba = int(text, 16)
            alpha = rgba & 0xFF
            return (alpha << 24) | (rgba >> 8)
    except ValueError:
        pass
    log.warning("could not parse colour; using default", extra={"value": value})
    return default


def with_alpha(color: int, alpha: float) -> int:
    """Scale a colour's alpha by ``alpha`` in [0, 1]."""
    base = (color >> 24) & 0xFF
    scaled = max(0, min(255, int(round(base * alpha))))
    return (scaled << 24) | (color & 0x00FFFFFF)


def load_typeface(preferred: Optional[str] = None) -> skia.Typeface:
    """Resolve a usable typeface, falling back through a list of common ones."""
    candidates = ([preferred] if preferred else []) + list(FONT_FALLBACKS)
    for name in candidates:
        if not name:
            continue
        typeface = skia.Typeface.MakeFromName(name, skia.FontStyle.Normal())
        # Skia substitutes silently, so confirm we got what we asked for before
        # accepting it; otherwise the first name always "works".
        if typeface is not None and typeface.getFamilyName().lower() == name.lower():
            log.info("using font", extra={"family": typeface.getFamilyName()})
            return typeface

    typeface = skia.Typeface.MakeFromName(candidates[0] or "", skia.FontStyle.Normal())
    if typeface is None:
        typeface = skia.Typeface.MakeDefault()
    log.warning("requested font not found; substituting",
                extra={"requested": preferred,
                       "using": typeface.getFamilyName()})
    return typeface


@dataclass
class Theme:
    """Everything the renderer needs to know about how things should look."""

    colors: dict = field(default_factory=dict)
    font_size: float = 15.0
    font_size_small: float = 12.0
    marker_radius: float = 5.0
    leader_length: float = 46.0
    label_padding: float = 6.0
    label_corner_radius: float = 4.0
    line_width: float = 1.6
    label_gap_px: float = 4.0

    typeface: Optional[skia.Typeface] = None
    typeface_bold: Optional[skia.Typeface] = None
    font: Optional[skia.Font] = None
    font_bold: Optional[skia.Font] = None
    font_small: Optional[skia.Font] = None

    @classmethod
    def from_config(cls, cfg) -> "Theme":
        render = cfg.sub("render")
        family = render.get("font_family", "DejaVu Sans")
        typeface = load_typeface(family)
        bold = skia.Typeface.MakeFromName(typeface.getFamilyName(),
                                          skia.FontStyle.Bold()) or typeface

        raw_colors = render.get("colors", {}) or {}
        if hasattr(raw_colors, "raw"):
            raw_colors = raw_colors.raw()

        theme = cls(
            colors={k: parse_color(v)
                    for k, v in {**DEFAULT_COLORS, **raw_colors}.items()},
            font_size=float(render.get("font_size", 15)),
            font_size_small=float(render.get("font_size_small", 12)),
            marker_radius=float(render.get("marker_radius", 5)),
            leader_length=float(render.get("leader_length", 46)),
            label_padding=float(render.get("label_padding", 6)),
            label_corner_radius=float(render.get("label_corner_radius", 4)),
            line_width=float(render.get("line_width", 1.6)),
            label_gap_px=float(render.get("label_gap_px", 4)),
            typeface=typeface, typeface_bold=bold,
        )
        theme.font = skia.Font(typeface, theme.font_size)
        theme.font_bold = skia.Font(bold, theme.font_size)
        theme.font_small = skia.Font(typeface, theme.font_size_small)
        for font in (theme.font, theme.font_bold, theme.font_small):
            font.setSubpixel(True)
            font.setEdging(skia.Font.Edging.kAntiAlias)
        return theme

    def color(self, name: str, default: int = 0xFFB0BEC5) -> int:
        return self.colors.get(name, default)

    def category_color(self, category: str) -> int:
        return self.colors.get(category, self.colors.get("unknown", 0xFFB0BEC5))


# ---------------------------------------------------------------------------
# Label text
# ---------------------------------------------------------------------------

@dataclass
class LabelLine:
    text: str
    bold: bool = False
    small: bool = False
    dim: bool = False


def format_distance(range_m: float) -> str:
    if range_m < 1000:
        return f"{range_m:.0f} m"
    if range_m < 10000:
        return f"{range_m / 1000:.1f} km"
    return f"{range_m / 1000:.0f} km"


def ship_type_name(code) -> Optional[str]:
    try:
        value = int(code)
    except (TypeError, ValueError):
        return None
    if value in SHIP_TYPE_EXACT:
        return SHIP_TYPE_EXACT[value]
    return SHIP_TYPE_NAMES.get(value // 10)


def build_label_lines(target, verbose: bool = False) -> list[LabelLine]:
    """Compose the text block for one target.

    Kept compact on purpose: a label that spans the frame is worse than no
    label. Anything genuinely missing is simply omitted rather than padded with
    placeholders.
    """
    labels = target.labels
    position = target.state.position
    lines: list[LabelLine] = []

    if target.kind is TrackKind.SATELLITE:
        return _satellite_lines(target, verbose)
    if target.kind is TrackKind.LIGHTNING:
        return _lightning_lines(target, verbose)

    if target.kind is TrackKind.AIRCRAFT:
        title = (labels.get("callsign") or labels.get("registration")
                 or labels.get("icao") or "aircraft")
        lines.append(LabelLine(str(title), bold=True))

        parts = []
        altitude = position.alt_m
        if altitude is not None and altitude > 0:
            feet = altitude * FEET_PER_M
            # Above the transition altitude, aviation talks in flight levels.
            parts.append(f"FL{round(feet / 100):03d}" if feet >= 18000
                         else f"{round(feet / 100) * 100:,} ft")
        elif labels.get("on_ground"):
            parts.append("ground")
        if position.speed_mps:
            parts.append(f"{position.speed_mps * KNOTS_PER_MPS:.0f} kt")
        if labels.get("type"):
            parts.append(str(labels["type"]))
        if parts:
            lines.append(LabelLine("  ".join(parts), small=True))

    else:
        title = (labels.get("name") or labels.get("callsign")
                 or (f"MMSI {labels['mmsi']}" if labels.get("mmsi") else "vessel"))
        lines.append(LabelLine(str(title), bold=True))

        parts = []
        type_name = ship_type_name(labels.get("ship_type"))
        if type_name:
            parts.append(type_name)
        if position.speed_mps is not None:
            parts.append(f"{position.speed_mps * KNOTS_PER_MPS:.1f} kt")
        if parts:
            lines.append(LabelLine("  ".join(parts), small=True))

        destination = labels.get("destination")
        if destination:
            lines.append(LabelLine(f"→ {destination}", small=True, dim=True))

    distance = format_distance(target.range_m)
    if verbose:
        distance += (f"   brg {target.bearing_deg:.0f}°"
                     f"  el {target.elevation_deg:+.2f}°"
                     f"  {position.mode}")
    lines.append(LabelLine(distance, small=True, dim=True))
    return lines


def _satellite_lines(target, verbose: bool) -> list[LabelLine]:
    labels = target.labels
    lines = [LabelLine(str(labels.get("name") or "satellite"), bold=True)]
    altitude_km = target.state.position.alt_m / 1000.0
    lines.append(LabelLine(
        f"{altitude_km:,.0f} km up  ·  {target.range_m / 1000:,.0f} km away",
        small=True))
    if not labels.get("sunlit", True):
        visibility = "in Earth's shadow"
    elif labels.get("sky") == "day":
        visibility = "sunlit · daylight sky"
    else:
        visibility = "sunlit"
    if verbose:
        visibility += (f"   brg {target.bearing_deg:.0f}°"
                       f"  el {target.elevation_deg:+.1f}°")
    lines.append(LabelLine(visibility, small=True, dim=True))
    return lines


def _lightning_lines(target, verbose: bool) -> list[LabelLine]:
    labels = target.labels
    lines = [LabelLine("⚡ Lightning", bold=True)]
    distance = labels.get("distance_km")
    age = labels.get("age_s")
    parts = []
    if distance is not None:
        parts.append(format_distance(float(distance) * 1000.0))
    if age is not None:
        parts.append("now" if age < 1.5 else f"{age:.0f} s ago")
    lines.append(LabelLine("  ·  ".join(parts), small=True))
    if verbose:
        lines.append(LabelLine(f"brg {target.bearing_deg:.0f}°", small=True,
                               dim=True))
    return lines
