"""Draws the overlay onto a decoded frame, in place.

Skia renders straight into the decoded BGRA buffer through a raster-direct
surface, so there is no intermediate image and no copy: the bytes the decoder
produced are the bytes ffmpeg consumes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import skia

from ..logging_setup import get_logger
from ..projection import ProjectedTarget, sort_by_priority
from .declutter import (LabelAnimator, LabelBox, initial_placement,
                        resolve_overlaps)
from .theme import Theme, build_label_lines, with_alpha

log = get_logger(__name__)


@dataclass
class OverlayStatus:
    """Everything the corner badges need to know."""

    reconnecting: bool = False
    reconnecting_since_s: float = 0.0
    drift_flagged: bool = False
    drift_shift_px: Optional[float] = None
    labels_hidden: bool = False
    attributions: list[str] = field(default_factory=list)
    track_counts: dict = field(default_factory=dict)
    extra_lines: list[str] = field(default_factory=list)


class OverlayRenderer:
    """Composites markers, labels and status badges onto frames."""

    def __init__(self, cfg, model=None):
        self.cfg = cfg
        self.model = model
        self.theme = Theme.from_config(cfg)

        render = cfg.sub("render")
        self.max_labels = int(render.get("max_labels", 28))
        self.declutter_iterations = int(render.get("declutter_iterations", 60))
        self.animator = LabelAnimator(
            fade_in_s=float(render.get("fade_in_s", 0.6)),
            fade_out_s=float(render.get("fade_out_s", 1.0)),
            linger_s=float(render.get("linger_s", 2.0)))

        corner = cfg.sub("render.corner")
        self.show_clock = bool(corner.get("show_clock", True))
        self.show_attribution = bool(corner.get("show_attribution", True))
        self.attribution_text = str(corner.get("attribution", ""))
        self.clock_format = str(corner.get("clock_format", "%Y-%m-%d %H:%M:%S %Z"))
        self.timezone_name = corner.get("timezone")
        self._tz = self._load_timezone(self.timezone_name)

        debug = cfg.sub("render.debug")
        self.debug_landmarks = bool(debug.get("draw_landmarks", False))
        self.debug_horizon = bool(debug.get("draw_horizon", False))
        self.verbose_labels = bool(debug.get("verbose_labels", False))

        self._landmarks = None          # lazily loaded control points
        self._last_render_time: Optional[float] = None
        self._boxes: list[LabelBox] = []
        self.last_overlap_count = 0

    @staticmethod
    def _load_timezone(name):
        if not name:
            return None
        try:
            from zoneinfo import ZoneInfo
            return ZoneInfo(str(name))
        except Exception:
            log.warning("unknown timezone; using UTC", extra={"timezone": name})
            return None

    def set_model(self, model) -> None:
        self.model = model
        self._landmarks = None

    # -- main entry point ---------------------------------------------------
    def render(self, image: np.ndarray, targets: Sequence[ProjectedTarget],
               frame_time: datetime, status: Optional[OverlayStatus] = None,
               now: Optional[float] = None) -> None:
        """Draw onto ``image`` in place. ``image`` must be (H, W, 4) BGRA uint8."""
        status = status or OverlayStatus()
        height, width = image.shape[:2]
        surface = self._wrap(image)
        if surface is None:
            return
        canvas = surface.getCanvas()

        now = now if now is not None else time.monotonic()
        dt = 0.0 if self._last_render_time is None else (now - self._last_render_time)
        self._last_render_time = now

        if self.debug_horizon and self.model is not None:
            self._draw_horizon(canvas, width, height)
        if self.debug_landmarks and self.model is not None:
            self._draw_landmarks(canvas)

        if not status.labels_hidden:
            self._draw_targets(canvas, targets, width, height, now, dt)
        else:
            # Keep the animator's clock moving so labels do not snap back to
            # full opacity the moment they are unhidden.
            self.animator.update([], now, dt)

        self._draw_corner(canvas, width, height, frame_time, status)
        self._draw_badges(canvas, width, height, status)

    def render_badges(self, image: np.ndarray, status: OverlayStatus) -> None:
        """Draw only the status badges, onto a frame that is already composited.

        Used while the source is stalled: the last frame is held on screen and
        needs the RECONNECTING badge (and its counter) redrawn on top of it.
        """
        height, width = image.shape[:2]
        surface = self._wrap(image)
        if surface is not None:
            self._draw_badges(surface.getCanvas(), width, height, status)

    @staticmethod
    def _wrap(image: np.ndarray):
        height, width = image.shape[:2]
        info = skia.ImageInfo.Make(width, height, skia.kBGRA_8888_ColorType,
                                   skia.kUnpremul_AlphaType)
        surface = skia.Surface.MakeRasterDirect(info, memoryview(image.reshape(-1)),
                                                width * 4)
        if surface is None:
            log.error("could not wrap frame buffer for drawing",
                      extra={"shape": list(image.shape)})
        return surface

    # -- targets ------------------------------------------------------------
    def _draw_targets(self, canvas, targets, width, height, now, dt) -> None:
        ranked = sort_by_priority(targets)[:self.max_labels]
        alphas = self.animator.update([t.id for t in ranked], now, dt)

        boxes: list[LabelBox] = []
        for target in ranked:
            alpha = alphas.get(target.id, 0.0)
            if alpha <= 0.01:
                continue
            boxes.append(self._make_box(target, alpha))

        # Report unresolved overlaps rather than discarding the count: a
        # persistently non-zero value means max_labels is set too high for
        # this scene.
        self.last_overlap_count = resolve_overlaps(
            boxes, width, height, gap=self.theme.label_gap_px,
            iterations=self.declutter_iterations)
        self._boxes = boxes

        # Leaders first so they pass under every marker and box.
        for box in boxes:
            self._draw_leader(canvas, box)
        for box in boxes:
            self._draw_marker(canvas, box)
        for box in boxes:
            self._draw_label(canvas, box)

    def _make_box(self, target: ProjectedTarget, alpha: float) -> LabelBox:
        theme = self.theme
        lines = build_label_lines(target, verbose=self.verbose_labels)

        text_width = 0.0
        text_height = 0.0
        for line in lines:
            font = theme.font_bold if line.bold else (
                theme.font_small if line.small else theme.font)
            text_width = max(text_width, font.measureText(line.text))
            text_height += self._line_height(font)

        padding = theme.label_padding
        box = LabelBox(
            target_id=target.id,
            anchor_x=target.u, anchor_y=target.v,
            x=0.0, y=0.0,
            width=text_width + padding * 2,
            height=text_height + padding * 2,
            lines=lines,
            color=theme.category_color(target.category),
            alpha=alpha,
            target=target,
        )
        initial_placement(box, theme.leader_length, self.model.width if self.model
                          else 1920, self.model.height if self.model else 1080)
        return box

    @staticmethod
    def _line_height(font) -> float:
        metrics = font.getMetrics()
        return (metrics.fDescent - metrics.fAscent) + metrics.fLeading

    def _draw_leader(self, canvas, box: LabelBox) -> None:
        theme = self.theme
        attach_x, attach_y = box.attach_point()
        paint = skia.Paint(
            Color=with_alpha(theme.color("leader", 0xB0FFFFFF), box.alpha),
            AntiAlias=True, StrokeWidth=theme.line_width,
            Style=skia.Paint.kStroke_Style)
        canvas.drawLine(box.anchor_x, box.anchor_y, attach_x, attach_y, paint)

    def _draw_marker(self, canvas, box: LabelBox) -> None:
        theme = self.theme
        radius = theme.marker_radius

        # A dark halo keeps the marker readable over bright water and sky alike.
        canvas.drawCircle(box.anchor_x, box.anchor_y, radius + 1.6, skia.Paint(
            Color=with_alpha(0xB0000000, box.alpha), AntiAlias=True,
            StrokeWidth=1.6, Style=skia.Paint.kStroke_Style))
        canvas.drawCircle(box.anchor_x, box.anchor_y, radius, skia.Paint(
            Color=with_alpha(box.color, box.alpha), AntiAlias=True,
            StrokeWidth=theme.line_width + 0.4, Style=skia.Paint.kStroke_Style))
        canvas.drawCircle(box.anchor_x, box.anchor_y, radius * 0.32, skia.Paint(
            Color=with_alpha(box.color, box.alpha), AntiAlias=True))

    def _draw_label(self, canvas, box: LabelBox) -> None:
        theme = self.theme
        rect = skia.Rect.MakeXYWH(box.x, box.y, box.width, box.height)
        radius = theme.label_corner_radius

        canvas.drawRoundRect(rect, radius, radius, skia.Paint(
            Color=with_alpha(theme.color("label_bg", 0xCC0B1622), box.alpha),
            AntiAlias=True))
        canvas.drawRoundRect(rect, radius, radius, skia.Paint(
            Color=with_alpha(box.color, box.alpha * 0.85), AntiAlias=True,
            StrokeWidth=1.2, Style=skia.Paint.kStroke_Style))

        # A short colour bar on the leading edge reads as "category" faster than
        # a coloured border does.
        canvas.drawRect(skia.Rect.MakeXYWH(box.x, box.y + 2, 2.5, box.height - 4),
                        skia.Paint(Color=with_alpha(box.color, box.alpha),
                                   AntiAlias=True))

        foreground = theme.color("label_fg", 0xFFFFFFFF)
        dim = theme.color("label_dim", 0xFFB9C6D2)
        y = box.y + theme.label_padding
        for line in box.lines:
            font = (theme.font_bold if line.bold
                    else theme.font_small if line.small else theme.font)
            metrics = font.getMetrics()
            baseline = y - metrics.fAscent
            color = box.color if line.bold else (dim if line.dim else foreground)
            canvas.drawString(line.text, box.x + theme.label_padding, baseline, font,
                              skia.Paint(Color=with_alpha(color, box.alpha),
                                         AntiAlias=True))
            y += self._line_height(font)

    # -- chrome -------------------------------------------------------------
    def _draw_corner(self, canvas, width, height, frame_time, status) -> None:
        theme = self.theme
        lines: list[str] = []

        if self.show_clock and frame_time is not None:
            stamp = frame_time.astimezone(self._tz) if self._tz else frame_time
            try:
                lines.append(stamp.strftime(self.clock_format))
            except ValueError:
                lines.append(stamp.isoformat(timespec="seconds"))

        if self.show_attribution:
            parts = [p for p in ([self.attribution_text] + list(status.attributions))
                     if p]
            if parts:
                lines.append("  ·  ".join(parts))

        lines.extend(status.extra_lines)
        if not lines:
            return

        font = theme.font_small
        line_height = self._line_height(font)
        box_width = max(font.measureText(line) for line in lines) + 20
        box_height = line_height * len(lines) + 12
        x, y = 14.0, height - box_height - 14.0

        canvas.drawRoundRect(skia.Rect.MakeXYWH(x, y, box_width, box_height), 4, 4,
                             skia.Paint(Color=0xB0000000, AntiAlias=True))
        text_y = y + 6
        for index, line in enumerate(lines):
            metrics = font.getMetrics()
            color = theme.color("label_fg") if index == 0 else theme.color("label_dim")
            canvas.drawString(line, x + 10, text_y - metrics.fAscent, font,
                              skia.Paint(Color=color, AntiAlias=True))
            text_y += line_height

    def _draw_badges(self, canvas, width, height, status: OverlayStatus) -> None:
        theme = self.theme
        badges: list[tuple[str, int]] = []
        if status.reconnecting:
            badges.append((
                f"RECONNECTING  {status.reconnecting_since_s:.0f}s",
                theme.color("warning", 0xFFFF5252)))
        if status.drift_flagged:
            text = "CALIBRATION DRIFT"
            if status.drift_shift_px is not None:
                text += f"  {status.drift_shift_px:.1f}px"
            badges.append((text, theme.color("warning", 0xFFFF5252)))
        if status.labels_hidden:
            badges.append(("LABELS HIDDEN", theme.color("label_dim", 0xFFB9C6D2)))
        if not badges:
            return

        font = theme.font_bold
        line_height = self._line_height(font)
        y = 14.0
        for text, color in badges:
            box_width = font.measureText(text) + 24
            x = width - box_width - 14
            rect = skia.Rect.MakeXYWH(x, y, box_width, line_height + 10)
            canvas.drawRoundRect(rect, 4, 4,
                                 skia.Paint(Color=0xC8000000, AntiAlias=True))
            canvas.drawRoundRect(rect, 4, 4, skia.Paint(
                Color=color, AntiAlias=True, StrokeWidth=1.5,
                Style=skia.Paint.kStroke_Style))
            metrics = font.getMetrics()
            canvas.drawString(text, x + 12, y + 5 - metrics.fAscent, font,
                              skia.Paint(Color=color, AntiAlias=True))
            y += line_height + 18

    # -- debug --------------------------------------------------------------
    def _draw_horizon(self, canvas, width, height) -> None:
        polyline = self.model.horizon_polyline()
        if len(polyline) < 2:
            return
        path = skia.Path()
        path.moveTo(float(polyline[0, 0]), float(polyline[0, 1]))
        for point in polyline[1:]:
            path.lineTo(float(point[0]), float(point[1]))
        canvas.drawPath(path, skia.Paint(
            Color=0x80FFD54F, AntiAlias=True, StrokeWidth=1.2,
            Style=skia.Paint.kStroke_Style))

    def _load_landmarks(self):
        from ..calib.points import load_points

        points_path = self.cfg.path("calibration.points", "./points.csv")
        try:
            return load_points(points_path)
        except (FileNotFoundError, ValueError) as exc:
            log.warning("cannot draw landmarks", extra={"error": str(exc)})
            return False

    def _draw_landmarks(self, canvas) -> None:
        if self._landmarks is None:
            self._landmarks = self._load_landmarks()
        if not self._landmarks:
            return

        theme = self.theme
        point_set = self._landmarks
        enu = point_set.enu(self.model.ref_lat, self.model.ref_lon)
        uv, in_front, _ = self.model.project_enu(enu, refract=True)

        clicked_color = theme.color("debug_point", 0xFFFF00FF)
        reproj_color = theme.color("debug_reproj", 0xFF00E5FF)
        font = theme.font_small

        for index, point in enumerate(point_set.active):
            self._cross(canvas, point.px, point.py, 7, clicked_color)
            if not in_front[index]:
                continue
            u, v = float(uv[index, 0]), float(uv[index, 1])
            canvas.drawCircle(u, v, 6, skia.Paint(
                Color=reproj_color, AntiAlias=True, StrokeWidth=1.4,
                Style=skia.Paint.kStroke_Style))
            canvas.drawLine(point.px, point.py, u, v, skia.Paint(
                Color=reproj_color, AntiAlias=True, StrokeWidth=1.0,
                Style=skia.Paint.kStroke_Style))
            error = float(np.hypot(u - point.px, v - point.py))
            canvas.drawString(f"{point.name} {error:.1f}px", u + 9, v - 8, font,
                              skia.Paint(Color=reproj_color, AntiAlias=True))

    @staticmethod
    def _cross(canvas, x, y, size, color) -> None:
        paint = skia.Paint(Color=color, AntiAlias=True, StrokeWidth=1.6,
                           Style=skia.Paint.kStroke_Style)
        canvas.drawLine(x - size, y, x + size, y, paint)
        canvas.drawLine(x, y - size, x, y + size, paint)
