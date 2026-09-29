#!/usr/bin/env python3
"""Calibration workflow for the Spotter overlay.

Four subcommands, normally run in this order::

    python calibrate.py grab            # pull a still off the live stream
    python calibrate.py click           # click landmarks, type their coordinates
    python calibrate.py solve           # fit the camera model, write calibration.json
    python calibrate.py check           # draw the fit back onto the still and look

See the README section "Calibration workflow" for the full walkthrough.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from spotter.config import ConfigError, load_config, parse_set_overrides
from spotter.logging_setup import get_logger, setup_logging

log = get_logger("calibrate")

DEFAULT_FRAME = "calib_frame.png"

HELP_TEXT = """
 ---------------------------------------------------------------------------
  left click   place a point (then nudge it, then Enter to name it)
  arrows/WASD  nudge the pending point by one full-resolution pixel
  Enter        accept the pending point and type its coordinates
  Esc          cancel the pending point
  u            undo the last accepted point
  l            list the points collected so far
  +/-          zoom the main view in / out
  drag (right) pan the main view
  s            save points.csv and exit
  q            quit without saving
  ?            show this help
 ---------------------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# grab
# ---------------------------------------------------------------------------

def cmd_grab(args, cfg) -> int:
    from spotter.ingest.grab import grab_frame, grab_frame_from_file, save_frame

    out = Path(args.output)
    if args.from_file:
        event = grab_frame_from_file(args.from_file, args.at)
    else:
        print(f"Connecting to {cfg.get('stream.url')} ...", file=sys.stderr)
        event = grab_frame(cfg, timeout_s=args.timeout, skip_frames=args.skip)

    if event is None:
        print("ERROR: could not grab a frame", file=sys.stderr)
        return 1

    save_frame(event.image, out)
    print(f"Wrote {out}  ({event.width}x{event.height})")
    print(f"Frame wall-clock time: {event.wall_time.isoformat()}"
          f"{'' if event.time_is_authoritative else '  (estimated, no PDT)'}")
    return 0


# ---------------------------------------------------------------------------
# click
# ---------------------------------------------------------------------------

class PointClicker:
    """OpenCV window for placing control points precisely.

    Clicking a 1920-wide image inside a window scaled to fit a laptop screen
    costs about two full-resolution pixels of precision per click, which is the
    same order as the reprojection errors we are trying to measure. So a click
    only *proposes* a location: a magnified inset opens and the point can be
    nudged a single full-resolution pixel at a time before it is accepted.
    """

    WINDOW = "spotter calibration - click landmarks"
    ZOOM_WINDOW = "zoom (arrows nudge, Enter accept, Esc cancel)"
    ZOOM_FACTOR = 8
    ZOOM_HALF = 24  # full-res pixels either side shown in the inset

    def __init__(self, image: np.ndarray, point_set, max_window=(1600, 900)):
        import cv2

        self.cv2 = cv2
        self.image = image
        self.point_set = point_set
        self.height, self.width = image.shape[:2]

        self.view_scale = min(max_window[0] / self.width,
                              max_window[1] / self.height, 1.0)
        self.pan = [0.0, 0.0]
        self.pending: Optional[list[float]] = None
        self._panning = False
        self._pan_origin = (0, 0)
        self.quit_requested = False
        self.save_requested = False

    # -- coordinate helpers ---------------------------------------------
    def to_full(self, wx: int, wy: int) -> tuple[float, float]:
        return (wx / self.view_scale + self.pan[0],
                wy / self.view_scale + self.pan[1])

    def to_window(self, fx: float, fy: float) -> tuple[int, int]:
        return (int(round((fx - self.pan[0]) * self.view_scale)),
                int(round((fy - self.pan[1]) * self.view_scale)))

    def _clamp_pan(self) -> None:
        view_w = self.width * self.view_scale
        view_h = self.height * self.view_scale
        max_x = max(0.0, self.width - view_w / self.view_scale)
        max_y = max(0.0, self.height - view_h / self.view_scale)
        self.pan[0] = float(np.clip(self.pan[0], 0.0, max_x))
        self.pan[1] = float(np.clip(self.pan[1], 0.0, max_y))

    # -- rendering ------------------------------------------------------
    def render(self) -> np.ndarray:
        cv2 = self.cv2
        canvas = cv2.resize(
            self.image[:, :, :3],
            (int(self.width * self.view_scale), int(self.height * self.view_scale)),
            interpolation=cv2.INTER_AREA)

        for index, point in enumerate(self.point_set.points):
            x, y = self.to_window(point.px, point.py)
            cv2.drawMarker(canvas, (x, y), (60, 230, 60), cv2.MARKER_CROSS, 16, 2)
            cv2.putText(canvas, f"{index + 1}:{point.name}", (x + 9, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, f"{index + 1}:{point.name}", (x + 9, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 230, 60), 1, cv2.LINE_AA)

        if self.pending is not None:
            x, y = self.to_window(*self.pending)
            cv2.drawMarker(canvas, (x, y), (0, 200, 255), cv2.MARKER_TILTED_CROSS, 22, 2)

        banner = (f"{len(self.point_set.points)} points   "
                  f"zoom {self.view_scale:.2f}x   "
                  f"[?] help  [s] save  [q] quit")
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(canvas, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 240, 240), 1, cv2.LINE_AA)
        return canvas

    def render_zoom(self) -> Optional[np.ndarray]:
        if self.pending is None:
            return None
        cv2 = self.cv2
        cx, cy = int(round(self.pending[0])), int(round(self.pending[1]))
        half = self.ZOOM_HALF
        x0, y0 = cx - half, cy - half
        crop = np.zeros((2 * half + 1, 2 * half + 1, 3), dtype=np.uint8)

        sx0, sy0 = max(0, x0), max(0, y0)
        sx1 = min(self.width, x0 + 2 * half + 1)
        sy1 = min(self.height, y0 + 2 * half + 1)
        if sx1 > sx0 and sy1 > sy0:
            crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                self.image[sy0:sy1, sx0:sx1, :3]

        zoom = cv2.resize(crop, None, fx=self.ZOOM_FACTOR, fy=self.ZOOM_FACTOR,
                          interpolation=cv2.INTER_NEAREST)
        centre = (half * self.ZOOM_FACTOR + self.ZOOM_FACTOR // 2,
                  half * self.ZOOM_FACTOR + self.ZOOM_FACTOR // 2)
        cv2.line(zoom, (centre[0], 0), (centre[0], zoom.shape[0]), (0, 200, 255), 1)
        cv2.line(zoom, (0, centre[1]), (zoom.shape[1], centre[1]), (0, 200, 255), 1)
        cv2.putText(zoom, f"({self.pending[0]:.0f}, {self.pending[1]:.0f})",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(zoom, f"({self.pending[0]:.0f}, {self.pending[1]:.0f})",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
        return zoom

    # -- events ---------------------------------------------------------
    def on_mouse(self, event, x, y, flags, _param) -> None:
        cv2 = self.cv2
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pending = list(self.to_full(x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._panning = True
            self._pan_origin = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self._panning = False
        elif event == cv2.EVENT_MOUSEMOVE and self._panning:
            self.pan[0] -= (x - self._pan_origin[0]) / self.view_scale
            self.pan[1] -= (y - self._pan_origin[1]) / self.view_scale
            self._pan_origin = (x, y)
            self._clamp_pan()
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.view_scale *= 1.2 if flags > 0 else (1 / 1.2)
            self.view_scale = float(np.clip(self.view_scale, 0.1, 8.0))
            self._clamp_pan()


def prompt_point_metadata(px: float, py: float, default_name: str):
    """Ask on the terminal for the landmark's identity and position."""
    from spotter.calib.points import ControlPoint

    print(f"\n  Pixel ({px:.1f}, {py:.1f})")
    name = input(f"    name [{default_name}]: ").strip() or default_name
    try:
        lat_raw = input("    latitude  (decimal degrees): ").strip()
        if not lat_raw:
            print("    -- cancelled (no latitude given)")
            return None
        lon_raw = input("    longitude (decimal degrees): ").strip()
        if not lon_raw:
            print("    -- cancelled (no longitude given)")
            return None
        elev_raw = input("    elevation above sea level, m [0]: ").strip() or "0"
        note = input("    note (optional): ").strip()
        lat, lon, elev = float(lat_raw), float(lon_raw), float(elev_raw)
    except ValueError as exc:
        print(f"    -- cancelled ({exc})")
        return None
    except (EOFError, KeyboardInterrupt):
        print("\n    -- cancelled")
        return None

    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        print("    -- cancelled (coordinates out of range)")
        return None

    print(f"    + {name} @ {lat:.6f},{lon:.6f} {elev:g}m")
    return ControlPoint(name=name, px=px, py=py, lat=lat, lon=lon,
                        elev_m=elev, note=note)


def cmd_click(args, cfg) -> int:
    import cv2

    from spotter.calib.points import ControlPointSet, load_points, save_points
    from spotter.ingest.grab import load_frame

    frame_path = Path(args.image)
    if not frame_path.is_file():
        print(f"ERROR: {frame_path} not found. Run 'calibrate.py grab' first.",
              file=sys.stderr)
        return 1
    image = load_frame(frame_path)

    points_path = Path(args.points or cfg.path("calibration.points", "./points.csv"))
    if points_path.is_file() and not args.fresh:
        point_set = load_points(points_path)
        print(f"Loaded {len(point_set)} existing points from {points_path}")
    else:
        point_set = ControlPointSet()
    point_set.image_width = int(image.shape[1])
    point_set.image_height = int(image.shape[0])

    print(HELP_TEXT)
    print(f"Image: {frame_path}  ({image.shape[1]}x{image.shape[0]})")
    print("Tip: pick things you can find on a map -- breakwater lights, jetty tips,")
    print("     channel buoys, chimneys, the corner of a pier. Mix near and far.\n")

    clicker = PointClicker(image, point_set)
    cv2.namedWindow(clicker.WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(clicker.WINDOW, clicker.on_mouse)

    zoom_open = False
    while True:
        cv2.imshow(clicker.WINDOW, clicker.render())

        zoom = clicker.render_zoom()
        if zoom is not None:
            cv2.imshow(clicker.ZOOM_WINDOW, zoom)
            zoom_open = True
        elif zoom_open:
            cv2.destroyWindow(clicker.ZOOM_WINDOW)
            zoom_open = False

        key = cv2.waitKeyEx(30)
        if key == -1:
            if cv2.getWindowProperty(clicker.WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            continue

        lower = key & 0xFF

        # Arrow keys differ per platform; accept the common codes plus WASD/HJKL.
        nudge = {
            2424832: (-1, 0), 65361: (-1, 0), 81: (-1, 0),      # left
            2555904: (1, 0), 65363: (1, 0), 83: (1, 0),         # right
            2490368: (0, -1), 65362: (0, -1), 82: (0, -1),      # up
            2621440: (0, 1), 65364: (0, 1), 84: (0, 1),         # down
        }.get(key)
        if nudge is None and clicker.pending is not None:
            nudge = {ord("a"): (-1, 0), ord("d"): (1, 0),
                     ord("w"): (0, -1), ord("x"): (0, 1),
                     ord("h"): (-1, 0), ord("l"): (1, 0),
                     ord("k"): (0, -1), ord("j"): (0, 1)}.get(lower)

        if nudge is not None and clicker.pending is not None:
            clicker.pending[0] = float(np.clip(clicker.pending[0] + nudge[0],
                                               0, image.shape[1] - 1))
            clicker.pending[1] = float(np.clip(clicker.pending[1] + nudge[1],
                                               0, image.shape[0] - 1))
            continue

        if lower in (13, 10) and clicker.pending is not None:      # Enter
            point = prompt_point_metadata(clicker.pending[0], clicker.pending[1],
                                          f"pt{len(point_set) + 1}")
            if point is not None:
                point_set.add(point)
            clicker.pending = None
        elif lower == 27:                                          # Esc
            clicker.pending = None
        elif lower == ord("u"):
            if point_set.points:
                removed = point_set.points.pop()
                print(f"  removed '{removed.name}'")
        elif lower == ord("l"):
            print(f"\n  {len(point_set)} points:")
            for i, p in enumerate(point_set.points, 1):
                print(f"    {i:2d}. {p.name:<22} px=({p.px:7.1f},{p.py:7.1f}) "
                      f"{p.lat:.6f},{p.lon:.6f} {p.elev_m:g}m")
            print()
        elif lower in (ord("+"), ord("=")):
            clicker.view_scale = float(min(8.0, clicker.view_scale * 1.2))
        elif lower == ord("-"):
            clicker.view_scale = float(max(0.1, clicker.view_scale / 1.2))
        elif lower == ord("?"):
            print(HELP_TEXT)
        elif lower == ord("s"):
            clicker.save_requested = True
            break
        elif lower == ord("q"):
            clicker.quit_requested = True
            break

    cv2.destroyAllWindows()

    if clicker.quit_requested:
        print("Quit without saving.")
        return 0
    if not point_set.points:
        print("No points collected; nothing written.")
        return 0

    save_points(points_path, point_set)
    print(f"\nWrote {len(point_set)} points to {points_path}")
    print("Next: python calibrate.py solve")
    return 0


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------

def cmd_solve(args, cfg) -> int:
    from spotter.calib.points import load_points
    from spotter.calib.solver import solve_calibration
    from spotter.drift import save_reference_patches
    from spotter.ingest.grab import load_frame

    points_path = Path(args.points or cfg.path("calibration.points", "./points.csv"))
    try:
        point_set = load_points(points_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    width, height = args.width, args.height
    image = None
    frame_path = Path(args.image)
    if frame_path.is_file():
        image = load_frame(frame_path)
        width, height = image.shape[1], image.shape[0]
    elif not (width and height):
        print(f"ERROR: {frame_path} not found and --width/--height not given.",
              file=sys.stderr)
        return 1

    from spotter.calib.lines import load_lines
    line_set = load_lines(cfg.path("calibration.lines", "./lines.json"))

    print(f"Solving from {len(point_set.active)} active points "
          f"({len(point_set)} total) and {len(line_set.active)} traced lines "
          f"at {width}x{height} ...")

    try:
        result = solve_calibration(point_set, cfg, width, height,
                                   run_loo=not args.no_loo, lines=line_set)
    except (ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    model = result.model
    lat, lon = model.camera_latlon
    print("\n=== Solved camera model " + "=" * 45)
    print(f"  position      {lat:.6f}, {lon:.6f}   "
          f"(offset {model.offset_e_m:+.1f}m E, {model.offset_n_m:+.1f}m N)")
    print(f"  height        {model.height_m:.2f} m above sea level")
    print(f"  yaw           {model.yaw_deg:.3f} deg  (bearing of the optical axis)")
    print(f"  pitch         {model.pitch_deg:+.3f} deg")
    print(f"  roll          {model.roll_deg:+.3f} deg")
    print(f"  focal         {model.focal_px:.1f} px   "
          f"-> {model.hfov_deg:.2f} deg horizontal, {model.vfov_deg:.2f} deg vertical")
    print(f"  distortion    k1={model.k1:+.5f}  k2={model.k2:+.5f}")
    print(f"  converged     {result.converged}")

    print("\n=== Per-point reprojection " + "=" * 42)
    header = f"  {'point':<22} {'clicked':>15} {'error':>8}"
    if not args.no_loo:
        header += f" {'LOO err':>9} {'rest RMS':>9}"
    print(header)
    for err in result.errors:
        line = (f"  {err.name:<22} "
                f"{f'{err.px:.0f},{err.py:.0f}':>15} "
                f"{err.error_px:7.2f}p")
        if not args.no_loo and err.loo_error_px is not None:
            line += f" {err.loo_error_px:8.2f}p {err.loo_rest_rms_px:8.2f}p"
        print(line)
    print(f"\n  RMS {result.rms_px:.2f} px    worst {result.max_px:.2f} px")

    if result.warnings:
        print("\n=== Warnings " + "=" * 56)
        for warning in result.warnings:
            print(f"  ! {warning}")
    else:
        print("\n  No warnings: point geometry and residuals look healthy.")

    if args.dry_run:
        print("\n(dry run: nothing written)")
        return 0

    out_path = Path(args.output or cfg.path("calibration.path", "./calibration.json"))
    model.meta = {
        "points_file": str(points_path),
        "n_points": result.n_points,
        "rms_px": round(result.rms_px, 3),
        "max_px": round(result.max_px, 3),
        "warnings": result.warnings,
        "per_point": [e.as_dict() for e in result.errors],
        "seed": result.seed_used,
        "solved_from_image": str(frame_path) if image is not None else None,
    }
    model.save(out_path)
    print(f"\nWrote {out_path}")

    report_path = out_path.with_suffix(".report.json")
    report_path.write_text(json.dumps({
        "summary": result.summary(),
        "points": [e.as_dict() for e in result.errors],
    }, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")

    if image is not None and not args.no_patches:
        patches_dir = cfg.path("drift.patches_dir", "./state/drift_patches")
        refs = save_reference_patches(
            image,
            [(p.name, p.px, p.py) for p in point_set.active],
            patches_dir,
            half_size=int(cfg.get("drift.patch_half_size", 32)),
            want=int(args.patches))
        if refs:
            print(f"Wrote {len(refs)} drift reference patches to {patches_dir}")
            print(f"  using: {', '.join(r.name for r in refs)}")
        else:
            print("WARNING: no landmark had enough texture for a drift patch; "
                  "drift detection will stay disabled")

    print("\nNext: python calibrate.py check")
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def cmd_check(args, cfg) -> int:
    import cv2

    from spotter.calib.model import CameraModel
    from spotter.calib.points import load_points
    from spotter.ingest.grab import load_frame, save_frame

    calib_path = Path(args.calibration or cfg.path("calibration.path",
                                                   "./calibration.json"))
    if not calib_path.is_file():
        print(f"ERROR: {calib_path} not found. Run 'calibrate.py solve' first.",
              file=sys.stderr)
        return 1
    model = CameraModel.load(calib_path)

    frame_path = Path(args.image)
    if not frame_path.is_file():
        print(f"ERROR: {frame_path} not found.", file=sys.stderr)
        return 1
    image = load_frame(frame_path)

    if image.shape[1] != model.width or image.shape[0] != model.height:
        model = model.scaled_to(image.shape[1], image.shape[0])

    points_path = Path(args.points or cfg.path("calibration.points", "./points.csv"))
    point_set = load_points(points_path)

    canvas = image[:, :, :3].copy()

    if args.horizon:
        polyline = model.horizon_polyline()
        inside = polyline[(polyline[:, 0] > -5000) & (polyline[:, 0] < model.width + 5000)]
        if len(inside) > 1:
            pts = inside.astype(np.int32)
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], False, (0, 220, 255), 1,
                          cv2.LINE_AA)

    enu = point_set.enu(model.ref_lat, model.ref_lon)
    uv, in_front, _ = model.project_enu(enu, refract=True)
    errors = []
    for i, point in enumerate(point_set.active):
        clicked = (int(round(point.px)), int(round(point.py)))
        cv2.drawMarker(canvas, clicked, (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
        if not in_front[i]:
            continue
        reproj = (int(round(uv[i, 0])), int(round(uv[i, 1])))
        error = float(np.hypot(uv[i, 0] - point.px, uv[i, 1] - point.py))
        errors.append(error)
        cv2.circle(canvas, reproj, 7, (0, 229, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, clicked, reproj, (0, 229, 255), 1, cv2.LINE_AA)
        label = f"{point.name} {error:.1f}px"
        for color, thickness in (((0, 0, 0), 3), ((0, 229, 255), 1)):
            cv2.putText(canvas, label, (reproj[0] + 10, reproj[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA)

    legend = [
        f"calibration: {calib_path.name}",
        "magenta cross = clicked   yellow circle = reprojected   orange line = horizon",
        (f"RMS {np.sqrt(np.mean(np.square(errors))):.2f}px over {len(errors)} points"
         if errors else "no points reprojected in front of the camera"),
        f"yaw {model.yaw_deg:.2f}  pitch {model.pitch_deg:+.2f}  "
        f"roll {model.roll_deg:+.2f}  hfov {model.hfov_deg:.2f}",
    ]
    cv2.rectangle(canvas, (0, 0), (620, 20 + 22 * len(legend)), (0, 0, 0), -1)
    for i, line in enumerate(legend):
        cv2.putText(canvas, line, (10, 26 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (240, 240, 240), 1, cv2.LINE_AA)

    out = Path(args.output)
    save_frame(np.dstack([canvas, np.full(canvas.shape[:2], 255, np.uint8)]), out)
    print(f"Wrote {out}")
    if errors:
        print(f"RMS {np.sqrt(np.mean(np.square(errors))):.2f}px  "
              f"worst {max(errors):.2f}px over {len(errors)} points")

    if args.show:
        scale = min(1600 / canvas.shape[1], 900 / canvas.shape[0], 1.0)
        preview = cv2.resize(canvas, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        cv2.imshow("calibration check - any key to close", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--config", default=None, help="path to config.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set camera.height_m=22")
    parser.add_argument("--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    p_grab = sub.add_parser("grab", help="grab a still frame from the stream")
    p_grab.add_argument("-o", "--output", default=DEFAULT_FRAME)
    p_grab.add_argument("--timeout", type=float, default=120.0)
    p_grab.add_argument("--skip", type=int, default=15,
                        help="discard this many frames before keeping one")
    p_grab.add_argument("--from-file", default=None,
                        help="grab from a local video file instead of the stream")
    p_grab.add_argument("--at", type=float, default=0.0,
                        help="seconds into the file to grab from")
    p_grab.set_defaults(func=cmd_grab)

    p_click = sub.add_parser("click", help="click landmarks and type coordinates")
    p_click.add_argument("-i", "--image", default=DEFAULT_FRAME)
    p_click.add_argument("-p", "--points", default=None)
    p_click.add_argument("--fresh", action="store_true",
                         help="start a new points file instead of appending")
    p_click.set_defaults(func=cmd_click)

    p_solve = sub.add_parser("solve", help="fit the camera model")
    p_solve.add_argument("-p", "--points", default=None)
    p_solve.add_argument("-i", "--image", default=DEFAULT_FRAME)
    p_solve.add_argument("-o", "--output", default=None)
    p_solve.add_argument("--width", type=int, default=0)
    p_solve.add_argument("--height", type=int, default=0)
    p_solve.add_argument("--no-loo", action="store_true",
                         help="skip the leave-one-out report (faster)")
    p_solve.add_argument("--no-patches", action="store_true",
                         help="do not write drift reference patches")
    p_solve.add_argument("--patches", type=int, default=6,
                         help="how many drift reference patches to save")
    p_solve.add_argument("--dry-run", action="store_true")
    p_solve.set_defaults(func=cmd_solve)

    p_check = sub.add_parser("check", help="draw the fit back onto the still")
    p_check.add_argument("-i", "--image", default=DEFAULT_FRAME)
    p_check.add_argument("-p", "--points", default=None)
    p_check.add_argument("--calibration", default=None)
    p_check.add_argument("-o", "--output", default="calib_check.png")
    p_check.add_argument("--horizon", action="store_true",
                         help="also draw the computed horizon line")
    p_check.add_argument("--show", action="store_true",
                         help="open a window as well as writing the file")
    p_check.set_defaults(func=cmd_check)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, overrides=parse_set_overrides(args.set))
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    setup_logging(cfg, level=args.log_level, fmt="console")
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
