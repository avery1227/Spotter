"""Web UI: calibrate by clicking, and watch the composited output.

Two pages, one small Flask app:

* ``/``         -- calibration. Click a landmark in the frame, click the same
                   spot on a map (or paste coordinates straight out of Google
                   Maps), solve, and see the reprojection drawn back on the
                   image.
* ``/monitor``  -- live view of what is being streamed, plus pipeline status.

The server can run standalone (``spotter web``) or inside a running pipeline
(``spotter run --web``), in which case the monitor shows real composited frames.

It binds to localhost by default. There is no authentication: this is a tool for
your own machine or a trusted LAN, and ``web.host`` should only be widened with
that in mind.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
from flask import Flask, Response, jsonify, request, send_file, send_from_directory

from ..calib.lines import LineFeature, LineSet, load_lines, save_lines
from ..calib.model import CameraModel
from ..calib.points import (KIND_AIRCRAFT, KIND_LANDMARK, ControlPoint, ControlPointSet,
                            load_points, save_points)
from ..calib.solver import prepare_lines, solve_calibration
from ..config import ConfigError, load_config, update_config_file
from ..logging_setup import get_logger
from .frames import FrameHub

log = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class WebState:
    """Everything the routes need, so the app factory stays a factory."""

    def __init__(self, cfg, hub: Optional[FrameHub] = None, pipeline=None):
        self.cfg = cfg
        self.hub = hub if hub is not None else FrameHub(
            max_width=int(cfg.get("web.preview_width", 960)),
            quality=int(cfg.get("web.preview_quality", 70)))
        self.pipeline = pipeline
        self.lock = threading.Lock()

        self.frame_path = Path(cfg.get("web.calibration_frame",
                                       "calib_frame.png"))
        if not self.frame_path.is_absolute():
            self.frame_path = cfg.root / self.frame_path
        self.points_path = cfg.path("calibration.points", "./points.csv")
        self.lines_path = cfg.path("calibration.lines", "./lines.json")
        self.calibration_path = cfg.path("calibration.path", "./calibration.json")

        #: Last solve result, so the overlay endpoint can draw it.
        self.last_solve = None
        #: Recent frozen frames, newest last. A few, not one, so a second
        #: freeze does not pull the image out from under a click in progress.
        self.freezes: "OrderedDict[str, object]" = OrderedDict()

    def reload_config(self) -> None:
        """Re-read config.yaml after the UI has written to it."""
        source = self.cfg.source_path
        if source is None:
            raise ConfigError("this config was not loaded from a file")
        self.cfg = load_config(source)
        self.points_path = self.cfg.path("calibration.points", "./points.csv")
        self.lines_path = self.cfg.path("calibration.lines", "./lines.json")
        self.calibration_path = self.cfg.path("calibration.path",
                                              "./calibration.json")

    # -- points -------------------------------------------------------------
    def load_point_set(self) -> ControlPointSet:
        if self.points_path and self.points_path.is_file():
            try:
                return load_points(self.points_path)
            except (ValueError, FileNotFoundError) as exc:
                log.warning("could not read points file", extra={"error": str(exc)})
        return ControlPointSet()

    def save_point_set(self, point_set: ControlPointSet) -> None:
        """Save, keeping a rotating backup of what was there before.

        points.csv represents half an hour of careful clicking. A stray click
        on a delete button, a stale browser tab, or two tabs open at once
        should not be able to destroy it irrecoverably.
        """
        self._backup_points()
        save_points(self.points_path, point_set)

    def _backup_points(self) -> None:
        if not self.points_path or not self.points_path.is_file():
            return
        try:
            history = self.points_path.parent / "state" / "points_history"
            history.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            target = history / f"points-{stamp}.csv"
            if not target.exists():
                target.write_bytes(self.points_path.read_bytes())
            # Keep the most recent few dozen; they are a couple of kB each.
            existing = sorted(history.glob("points-*.csv"))
            for stale in existing[:-40]:
                stale.unlink()
        except OSError as exc:
            log.warning("could not back up points file",
                        extra={"error": str(exc)})

    def load_line_set(self) -> LineSet:
        try:
            return load_lines(self.lines_path)
        except (ValueError, OSError) as exc:
            log.warning("could not read lines file", extra={"error": str(exc)})
            return LineSet()

    def remember_freeze(self, capture) -> None:
        self.freezes[capture.id] = capture
        while len(self.freezes) > 4:
            self.freezes.popitem(last=False)

    def frame_size(self) -> tuple[int, int]:
        """Size of the calibration frame, for the solver."""
        import cv2

        if self.frame_path.is_file():
            image = cv2.imread(str(self.frame_path), cv2.IMREAD_COLOR)
            if image is not None:
                return int(image.shape[1]), int(image.shape[0])
        return 1920, 1080


def create_app(state: WebState) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["JSON_SORT_KEYS"] = False

    # -- pages --------------------------------------------------------------
    @app.route("/")
    def index():
        return send_from_directory(STATIC_DIR, "index.html")

    @app.route("/monitor")
    def monitor_page():
        return send_from_directory(STATIC_DIR, "monitor.html")

    @app.route("/view")
    def view_page():
        """Full-screen output only, for a wall display."""
        return send_from_directory(STATIC_DIR, "view.html")

    @app.route("/static/<path:name>")
    def static_files(name):
        return send_from_directory(STATIC_DIR, name)

    # -- calibration frame --------------------------------------------------
    @app.route("/api/frame.png")
    def calibration_frame():
        if not state.frame_path.is_file():
            return jsonify({"error": "no calibration frame yet; press Grab"}), 404
        # The browser caches aggressively; the mtime query param busts it.
        return send_file(str(state.frame_path), mimetype="image/png",
                         max_age=0)

    @app.route("/api/grab", methods=["POST"])
    def grab():
        """Pull a fresh still off the live stream."""
        from ..ingest.grab import grab_frame, save_frame

        timeout = float(request.json.get("timeout", 120)) if request.is_json else 120
        try:
            event = grab_frame(state.cfg, timeout_s=timeout, skip_frames=15)
        except Exception as exc:
            log.exception("frame grab failed")
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
        if event is None:
            return jsonify({"error": "timed out waiting for a frame"}), 504

        save_frame(event.image, state.frame_path)
        return jsonify({
            "width": event.width, "height": event.height,
            "captured": event.wall_time.isoformat(),
            "authoritative": event.time_is_authoritative,
            "path": str(state.frame_path),
        })

    @app.route("/api/info")
    def info():
        width, height = state.frame_size()
        model = None
        if state.calibration_path and state.calibration_path.is_file():
            try:
                loaded = CameraModel.load(state.calibration_path)
                model = _model_summary(loaded)
            except Exception as exc:
                log.warning("could not load calibration",
                            extra={"error": str(exc)})
        return jsonify({
            "frame": {"width": width, "height": height,
                      "exists": state.frame_path.is_file(),
                      "mtime": (state.frame_path.stat().st_mtime
                                if state.frame_path.is_file() else 0)},
            "camera": {"lat": state.cfg.get("camera.lat"),
                       "lon": state.cfg.get("camera.lon"),
                       "height_m": state.cfg.get("camera.height_m"),
                       "height_sigma_m":
                           state.cfg.get("calibration.solver.height_sigma_m"),
                       "editable": state.cfg.source_path is not None},
            "calibration": model,
            "points_path": str(state.points_path),
            "live": state.pipeline is not None,
            "can_freeze": state.pipeline is not None
            and hasattr(state.pipeline, "capture_freeze"),
            "encoder_delay_s": state.cfg.get("stream.encoder_delay_s"),
        })

    # -- camera position ----------------------------------------------------
    @app.route("/api/camera", methods=["POST"])
    def set_camera():
        """Write the camera position back to config.yaml.

        The solver treats this as a soft prior rather than truth, so being tens
        of metres out is fine -- but the map opens here, and a wildly wrong
        value makes the fit fight the prior.
        """
        if state.cfg.source_path is None:
            return jsonify({
                "error": "this configuration was not loaded from a file, so "
                         "there is nothing to write to"}), 409

        body = request.get_json(silent=True) or {}
        try:
            lat = float(body["lat"])
            lon = float(body["lon"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "lat and lon are required"}), 400
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({"error": "coordinates out of range"}), 400

        height_unknown = bool(body.get("height_unknown", False))
        try:
            height = float(body.get("height_m") or 0.0)
        except (TypeError, ValueError):
            return jsonify({"error": "height must be a number"}), 400
        if height_unknown and height <= 0:
            # Somewhere to start from. The solver moves off this freely once
            # the prior is widened below.
            height = float(state.cfg.get("web.unknown_height_guess_m", 12.0))
        if height <= 0:
            return jsonify({"error": "height must be greater than zero"}), 400

        # An unknown height has to be free to move, so widen its prior. The
        # catch is that height and pitch are nearly degenerate when every point
        # sits on the horizon -- near-field points are what separate them, and
        # the UI says so.
        sigma = float(state.cfg.get("web.unknown_height_sigma_m", 60.0))             if height_unknown else             float(state.cfg.get("web.known_height_sigma_m", 5.0))

        try:
            applied = update_config_file(state.cfg.source_path, {
                "camera.lat": lat,
                "camera.lon": lon,
                "camera.height_m": height,
                "calibration.solver.height_sigma_m": sigma,
            })
            state.reload_config()
        except ConfigError as exc:
            return jsonify({"error": str(exc)}), 500

        # Report the values as stored, not as submitted: writing rounds to
        # eight decimal places, and the UI should show what is really in the
        # file rather than what was typed.
        stored = {
            "lat": applied.get("camera.lat", lat),
            "lon": applied.get("camera.lon", lon),
            "height_m": applied.get("camera.height_m", height),
        }
        log.info("camera position updated from the web UI", extra={
            **stored, "height_unknown": height_unknown, "height_sigma_m": sigma})
        return jsonify({
            "camera": stored,
            "height_unknown": height_unknown,
            "height_sigma_m": applied.get("calibration.solver.height_sigma_m", sigma),
            "written_to": str(state.cfg.source_path),
            "applied": list(applied),
        })

    # -- points CRUD --------------------------------------------------------
    @app.route("/api/points", methods=["GET"])
    def get_points():
        point_set = state.load_point_set()
        return jsonify({"points": [_point_json(p) for p in point_set.points]})

    @app.route("/api/points", methods=["POST"])
    def add_point():
        body = request.get_json(silent=True) or {}
        try:
            point = ControlPoint(
                name=str(body.get("name") or "").strip() or "point",
                px=float(body["px"]), py=float(body["py"]),
                lat=float(body["lat"]), lon=float(body["lon"]),
                elev_m=float(body.get("elev_m") or 0.0),
                enabled=bool(body.get("enabled", True)),
                note=str(body.get("note") or ""),
                kind=str(body.get("kind") or KIND_LANDMARK),
                time=str(body.get("time") or ""),
                delay_s=_opt_float(body.get("delay_s")),
                ve=_opt_float(body.get("ve")),
                vn=_opt_float(body.get("vn")),
                vu=_opt_float(body.get("vu")))
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": f"bad point: {exc}"}), 400
        if point.kind not in (KIND_LANDMARK, KIND_AIRCRAFT):
            return jsonify({"error": f"unknown point kind '{point.kind}'"}), 400
        if not (-90 <= point.lat <= 90) or not (-180 <= point.lon <= 180):
            return jsonify({"error": "coordinates out of range"}), 400

        with state.lock:
            point_set = state.load_point_set()
            point_set.add(point)
            state.save_point_set(point_set)
        return jsonify({"points": [_point_json(p) for p in point_set.points]})

    @app.route("/api/points/<int:index>", methods=["PATCH", "DELETE"])
    def modify_point(index: int):
        with state.lock:
            point_set = state.load_point_set()
            if not 0 <= index < len(point_set.points):
                return jsonify({"error": "no such point"}), 404

            # Index alone is not a safe handle: a stale tab, a second browser,
            # or a mis-aimed click can address a different point than the one
            # the user is looking at. The caller must name what it expects.
            expected = request.args.get("name")
            if expected is None and request.is_json:
                expected = (request.get_json(silent=True) or {}).get("expect_name")
            actual = point_set.points[index].name
            if expected is not None and expected != actual:
                return jsonify({
                    "error": f"point {index} is '{actual}', not '{expected}'; "
                             f"reload the page"}), 409

            if request.method == "DELETE":
                if expected is None:
                    return jsonify({
                        "error": "delete requires ?name=<expected> so a stale "
                                 "page cannot remove the wrong point"}), 400
                removed = point_set.points.pop(index)
                state.save_point_set(point_set)
                return jsonify({
                    "points": [_point_json(p) for p in point_set.points],
                    "removed": _point_json(removed)})
            else:
                body = request.get_json(silent=True) or {}
                point = point_set.points[index]
                for field in ("name", "note"):
                    if field in body:
                        setattr(point, field, str(body[field]))
                for field in ("px", "py", "lat", "lon", "elev_m"):
                    if field in body:
                        try:
                            setattr(point, field, float(body[field]))
                        except (TypeError, ValueError):
                            return jsonify({"error": f"bad {field}"}), 400
                if "enabled" in body:
                    point.enabled = bool(body["enabled"])
            state.save_point_set(point_set)
        return jsonify({"points": [_point_json(p) for p in point_set.points]})

    # -- solve --------------------------------------------------------------
    @app.route("/api/solve", methods=["POST"])
    def solve():
        body = request.get_json(silent=True) or {}
        point_set = state.load_point_set()
        line_set = state.load_line_set()
        if len(point_set.active) + 2 * len(line_set.active) < 4:
            return jsonify({
                "error": f"need at least 4 enabled points (a traced line counts "
                         f"as 2), have {len(point_set.active)} points and "
                         f"{len(line_set.active)} lines"}), 400

        width, height = state.frame_size()
        overrides = {}
        for key in ("lock_height", "lock_position", "lock_roll", "lock_distortion",
                    "fit_timing"):
            if key in body:
                overrides[key] = bool(body[key])

        cfg = state.cfg
        if overrides:
            from ..config import Config, deep_merge
            cfg = Config(deep_merge(cfg.raw(),
                                    {"calibration": {"solver": overrides}}),
                         root=cfg.root)

        try:
            result = solve_calibration(point_set, cfg, width, height,
                                       run_loo=bool(body.get("loo", True)),
                                       lines=line_set)
        except (ValueError, RuntimeError) as exc:
            return jsonify({"error": str(exc)}), 400

        state.last_solve = result

        if body.get("save", False):
            model = result.model
            model.meta = {
                "points_file": str(state.points_path),
                "n_points": result.n_points,
                "rms_px": round(result.rms_px, 3),
                "max_px": round(result.max_px, 3),
                "warnings": result.warnings,
                "per_point": [e.as_dict() for e in result.errors],
                "n_lines": result.n_lines,
                "per_line": result.line_errors,
                "timing_offset_s": result.timing_offset_s,
                "suggested_encoder_delay_s": result.suggested_encoder_delay_s,
                "solved_from_image": str(state.frame_path),
                "solved_via": "web",
            }
            model.save(state.calibration_path)
            _write_drift_patches(state, point_set)

        return jsonify({
            "summary": result.summary(),
            "model": _model_summary(result.model),
            "points": [e.as_dict() for e in result.errors],
            "lines": result.line_errors,
            "warnings": result.warnings,
            "saved": bool(body.get("save", False)),
            "pipeline_waiting": bool(
                state.pipeline is not None
                and getattr(state.pipeline, "waiting_for_calibration", False)),
        })

    @app.route("/api/reprojection")
    def reprojection():
        """Where the current calibration puts each point, for drawing on the frame."""
        path = state.calibration_path
        if not path or not path.is_file():
            return jsonify({"error": "no calibration yet"}), 404
        model = CameraModel.load(path)
        width, height = state.frame_size()
        if (model.width, model.height) != (width, height):
            try:
                model = model.scaled_to(width, height)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400

        point_set = state.load_point_set()
        out = []
        if point_set.points:
            enu = point_set.enu(model.ref_lat, model.ref_lon, active_only=False)
            uv, in_front, _ = model.project_enu(enu, refract=True)
            for i, point in enumerate(point_set.points):
                entry = {"name": point.name, "px": point.px, "py": point.py,
                         "in_front": bool(in_front[i]), "enabled": point.enabled}
                if in_front[i]:
                    entry["u"] = float(uv[i, 0])
                    entry["v"] = float(uv[i, 1])
                    entry["error_px"] = float(np.hypot(uv[i, 0] - point.px,
                                                       uv[i, 1] - point.py))
                out.append(entry)

        # Each map line as the saved calibration projects it, at the height the
        # solve settled on, so a bad tracing shows up as two lines that part.
        fitted = {entry["name"]: entry.get("elev_m")
                  for entry in (model.meta.get("per_line") or [])}
        lines_out = []
        for index, line in enumerate(state.load_line_set().lines):
            elev = fitted.get(line.name, line.elev_m)
            shifted = LineFeature(name=line.name, image=line.image, map=line.map,
                                  elev_m=elev if elev is not None else line.elev_m,
                                  elev_sigma_m=0.0)
            samples = prepare_lines(LineSet(lines=[shifted]), model.ref_lat,
                                    model.ref_lon)[0].samples_enu
            uv, front, _ = model.project_enu(samples, refract=True)
            runs, current = [], []
            for (u, v), ok in zip(uv, front):
                if ok:
                    current.append([round(float(u), 1), round(float(v), 1)])
                elif current:
                    runs.append(current)
                    current = []
            if current:
                runs.append(current)
            lines_out.append({"index": index, "name": line.name,
                              "enabled": line.enabled, "projected": runs})

        horizon = model.horizon_polyline()
        return jsonify({
            "points": out,
            "lines": lines_out,
            "horizon": [[float(x), float(y)] for x, y in horizon],
            "model": _model_summary(model),
        })

    # -- traced lines -------------------------------------------------------
    @app.route("/api/lines", methods=["GET"])
    def get_lines():
        return jsonify({"lines": [line.to_json() for line in state.load_line_set().lines]})

    @app.route("/api/lines", methods=["POST"])
    def add_line():
        body = request.get_json(silent=True) or {}
        try:
            line = LineFeature.from_json(body)
        except (KeyError, TypeError, ValueError) as exc:
            return jsonify({"error": f"bad line: {exc}"}), 400
        with state.lock:
            line_set = state.load_line_set()
            line_set.lines.append(line)
            save_lines(state.lines_path, line_set)
        return jsonify({"lines": [entry.to_json() for entry in line_set.lines]})

    @app.route("/api/lines/<int:index>", methods=["PATCH", "DELETE"])
    def modify_line(index: int):
        with state.lock:
            line_set = state.load_line_set()
            if not 0 <= index < len(line_set.lines):
                return jsonify({"error": "no such line"}), 404
            # Same guard as points: name the line you mean.
            expected = request.args.get("name")
            actual = line_set.lines[index].name
            if expected is None:
                return jsonify({"error": "pass ?name=<expected> so a stale page "
                                         "cannot change the wrong line"}), 400
            if expected != actual:
                return jsonify({"error": f"line {index} is '{actual}', not "
                                         f"'{expected}'; reload the page"}), 409
            if request.method == "DELETE":
                removed = line_set.lines.pop(index)
                save_lines(state.lines_path, line_set)
                return jsonify({"lines": [e.to_json() for e in line_set.lines],
                                "removed": removed.to_json()})
            body = request.get_json(silent=True) or {}
            line = line_set.lines[index]
            try:
                if "name" in body:
                    line.name = str(body["name"]).strip() or line.name
                if "enabled" in body:
                    line.enabled = bool(body["enabled"])
                if "elev_m" in body:
                    line.elev_m = float(body["elev_m"])
                if "elev_sigma_m" in body:
                    line.elev_sigma_m = float(body["elev_sigma_m"])
                line.validate()
            except (TypeError, ValueError) as exc:
                return jsonify({"error": str(exc)}), 400
            save_lines(state.lines_path, line_set)
        return jsonify({"lines": [e.to_json() for e in line_set.lines]})

    # -- frozen frames for aircraft points ---------------------------------
    @app.route("/api/freeze", methods=["POST"])
    def freeze():
        pipeline = state.pipeline
        if pipeline is None or not hasattr(pipeline, "capture_freeze"):
            return jsonify({"error": "freezing needs the live pipeline: start "
                                     "Spotter with 'run', not 'web'"}), 409
        try:
            capture = pipeline.capture_freeze(
                timeout_s=float(state.cfg.get("web.freeze_timeout_s", 8.0)))
        except TimeoutError as exc:
            return jsonify({"error": str(exc)}), 504
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 409
        state.remember_freeze(capture)
        log.info("frame frozen for aircraft calibration", extra={
            "frame_time": capture.frame_time.isoformat(),
            "aircraft": len(capture.aircraft)})
        return jsonify(capture.to_json())

    @app.route("/api/freeze/<freeze_id>.png")
    def freeze_image(freeze_id: str):
        import cv2

        capture = state.freezes.get(freeze_id)
        if capture is None:
            return jsonify({"error": "that frozen frame has expired; freeze again"}), 404
        image = capture.image
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        ok, data = cv2.imencode(".png", image)
        if not ok:
            return jsonify({"error": "could not encode frame"}), 500
        response = Response(data.tobytes(), mimetype="image/png")
        response.headers["Cache-Control"] = "private, max-age=3600"
        return response

    @app.route("/api/freeze/<freeze_id>/points", methods=["POST"])
    def add_aircraft_point(freeze_id: str):
        """Add a plane clicked on a frozen frame, positioned from that capture.

        The position comes from the server's copy of the capture rather than
        the request, so the page cannot post an aircraft point for a time the
        frame was not taken at.
        """
        capture = state.freezes.get(freeze_id)
        if capture is None:
            return jsonify({"error": "that frozen frame has expired; freeze again"}), 404
        body = request.get_json(silent=True) or {}
        plane = capture.find(str(body.get("aircraft") or ""))
        if plane is None:
            return jsonify({"error": "that aircraft is not in this capture"}), 400
        try:
            px, py = float(body["px"]), float(body["py"])
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "px and py are required"}), 400

        stamp = capture.frame_time.strftime("%H:%M:%S")
        point = ControlPoint(
            name=f"{plane.label} {stamp}", px=px, py=py,
            lat=plane.lat, lon=plane.lon, elev_m=plane.alt_m,
            kind=KIND_AIRCRAFT, time=capture.frame_time.isoformat(),
            delay_s=capture.delay_s, ve=plane.ve, vn=plane.vn, vu=plane.vu,
            note=(f"{plane.alt_source} altitude, {plane.range_km:.1f} km"
                  + (" (pressure altitude: can be 100 m+ off)"
                     if plane.alt_source == "baro" else "")))
        with state.lock:
            point_set = state.load_point_set()
            point_set.add(point)
            state.save_point_set(point_set)
        return jsonify({"points": [_point_json(p) for p in point_set.points],
                        "added": _point_json(point)})

    # -- live monitor -------------------------------------------------------
    @app.route("/api/live.jpg")
    def live_jpeg():
        result = state.hub.jpeg(
            max_width=request.args.get("width", type=int),
            quality=request.args.get("quality", type=int))
        if result is None:
            return jsonify({"error": "no frame yet"}), 404
        data, sequence = result
        response = Response(data, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Frame-Sequence"] = str(sequence)
        return response

    @app.route("/api/live.mjpg")
    def live_mjpeg():
        fps = float(request.args.get("fps", state.cfg.get("web.preview_fps", 6)))
        width = request.args.get("width", type=int)
        quality = request.args.get("quality", type=int)
        interval = 1.0 / max(0.5, min(fps, 30.0))
        boundary = "spotterframe"

        def generate():
            last = -1
            while True:
                started = time.monotonic()
                last = state.hub.wait_for_frame(last, timeout=5.0)
                result = state.hub.jpeg(max_width=width, quality=quality)
                if result is not None:
                    data, last = result
                    yield (b"--" + boundary.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(data)).encode()
                           + b"\r\n\r\n" + data + b"\r\n")
                # Rate-limit regardless of how fast frames arrive; a preview at
                # full frame rate would cost more CPU than the real encode.
                delay = interval - (time.monotonic() - started)
                if delay > 0:
                    time.sleep(delay)

        return Response(generate(),
                        mimetype=f"multipart/x-mixed-replace; boundary={boundary}")

    @app.route("/api/status")
    def status():
        payload = {"live": state.pipeline is not None,
                   "frame_age_s": state.hub.age_s(),
                   "preview_sequence": state.hub.sequence}
        pipeline = state.pipeline
        if pipeline is not None:
            payload.update(pipeline.web_status())
        return jsonify(payload)

    return app

# ---------------------------------------------------------------------------


def _opt_float(value):
    return None if value in (None, "") else float(value)


def _point_json(point: ControlPoint) -> dict:
    return {"name": point.name, "px": point.px, "py": point.py,
            "lat": point.lat, "lon": point.lon, "elev_m": point.elev_m,
            "enabled": point.enabled, "note": point.note,
            "kind": point.kind, "time": point.time, "delay_s": point.delay_s,
            "ve": point.ve, "vn": point.vn, "vu": point.vu}


def _model_summary(model: CameraModel) -> dict:
    lat, lon = model.camera_latlon
    return {
        "yaw_deg": round(model.yaw_deg, 4),
        "pitch_deg": round(model.pitch_deg, 4),
        "roll_deg": round(model.roll_deg, 4),
        "focal_px": round(model.focal_px, 2),
        "hfov_deg": round(model.hfov_deg, 3),
        "vfov_deg": round(model.vfov_deg, 3),
        "k1": round(model.k1, 6), "k2": round(model.k2, 6),
        "height_m": round(model.height_m, 3),
        "offset_e_m": round(model.offset_e_m, 2),
        "offset_n_m": round(model.offset_n_m, 2),
        "lat": round(lat, 8), "lon": round(lon, 8),
        "width": model.width, "height": model.height,
        "rms_px": model.meta.get("rms_px"),
    }


def _write_drift_patches(state: WebState, point_set: ControlPointSet) -> None:
    from ..drift import save_reference_patches
    from ..ingest.grab import load_frame

    if not state.frame_path.is_file():
        return
    try:
        image = load_frame(state.frame_path)
        save_reference_patches(
            image, [(p.name, p.px, p.py) for p in point_set.active],
            state.cfg.path("drift.patches_dir", "./state/drift_patches"),
            half_size=int(state.cfg.get("drift.patch_half_size", 32)))
    except Exception as exc:
        log.warning("could not write drift patches", extra={"error": str(exc)})


def serve(cfg, hub: Optional[FrameHub] = None, pipeline=None,
          host: Optional[str] = None, port: Optional[int] = None,
          background: bool = False):
    """Start the web server. Returns the thread when ``background`` is set."""
    state = WebState(cfg, hub=hub, pipeline=pipeline)
    app = create_app(state)

    host = host or str(cfg.get("web.host", "127.0.0.1"))
    port = int(port or cfg.get("web.port", 8090))

    # Each MJPEG viewer holds a thread for as long as it watches, so the pool
    # has to be comfortably larger than the handful of request threads a
    # default install would give us.
    threads = int(cfg.get("web.threads", 16))

    def run() -> None:
        try:
            from waitress import serve as waitress_serve
        except ImportError:
            log.warning("waitress not installed; falling back to the Flask "
                        "development server (fine for local use, but it is "
                        "single-process and not meant to be left running)")
            app.run(host=host, port=port, threaded=True, debug=False,
                    use_reloader=False)
            return
        waitress_serve(app, host=host, port=port, threads=threads,
                       # A monitor tab can sit open for days; do not reap it.
                       channel_timeout=86400,
                       ident="spotter", clear_untrusted_proxy_headers=True)

    log.info("web UI listening", extra={
        "url": f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}/",
        "threads": threads,
        "live": pipeline is not None})

    if background:
        thread = threading.Thread(target=run, name="web-ui", daemon=True)
        thread.start()
        return thread
    run()
    return None
