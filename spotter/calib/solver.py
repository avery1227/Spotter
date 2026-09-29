"""Solve the camera model from hand-clicked control points.

Free parameters: yaw, pitch, roll, focal length, radial distortion k1/k2, and a
camera position offset (East, North, height). Position is held near the surveyed
value by soft priors rather than hard bounds, so it can absorb a sloppy GPS fix
without wandering across the county.

Getting a good starting point matters more than the optimiser does. A cold
``least_squares`` from an arbitrary yaw will happily settle into a mirror-image
or wildly-wrong-focal-length local minimum. So we bootstrap analytically:

1. Scan yaw over the full circle. At each candidate yaw the best focal length
   has a closed form (a one-parameter regression of ``px - cx`` on
   ``tan(bearing - yaw)``), so the scan is cheap.
2. Derive pitch from the vertical residual at that focal length.
3. Run the full non-linear solve from the best few seeds and keep the winner.

Besides points, two other kinds of evidence can go into the same fit:

* **Traced lines** (:mod:`.lines`). Each contributes the distance from its
  frame tracing to its map tracing projected into the frame, so it constrains
  the camera across the line but not along it. A line whose height is uncertain
  gets its own height parameter, held near the given value by a prior.
* **Aircraft points**: a plane clicked on a frozen frame, positioned from
  ADS-B at that frame's time. They carry a velocity, and one shared parameter
  ``dt`` lets every aircraft slide along its track by ``v * dt``. A pointing
  error moves all of them the same way on screen; a timing error moves each
  along its own heading. Two aircraft on different headings separate the two,
  and ``dt`` is then how far ``stream.encoder_delay_s`` is out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from ..geodesy import bearing_deg, elevation_deg, refraction_lift_m
from ..logging_setup import get_logger
from ..util import angle_diff
from .lines import LineSet
from .model import CameraModel
from .points import KIND_AIRCRAFT, ControlPointSet

log = get_logger(__name__)

#: Order of the full parameter vector. Locked parameters are dropped from the
#: vector handed to the optimiser but keep their initial value.
PARAM_NAMES = ("yaw", "pitch", "roll", "focal", "k1", "k2", "off_e", "off_n", "height")

#: Penalty (pixels) applied per point that projects behind the camera. Large
#: enough to dominate, finite so the optimiser can still find its way out.
BEHIND_PENALTY_PX = 5000.0

#: Spacing of the samples taken along a traced frame line, pixels.
LINE_SAMPLE_SPACING_PX = 15.0
#: Map lines are densified to at most this spacing before projection, metres,
#: so a straight map segment still bends correctly under lens distortion.
LINE_DENSIFY_M = 2.0
#: Limits on that densification, per map segment.
LINE_MIN_SUBDIV, LINE_MAX_SUBDIV = 2, 24


@dataclass
class PointError:
    name: str
    px: float
    py: float
    reproj_px: float
    reproj_py: float
    error_px: float
    #: This point's error measured against a fit that excluded it. High values
    #: mean the point disagrees with what every other point implies.
    loo_error_px: Optional[float] = None
    loo_delta_px: Optional[float] = None
    #: RMS of all the *other* points under that same reduced fit. If this is far
    #: below the full-fit RMS, this point was the one dragging the solve around.
    loo_rest_rms_px: Optional[float] = None
    #: How much the other points improve when this one is dropped.
    rms_improvement_px: Optional[float] = None
    in_front: bool = True

    def as_dict(self) -> dict:
        def r(value, digits=2):
            return round(value, digits) if value is not None else None

        return {
            "name": self.name,
            "clicked": [round(self.px, 1), round(self.py, 1)],
            "reprojected": [round(self.reproj_px, 1), round(self.reproj_py, 1)],
            "error_px": r(self.error_px),
            "loo_error_px": r(self.loo_error_px),
            "loo_delta_px": r(self.loo_delta_px),
            "loo_rest_rms_px": r(self.loo_rest_rms_px),
            "rms_improvement_px": r(self.rms_improvement_px),
            "in_front": self.in_front,
        }


@dataclass
class SolveResult:
    model: CameraModel
    errors: list[PointError] = field(default_factory=list)
    rms_px: float = 0.0
    max_px: float = 0.0
    warnings: list[str] = field(default_factory=list)
    n_points: int = 0
    converged: bool = False
    cost: float = 0.0
    seed_used: dict = field(default_factory=dict)
    #: Per traced line: fit quality and, where it was free, the solved height.
    line_errors: list[dict] = field(default_factory=list)
    n_lines: int = 0
    #: Seconds by which frames were really later than the overlay assumed;
    #: None unless aircraft points let it be measured.
    timing_offset_s: Optional[float] = None
    #: The ``encoder_delay_s`` in force when the aircraft points were taken.
    aircraft_delay_s: Optional[float] = None
    #: The raw solved parameter dict, for warm-starting refits.
    params: dict = field(default_factory=dict)

    @property
    def suggested_encoder_delay_s(self) -> Optional[float]:
        if self.timing_offset_s is None or self.aircraft_delay_s is None:
            return None
        return self.aircraft_delay_s - self.timing_offset_s

    def summary(self) -> dict:
        return {
            "rms_px": round(self.rms_px, 3),
            "max_px": round(self.max_px, 3),
            "n_points": self.n_points,
            "converged": self.converged,
            "yaw_deg": round(self.model.yaw_deg, 4),
            "pitch_deg": round(self.model.pitch_deg, 4),
            "roll_deg": round(self.model.roll_deg, 4),
            "focal_px": round(self.model.focal_px, 2),
            "hfov_deg": round(self.model.hfov_deg, 3),
            "k1": round(self.model.k1, 6),
            "k2": round(self.model.k2, 6),
            "offset_e_m": round(self.model.offset_e_m, 2),
            "offset_n_m": round(self.model.offset_n_m, 2),
            "height_m": round(self.model.height_m, 2),
            "n_lines": self.n_lines,
            "timing_offset_s": (round(self.timing_offset_s, 2)
                                if self.timing_offset_s is not None else None),
            "suggested_encoder_delay_s": (
                round(self.suggested_encoder_delay_s, 1)
                if self.suggested_encoder_delay_s is not None else None),
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# The problem: everything the residual function needs
# ---------------------------------------------------------------------------

@dataclass
class _PreparedLine:
    name: str
    #: Densified map tracing, ENU metres at the line's nominal height.
    samples_enu: np.ndarray
    #: Samples along the frame tracing, pixels.
    image_samples: np.ndarray
    image_vertices: np.ndarray
    elev_m: float
    elev_sigma_m: float
    #: Scales this line's residuals so it weighs as ``line_weight_points``
    #: points however many samples it has.
    weight: float

    @property
    def fits_height(self) -> bool:
        return self.elev_sigma_m > 0


@dataclass
class _Problem:
    enu: np.ndarray
    pixels: np.ndarray
    velocities: np.ndarray
    names: list
    lines: list = field(default_factory=list)
    fit_timing: bool = False

    def subset(self, indices) -> "_Problem":
        idx = list(indices)
        return _Problem(enu=self.enu[idx], pixels=self.pixels[idx],
                        velocities=self.velocities[idx],
                        names=[self.names[i] for i in idx],
                        lines=self.lines, fit_timing=self.fit_timing)

    def extra_params(self) -> list[str]:
        names = [f"dz{i}" for i, line in enumerate(self.lines) if line.fits_height]
        if self.fit_timing:
            names.append("dt")
        return names


def _resample_polyline(vertices: np.ndarray, spacing: float) -> np.ndarray:
    """Evenly spaced samples along a polyline, vertices included."""
    out = [vertices[0]]
    for a, b in zip(vertices[:-1], vertices[1:]):
        length = float(np.hypot(*(b - a)))
        steps = max(1, int(np.ceil(length / spacing)))
        for k in range(1, steps + 1):
            out.append(a + (b - a) * (k / steps))
    return np.array(out, dtype=float)


def _densify_map(vertices_enu: np.ndarray) -> np.ndarray:
    out = [vertices_enu[0]]
    for a, b in zip(vertices_enu[:-1], vertices_enu[1:]):
        length = float(np.linalg.norm(b - a))
        steps = int(np.clip(np.ceil(length / LINE_DENSIFY_M),
                            LINE_MIN_SUBDIV, LINE_MAX_SUBDIV))
        for k in range(1, steps + 1):
            out.append(a + (b - a) * (k / steps))
    return np.array(out, dtype=float)


def prepare_lines(line_set: Optional[LineSet], ref_lat: float, ref_lon: float,
                  weight_points: float = 4.0) -> list[_PreparedLine]:
    prepared = []
    for line in (line_set.active if line_set is not None else []):
        image = np.array(line.image, dtype=float)
        samples = _resample_polyline(image, LINE_SAMPLE_SPACING_PX)
        prepared.append(_PreparedLine(
            name=line.name,
            samples_enu=_densify_map(line.map_enu(ref_lat, ref_lon)),
            image_samples=samples,
            image_vertices=image,
            elev_m=line.elev_m,
            elev_sigma_m=line.elev_sigma_m,
            weight=float(np.sqrt(2.0 * weight_points / len(samples))),
        ))
    return prepared


def _distance_to_polyline(queries: np.ndarray, polyline: np.ndarray,
                          valid_vertex: np.ndarray) -> np.ndarray:
    """Distance from each query point to the nearest valid polyline segment."""
    usable = valid_vertex[:-1] & valid_vertex[1:]
    if not np.any(usable):
        return np.full(len(queries), np.inf)
    a = polyline[:-1][usable]
    b = polyline[1:][usable]
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab) + 1e-12
    rel = queries[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("qsj,sj->qs", rel, ab) / denom, 0.0, 1.0)
    nearest = a[None, :, :] + t[..., None] * ab[None, :, :]
    return np.linalg.norm(queries[:, None, :] - nearest, axis=2).min(axis=1)


def _line_residual(model: CameraModel, line: _PreparedLine, dz: float) -> np.ndarray:
    samples = line.samples_enu
    if dz:
        samples = samples + np.array([0.0, 0.0, dz])
    uv, in_front, _ = model.project_enu(samples, refract=True)
    distance = _distance_to_polyline(line.image_samples, uv, in_front)
    # Nothing of the map line in view: a large, finite penalty with no useful
    # gradient, which the seeding is there to avoid in the first place.
    return np.where(np.isfinite(distance), distance, BEHIND_PENALTY_PX)


# ---------------------------------------------------------------------------
# Parameter packing
# ---------------------------------------------------------------------------

class _ParamSpec:
    """Maps between the full parameter dict and the optimiser's flat vector."""

    def __init__(self, initial: dict, locked: Sequence[str],
                 extras: Sequence[str] = (), extra_limits: Optional[dict] = None):
        self.initial = dict(initial)
        for name in extras:
            self.initial.setdefault(name, 0.0)
        self.locked = set(locked)
        self.free = ([name for name in PARAM_NAMES if name not in self.locked]
                     + [name for name in extras if name not in self.locked])
        self.extra_limits = dict(extra_limits or {})

    def pack(self, values: dict) -> np.ndarray:
        return np.array([values[name] for name in self.free], dtype=float)

    def unpack(self, vector: np.ndarray) -> dict:
        out = dict(self.initial)
        for name, value in zip(self.free, vector):
            out[name] = float(value)
        return out

    def bounds(self, width: int) -> tuple[np.ndarray, np.ndarray]:
        limits = {
            # Yaw is scanned globally during seeding, so a generous box is fine.
            "yaw": (-4 * np.pi, 4 * np.pi),
            "pitch": (np.radians(-89.0), np.radians(89.0)),
            "roll": (np.radians(-45.0), np.radians(45.0)),
            # Below ~0.15*W the projection is a fisheye we are not modelling;
            # above ~60*W the lens is longer than any practical webcam.
            "focal": (0.15 * width, 60.0 * width),
            "k1": (-2.0, 2.0),
            "k2": (-2.0, 2.0),
            "off_e": (-5000.0, 5000.0),
            "off_n": (-5000.0, 5000.0),
            "height": (0.0, 1000.0),
            **self.extra_limits,
        }
        low = np.array([limits[n][0] for n in self.free], dtype=float)
        high = np.array([limits[n][1] for n in self.free], dtype=float)
        return low, high


def _model_from_params(params: dict, base: CameraModel) -> CameraModel:
    model = CameraModel(
        ref_lat=base.ref_lat, ref_lon=base.ref_lon,
        width=base.width, height=base.height,
        yaw_deg=float(np.degrees(params["yaw"])),
        pitch_deg=float(np.degrees(params["pitch"])),
        roll_deg=float(np.degrees(params["roll"])),
        offset_e_m=params["off_e"], offset_n_m=params["off_n"],
        height_m=params["height"],
        focal_px=params["focal"], cx=base.cx, cy=base.cy,
        k1=params["k1"], k2=params["k2"],
        refraction_k=base.refraction_k,
    )
    return model


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def _seed_yaw_focal(enu: np.ndarray, pixels: np.ndarray, base: CameraModel,
                    yaw_step_deg: float = 1.0) -> list[tuple[float, float, float]]:
    """Scan yaw; at each yaw solve focal in closed form. Returns (cost, yaw, focal).

    At small pitch the horizontal projection reduces to
    ``px - cx = f * tan(bearing - yaw)``, which is linear in ``f``. So for each
    candidate yaw the optimal focal is just ``sum(t*X) / sum(t*t)``.
    """
    rel = enu - base.camera_enu
    horizontal = np.hypot(rel[:, 0], rel[:, 1])
    rel = rel.copy()
    rel[:, 2] += refraction_lift_m(horizontal, base.refraction_k)

    bearings = bearing_deg(rel)
    x_obs = pixels[:, 0] - base.cx

    candidates: list[tuple[float, float, float]] = []
    for yaw in np.arange(0.0, 360.0, yaw_step_deg):
        delta = np.array([angle_diff(b, yaw) for b in bearings])
        # Points more than 75 degrees off axis would need a fisheye; a real
        # frame's control points are all well inside that.
        if np.any(np.abs(delta) > 75.0):
            continue
        t = np.tan(np.radians(delta))
        denom = float(np.dot(t, t))
        if denom < 1e-12:
            continue
        focal = float(np.dot(t, x_obs) / denom)
        if not np.isfinite(focal) or focal <= 0.15 * base.width:
            continue
        residual = x_obs - focal * t
        candidates.append((float(np.dot(residual, residual)), float(yaw), focal))

    candidates.sort(key=lambda c: c[0])
    return candidates


def _seed_pitch(enu: np.ndarray, pixels: np.ndarray, base: CameraModel,
                focal: float) -> float:
    """Pitch that best centres the vertical residual, small-angle approximation."""
    rel = enu - base.camera_enu
    horizontal = np.hypot(rel[:, 0], rel[:, 1])
    rel = rel.copy()
    rel[:, 2] += refraction_lift_m(horizontal, base.refraction_k)
    elevations = np.radians(elevation_deg(rel))
    y_obs = pixels[:, 1] - base.cy
    # py - cy ~= f * (pitch - elevation)  =>  pitch ~= elevation + (py - cy)/f
    return float(np.mean(elevations + y_obs / focal))


# ---------------------------------------------------------------------------
# Residuals
# ---------------------------------------------------------------------------

def _residuals(vector: np.ndarray, spec: _ParamSpec, base: CameraModel,
               problem: _Problem, prior_weights: dict) -> np.ndarray:
    params = spec.unpack(vector)
    model = _model_from_params(params, base)

    parts = []
    if len(problem.enu):
        enu = problem.enu
        dt = params.get("dt", 0.0)
        if dt:
            enu = enu + problem.velocities * dt
        uv, in_front, depth = model.project_enu(enu, refract=True)

        residual = (uv - problem.pixels).ravel()
        if not np.all(in_front):
            # Push wrong-side points back through the image plane with a penalty
            # that grows the further behind they are, so there is a gradient to
            # follow rather than a flat wall.
            behind = ~in_front
            penalty = BEHIND_PENALTY_PX * (1.0 + np.abs(depth[behind]) / 1000.0)
            shaped = residual.reshape(-1, 2).copy()
            shaped[behind, 0] = penalty
            shaped[behind, 1] = penalty
            residual = shaped.ravel()
        parts.append(residual)

    for i, line in enumerate(problem.lines):
        parts.append(line.weight * _line_residual(model, line, params.get(f"dz{i}", 0.0)))

    priors = []
    if "off_e" not in spec.locked:
        priors.append(params["off_e"] / prior_weights["position_sigma_m"])
    if "off_n" not in spec.locked:
        priors.append(params["off_n"] / prior_weights["position_sigma_m"])
    if "height" not in spec.locked:
        priors.append((params["height"] - prior_weights["height_prior_m"])
                      / prior_weights["height_sigma_m"])
    for i, line in enumerate(problem.lines):
        if line.fits_height:
            priors.append(params[f"dz{i}"] / line.elev_sigma_m)
    if problem.fit_timing:
        priors.append(params["dt"] / prior_weights["timing_sigma_s"])
    if priors:
        parts.append(np.array(priors, dtype=float))
    return np.concatenate(parts) if parts else np.zeros(1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def solve_calibration(point_set: ControlPointSet, cfg, width: int, height: int,
                      run_loo: bool = True,
                      lines: Optional[LineSet] = None) -> SolveResult:
    """Fit a :class:`CameraModel` to ``point_set`` and any traced ``lines``."""
    base = CameraModel.from_config(cfg, width, height)
    active = point_set.active
    solver_cfg = cfg.sub("calibration.solver")

    prepared = prepare_lines(lines, base.ref_lat, base.ref_lon,
                             float(solver_cfg.get("line_weight_points", 4.0)))
    # A line pins roughly two degrees of freedom, like a point pins two pixels.
    evidence = len(active) + 2 * len(prepared)
    if evidence < 4:
        raise ValueError(
            f"need at least 4 control points (a traced line counts as 2) to "
            f"solve, got {len(active)} points and {len(prepared)} lines")

    aircraft = [p for p in active if p.kind == KIND_AIRCRAFT and p.velocity]
    fit_timing = bool(solver_cfg.get("fit_timing", True)) and len(aircraft) >= 2
    problem = _Problem(
        enu=point_set.enu(base.ref_lat, base.ref_lon),
        pixels=point_set.pixels(),
        velocities=point_set.velocities(),
        names=point_set.names(),
        lines=prepared,
        fit_timing=fit_timing,
    )

    locked: list[str] = []
    if solver_cfg.get("lock_height", False):
        locked.append("height")
    if solver_cfg.get("lock_position", False):
        locked += ["off_e", "off_n"]
    if solver_cfg.get("lock_roll", False):
        locked.append("roll")
    if solver_cfg.get("lock_distortion", False):
        locked += ["k1", "k2"]

    # Distortion is only identifiable with points spread across the frame; with
    # a handful of clustered points it will happily absorb pose error.
    if evidence < 6 and "k2" not in locked:
        locked.append("k2")
        log.info("locking k2: too few points to identify second-order distortion",
                 extra={"n_points": len(active), "n_lines": len(prepared)})

    prior_weights = {
        "position_sigma_m": float(solver_cfg.get("position_sigma_m", 75.0)),
        "height_sigma_m": float(solver_cfg.get("height_sigma_m", 5.0)),
        "height_prior_m": float(base.height_m),
        "timing_sigma_s": float(solver_cfg.get("timing_sigma_s", 20.0)),
    }

    result = _solve_once(problem, base, locked, prior_weights, solver_cfg)
    result.n_points = len(active)
    result.n_lines = len(prepared)
    if fit_timing:
        delays = {p.delay_s for p in aircraft if p.delay_s is not None}
        if len(delays) == 1:
            result.aircraft_delay_s = delays.pop()
        elif len(delays) > 1:
            result.warnings.append(
                "aircraft points were captured with different encoder_delay_s "
                f"settings ({', '.join(f'{d:g}' for d in sorted(delays))} s); the "
                "timing offset mixes them. Delete the old ones and capture again")

    if run_loo and len(active) >= 5:
        _leave_one_out(result, problem, base, locked, prior_weights, solver_cfg)

    result.warnings.extend(check_distribution(point_set, result, cfg, width, height,
                                              prepared))
    return result


def _solve_once(problem: _Problem, base: CameraModel, locked, prior_weights,
                solver_cfg, warm_start: Optional[dict] = None) -> SolveResult:
    focal_init = solver_cfg.get("focal_px_init")
    max_nfev = int(solver_cfg.get("max_nfev", 20000))
    loss = str(solver_cfg.get("loss", "soft_l1"))
    f_scale = float(solver_cfg.get("f_scale_px", 3.0))

    if warm_start is not None:
        # A refit of nearly the same problem: start where the full fit ended
        # instead of scanning for basins again.
        seeds = [{"yaw_deg": float(np.degrees(warm_start["yaw"])),
                  "focal_px": warm_start["focal"],
                  "pitch_rad": warm_start["pitch"], "params": warm_start}]
    else:
        seed_enu, seed_pixels = _seed_features(problem)
        seeds = _build_seeds(seed_enu, seed_pixels, base, focal_init)

    extras = problem.extra_params()
    extra_limits = {}
    for i, line in enumerate(problem.lines):
        if line.fits_height:
            span = max(3.0, 5.0 * line.elev_sigma_m)
            extra_limits[f"dz{i}"] = (-span, span)
    if problem.fit_timing:
        extra_limits["dt"] = (-60.0, 60.0)

    best: Optional[tuple[float, object, dict, _ParamSpec]] = None
    for seed in seeds:
        initial = {
            "yaw": np.radians(seed["yaw_deg"]),
            "pitch": seed["pitch_rad"],
            "roll": 0.0,
            "focal": seed["focal_px"],
            "k1": 0.0, "k2": 0.0,
            "off_e": 0.0, "off_n": 0.0,
            "height": base.height_m,
        }
        if "params" in seed:
            initial.update({k: v for k, v in seed["params"].items()
                            if k in PARAM_NAMES})
        spec = _ParamSpec(initial, locked, extras, extra_limits)
        if "params" in seed:
            spec.initial.update({k: v for k, v in seed["params"].items()
                                 if k in extras})
        low, high = spec.bounds(base.width)
        x0 = np.clip(spec.pack(spec.initial), low, high)

        try:
            fit = least_squares(
                _residuals, x0,
                bounds=(low, high),
                args=(spec, base, problem, prior_weights),
                loss=loss, f_scale=f_scale,
                max_nfev=max_nfev,
                xtol=1e-12, ftol=1e-12, gtol=1e-12,
            )
        except Exception as exc:  # pragma: no cover - optimiser blowup
            log.debug("seed failed", extra={"seed": seed, "error": str(exc)})
            continue

        if best is None or fit.cost < best[0]:
            best = (float(fit.cost), fit, seed, spec)

    if best is None:
        raise RuntimeError("calibration solver failed from every seed")

    cost, fit, seed, spec = best
    params = spec.unpack(fit.x)
    params["yaw"] = float(np.radians(np.degrees(params["yaw"]) % 360.0))
    model = _model_from_params(params, base)
    dt = params.get("dt", 0.0) if problem.fit_timing else 0.0

    errors = _point_errors(model, problem, dt)
    magnitudes = np.array([e.error_px for e in errors])

    line_errors = []
    for i, line in enumerate(problem.lines):
        dz = params.get(f"dz{i}", 0.0)
        distance = _line_residual(model, line, dz)
        line_errors.append({
            "name": line.name,
            "rms_px": round(float(np.sqrt(np.mean(distance ** 2))), 2),
            "max_px": round(float(distance.max()), 2),
            "elev_m": round(line.elev_m + dz, 2),
            "elev_fitted": line.fits_height,
        })

    return SolveResult(
        model=model,
        errors=errors,
        rms_px=float(np.sqrt(np.mean(magnitudes ** 2))) if len(magnitudes) else 0.0,
        max_px=float(magnitudes.max()) if len(magnitudes) else 0.0,
        converged=bool(fit.success),
        cost=cost,
        seed_used={"yaw_deg": round(seed["yaw_deg"], 2),
                   "focal_px": round(seed["focal_px"], 1)},
        line_errors=line_errors,
        timing_offset_s=float(dt) if problem.fit_timing else None,
        params=params,
    )


def _seed_features(problem: _Problem) -> tuple[np.ndarray, np.ndarray]:
    """Correspondences for the analytic seeding.

    Points are exact. Each line adds its middle as a rough pseudo-point: the
    map and frame tracings need not cover the same stretch, but their middles
    are close enough to pick the right yaw basin, which is all seeding needs.
    """
    enu = [problem.enu] if len(problem.enu) else []
    pixels = [problem.pixels] if len(problem.pixels) else []
    if len(problem.enu) < 3:
        for line in problem.lines:
            mid = len(line.samples_enu) // 2
            enu.append(line.samples_enu[mid:mid + 1])
            pixels.append(line.image_samples[len(line.image_samples) // 2:
                                             len(line.image_samples) // 2 + 1])
    if not enu:
        return np.empty((0, 3)), np.empty((0, 2))
    return np.concatenate(enu), np.concatenate(pixels)


def _build_seeds(enu, pixels, base: CameraModel, focal_init) -> list[dict]:
    """Assemble a shortlist of starting points for the non-linear solve."""
    seeds: list[dict] = []
    scanned = _seed_yaw_focal(enu, pixels, base)

    # Take the best few distinct yaw basins from the scan.
    taken_yaws: list[float] = []
    for cost, yaw, focal in scanned:
        if any(abs(angle_diff(yaw, prev)) < 10.0 for prev in taken_yaws):
            continue
        taken_yaws.append(yaw)
        seeds.append({"yaw_deg": yaw, "focal_px": focal,
                      "pitch_rad": _seed_pitch(enu, pixels, base, focal)})
        if len(taken_yaws) >= 4:
            break

    # Perturb the best seed's focal length: the closed-form estimate assumes no
    # distortion, which biases it when k1 is significant.
    if seeds:
        primary = seeds[0]
        for scale in (0.75, 1.35):
            focal = primary["focal_px"] * scale
            seeds.append({"yaw_deg": primary["yaw_deg"], "focal_px": focal,
                          "pitch_rad": _seed_pitch(enu, pixels, base, focal)})

    # An explicit config hint, and a plain 45-degree-FOV fallback.
    extra_focals = [f for f in (focal_init, base.width / (2 * np.tan(np.radians(22.5))))
                    if f]
    for focal in extra_focals:
        yaw = seeds[0]["yaw_deg"] if seeds else float(bearing_deg(
            (enu - base.camera_enu))[len(enu) // 2])
        seeds.append({"yaw_deg": yaw, "focal_px": float(focal),
                      "pitch_rad": _seed_pitch(enu, pixels, base, float(focal))})

    if not seeds:  # pathological input; let the optimiser try from nothing
        seeds.append({"yaw_deg": 0.0, "focal_px": base.width,
                      "pitch_rad": 0.0})
    return seeds


def _point_errors(model: CameraModel, problem: _Problem, dt: float = 0.0) -> list[PointError]:
    enu = problem.enu + problem.velocities * dt if dt else problem.enu
    if not len(enu):
        return []
    uv, in_front, _ = model.project_enu(enu, refract=True)
    pixels = problem.pixels
    out = []
    for i, name in enumerate(problem.names):
        error = float(np.hypot(*(uv[i] - pixels[i]))) if in_front[i] else float("inf")
        out.append(PointError(
            name=name, px=float(pixels[i, 0]), py=float(pixels[i, 1]),
            reproj_px=float(uv[i, 0]), reproj_py=float(uv[i, 1]),
            error_px=error, in_front=bool(in_front[i]),
        ))
    return out


def _leave_one_out(result: SolveResult, problem: _Problem, base: CameraModel,
                   locked, prior_weights, solver_cfg) -> None:
    """Refit without each point in turn and score it against the reduced fit.

    A point whose error jumps when it is excluded is one the solve was bending
    to accommodate -- usually a mis-clicked pixel or a wrong lat/lon. Lines
    stay in every refit; only points are left out.
    """
    n = len(problem.names)
    for i in range(n):
        reduced = problem.subset([j for j in range(n) if j != i])
        try:
            sub = _solve_once(reduced, base, locked, prior_weights, solver_cfg,
                              warm_start=result.params)
        except Exception as exc:  # pragma: no cover
            log.warning("leave-one-out fit failed", extra={
                "point": problem.names[i], "error": str(exc)})
            continue

        dt = sub.timing_offset_s or 0.0
        enu_i = problem.enu[i:i + 1] + problem.velocities[i:i + 1] * dt
        uv, in_front, _ = sub.model.project_enu(enu_i, refract=True)
        loo_error = (float(np.hypot(*(uv[0] - problem.pixels[i])))
                     if in_front[0] else float("inf"))

        entry = result.errors[i]
        entry.loo_error_px = loo_error
        entry.loo_delta_px = loo_error - entry.error_px
        entry.loo_rest_rms_px = sub.rms_px
        entry.rms_improvement_px = result.rms_px - sub.rms_px


def check_distribution(point_set: ControlPointSet, result: SolveResult,
                       cfg, width: int, height: int,
                       lines: Sequence[_PreparedLine] = ()) -> list[str]:
    """Warn about point geometry that makes a fit look better than it is."""
    warn_cfg = cfg.sub("calibration.solver.warn")
    warnings: list[str] = []
    # Where a traced line runs counts towards coverage just like a point.
    pixels = np.concatenate([point_set.pixels().reshape(-1, 2)]
                            + [line.image_vertices for line in lines])
    n = len(pixels)

    min_points = int(warn_cfg.get("min_points", 6))
    evidence = len(point_set.active) + 2 * len(lines)
    if evidence < min_points:
        warnings.append(
            f"only {len(point_set.active)} control points and {len(lines)} lines "
            f"({min_points}+ points recommended, a line counts as 2); the fit is "
            f"under-constrained and distortion terms are unreliable")

    # All on the horizon: pitch, height and focal length become degenerate,
    # because raising the camera and tilting it down look identical.
    band = float(warn_cfg.get("horizon_band_frac", 0.08)) * height
    if n:
        median_y = float(np.median(pixels[:, 1]))
        on_horizon = np.sum(np.abs(pixels[:, 1] - median_y) < band) / n
        if on_horizon > float(warn_cfg.get("max_horizon_fraction", 0.85)):
            warnings.append(
                f"{on_horizon:.0%} of points lie within {band:.0f}px of the same "
                f"height; add near-field points (a dock, a buoy, a rooftop) or "
                f"pitch/height/focal will trade off against each other")

    if n:
        spread_x = (pixels[:, 0].max() - pixels[:, 0].min()) / width
        spread_y = (pixels[:, 1].max() - pixels[:, 1].min()) / height
        if spread_x < float(warn_cfg.get("min_spread_x_frac", 0.35)):
            warnings.append(
                f"points span only {spread_x:.0%} of the frame width; yaw and "
                f"focal length are poorly separated")
        if spread_y < float(warn_cfg.get("min_spread_y_frac", 0.12)):
            warnings.append(
                f"points span only {spread_y:.0%} of the frame height")

        # Clustered on one side: distortion will be fit to one half of the lens.
        left = np.sum(pixels[:, 0] < width / 2)
        if n >= 4 and (left == 0 or left == n):
            warnings.append(
                "all points are on one side of the frame; radial distortion is "
                "extrapolated across the other half")

    max_rms = float(warn_cfg.get("max_rms_px", 6.0))
    if result.rms_px > max_rms:
        warnings.append(
            f"reprojection RMS {result.rms_px:.1f}px exceeds {max_rms:.0f}px; "
            f"check for a mistyped lat/lon or a mis-clicked pixel")

    # Outlier hunt. A point with a bad lat/lon does NOT show up as a large
    # "error grew when I left it out" delta -- its in-fit error is already large,
    # so the delta is near zero. What identifies it is that every *other* point
    # gets dramatically better once it is dropped.
    scored = [e for e in result.errors if e.rms_improvement_px is not None]
    if scored and result.rms_px > 1e-6:
        worst = max(scored, key=lambda e: e.rms_improvement_px)
        frac = worst.rms_improvement_px / result.rms_px
        min_frac = float(warn_cfg.get("outlier_rms_improvement_frac", 0.4))
        if frac > min_frac and worst.rms_improvement_px > 1.0:
            warnings.append(
                f"point '{worst.name}' looks wrong: dropping it takes the "
                f"reprojection RMS from {result.rms_px:.1f}px to "
                f"{worst.loo_rest_rms_px:.1f}px. Re-check its lat/lon and the "
                f"pixel you clicked, or set enabled=0 for it in points.csv")

    max_loo = float(warn_cfg.get("max_loo_delta_px", 12.0))
    for err in result.errors:
        if err.loo_error_px is not None and err.loo_error_px > max_loo:
            warnings.append(
                f"point '{err.name}' lands {err.loo_error_px:.1f}px away when the "
                f"fit is built without it; it disagrees with the other points")

    max_line = float(warn_cfg.get("max_line_rms_px", 6.0))
    for entry in result.line_errors:
        if entry["rms_px"] > max_line:
            warnings.append(
                f"line '{entry['name']}' is {entry['rms_px']:.1f}px off on average; "
                f"check that both tracings follow the same feature, and that the "
                f"map tracing covers everything traced in the frame")

    if not result.converged:
        warnings.append("optimiser did not report convergence")

    return warnings
