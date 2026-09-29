"""Calibration from traced lines and aircraft points, on synthetic scenes.

Each test builds a known camera, projects map features through it to make the
"traced" frame lines and "clicked" pixels, starts the solver from a deliberately
wrong prior, and checks it gets the true camera back.
"""

import math

import numpy as np
import pytest

from spotter.calib.lines import LineFeature, LineSet, load_lines, save_lines
from spotter.calib.model import CameraModel
from spotter.calib.points import (KIND_AIRCRAFT, ControlPoint, ControlPointSet,
                                  load_points, save_points)
from spotter.calib.solver import solve_calibration
from spotter.config import Config
from spotter.geodesy import enu_to_geodetic, geodetic_to_enu

REF_LAT, REF_LON = 41.2796, -72.4377


def true_camera() -> CameraModel:
    """Roughly the Waters Edge camera: looking SSW over the Sound, rolled a little."""
    return CameraModel(ref_lat=REF_LAT, ref_lon=REF_LON, width=1920, height=1080,
                       yaw_deg=198.0, pitch_deg=-6.5, roll_deg=-0.6,
                       offset_e_m=12.0, offset_n_m=-8.0, height_m=28.0,
                       focal_px=1920 / (2 * math.tan(math.radians(52.0 / 2))),
                       k1=-0.04, k2=0.0)


def latlon(e: float, n: float) -> tuple[float, float]:
    out = enu_to_geodetic(np.array([[e, n, 0.0]]), REF_LAT, REF_LON, 0.0)
    return float(out[0, 0]), float(out[0, 1])


def traced_line(model, name, enu_vertices, elev, told_elev=None, sigma=0.0,
                trim=0.15, noise_px=0.4, rng=None):
    """A line as a user would trace it: the map fully, the frame partially."""
    rng = rng or np.random.default_rng(0)
    enu_vertices = np.asarray(enu_vertices, dtype=float)
    # Densify in 3D, project, and keep the middle stretch as the frame tracing.
    dense = []
    for a, b in zip(enu_vertices[:-1], enu_vertices[1:]):
        for t in np.linspace(0, 1, 60, endpoint=False):
            dense.append(a + (b - a) * t)
    dense.append(enu_vertices[-1])
    dense = np.array(dense)
    dense[:, 2] = elev
    uv, front, _ = model.project_enu(dense, refract=True)
    inside = front & (uv[:, 0] > 0) & (uv[:, 0] < 1920) & (uv[:, 1] > 0) & (uv[:, 1] < 1080)
    uv = uv[inside]
    lo, hi = int(len(uv) * trim), int(len(uv) * (1 - trim))
    picks = np.linspace(lo, hi - 1, 7).astype(int)
    image = uv[picks] + rng.normal(0, noise_px, (len(picks), 2))
    return LineFeature(
        name=name,
        image=[tuple(p) for p in image],
        map=[latlon(e, n) for e, n, _ in enu_vertices],
        elev_m=elev if told_elev is None else told_elev,
        elev_sigma_m=sigma)


def polar(bearing_deg, distance_m, camera=None):
    """ENU of a spot at a bearing and distance from the true camera."""
    camera = camera or true_camera()
    b = math.radians(bearing_deg)
    return [camera.offset_e_m + distance_m * math.sin(b),
            camera.offset_n_m + distance_m * math.cos(b), 0.0]


def near_field_lines(model, rng=None, wrong_heights=False):
    """Seawall, waterline, a path and a property line, 80-160 m from the camera."""
    wall = [(224, 150), (207, 124), (191, 117), (174, 126)]
    lines = [
        # The retaining wall along the shore, top at 3.0 m.
        ("seawall", [polar(b, d) for b, d in wall],
         3.0, 3.8 if wrong_heights else 3.0, 1.0),
        # The waterline at its foot: sea level, known exactly.
        ("waterline", [polar(b, d + 9) for b, d in wall], 0.0, None, 0.0),
        # A path running out towards the water.
        ("path", [polar(201, 82), polar(202, 100), polar(203, 114)],
         2.5, 2.0 if wrong_heights else 2.5, 1.0),
        # A property divider.
        ("divider", [polar(184, 80), polar(185, 97), polar(186, 112)],
         2.8, 2.8, 1.0),
    ]
    out = []
    for name, vertices, elev, told, sigma in lines:
        out.append(traced_line(model, name, vertices, elev, told, sigma, rng=rng))
    return LineSet(lines=out)


def far_points(model):
    """Horton Point and two towers across the Sound, clicked exactly."""
    features = [("Horton Point", 41.08514, -72.44577, 31.0),
                ("Mattituck tower", 40.98641, -72.58515, 142.0),
                ("Manorville tower", 40.85506, -72.76916, 233.0)]
    points = []
    for name, lat, lon, elev in features:
        enu = geodetic_to_enu(lat, lon, elev, REF_LAT, REF_LON, 0.0)
        uv, front, _ = model.project_enu(enu, refract=True)
        assert front[0]
        points.append(ControlPoint(name=name, px=float(uv[0, 0]), py=float(uv[0, 1]),
                                   lat=lat, lon=lon, elev_m=elev))
    return points


def aircraft_points(model, dt_true=0.0, delay_s=12.0):
    """Planes at altitude on different headings, clicked where they really were.

    The recorded position is where ADS-B said they were at the frame time the
    overlay assumed; the click is where they were ``dt_true`` seconds later.
    """
    planes = [  # bearing, distance (m), altitude (m), velocity east, north, up (m/s)
        (195, 40000, 4900, 180.0, 90.0, 0.0),
        (210, 35000, 3500, -150.0, 140.0, 0.0),
        (185, 55000, 7300, 60.0, -210.0, 0.0),
        (200, 25000, 2000, -120.0, -60.0, -6.0),
    ]
    points = []
    for i, (bearing, distance, alt, ve, vn, vu) in enumerate(planes):
        e, n, _ = polar(bearing, distance, model)
        recorded = np.array([[e, n, alt]])
        actual = recorded + np.array([[ve, vn, vu]]) * dt_true
        uv, front, _ = model.project_enu(actual, refract=True)
        assert front[0] and 0 < uv[0, 0] < 1920 and 0 < uv[0, 1] < 1080, uv
        lat, lon, h = enu_to_geodetic(recorded, REF_LAT, REF_LON, 0.0)[0]
        points.append(ControlPoint(
            name=f"plane {i}", px=float(uv[0, 0]), py=float(uv[0, 1]),
            lat=float(lat), lon=float(lon), elev_m=float(h),
            kind=KIND_AIRCRAFT, time="2026-09-29T20:00:00+00:00",
            delay_s=delay_s, ve=ve, vn=vn, vu=vu))
    return points


def solver_config(**solver) -> Config:
    # A deliberately poor prior: 40 m off, and a height guess 6 m low.
    lat, lon = latlon(12.0 + 30.0, -8.0 - 25.0)
    return Config({
        "camera": {"lat": lat, "lon": lon, "height_m": 22.0},
        "calibration": {"solver": {"position_sigma_m": 75.0, "height_sigma_m": 15.0,
                                   **solver}},
    })


def assert_recovers(result, truth, angle_tol=0.08, fov_tol=0.3, height_tol=1.0):
    model = result.model
    assert model.yaw_deg == pytest.approx(truth.yaw_deg, abs=angle_tol)
    assert model.pitch_deg == pytest.approx(truth.pitch_deg, abs=angle_tol)
    assert model.roll_deg == pytest.approx(truth.roll_deg, abs=angle_tol)
    assert model.hfov_deg == pytest.approx(truth.hfov_deg, abs=fov_tol)
    assert model.height_m == pytest.approx(truth.height_m, abs=height_tol)


# ---------------------------------------------------------------------------

def test_lines_plus_far_lights_recover_the_camera():
    truth = true_camera()
    rng = np.random.default_rng(1)
    lines = near_field_lines(truth, rng)
    points = ControlPointSet(points=far_points(truth))

    result = solve_calibration(points, solver_config(), 1920, 1080,
                               run_loo=False, lines=lines)
    assert result.n_lines == 4
    assert_recovers(result, truth)
    assert all(entry["rms_px"] < 1.5 for entry in result.line_errors)
    # Where it matters: a plane at 16,000 ft lands where it really is.
    assert far_field_error_px(result, truth) < 4.0


def test_line_heights_are_fitted_when_given_loosely():
    truth = true_camera()
    lines = near_field_lines(truth, np.random.default_rng(2), wrong_heights=True)
    points = ControlPointSet(points=far_points(truth))

    result = solve_calibration(points, solver_config(), 1920, 1080,
                               run_loo=False, lines=lines)
    fitted = {e["name"]: e for e in result.line_errors}
    # Told 3.8 m and 2.0 m; really 3.0 m and 2.5 m.
    assert fitted["seawall"]["elev_fitted"]
    assert fitted["seawall"]["elev_m"] == pytest.approx(3.0, abs=0.35)
    assert fitted["path"]["elev_m"] == pytest.approx(2.5, abs=0.35)
    assert not fitted["waterline"]["elev_fitted"]
    assert_recovers(result, truth, angle_tol=0.12, fov_tol=0.5)


def far_field_error_px(result, truth) -> float:
    """How far off a plane at 16,000 ft, 30 km out, would be drawn."""
    plane = np.array([[-6000.0, -30000.0, 4900.0]])
    got, _, _ = result.model.project_enu(plane)
    want, _, _ = truth.project_enu(plane)
    return float(np.hypot(*(got[0] - want[0])))


def test_near_lines_fit_nearby_but_one_far_light_fixes_the_sky():
    """Why a far-shore light is worth tracing a dozen walls.

    Near-field lines pin the camera's position, height and roll, and fit
    themselves to a fraction of a pixel -- but they all sit in the lower half of
    the frame, so the scale above the horizon is extrapolated. One light across
    the Sound anchors it.
    """
    truth = true_camera()
    lines = near_field_lines(truth, np.random.default_rng(3))

    lines_only = solve_calibration(ControlPointSet(), solver_config(), 1920, 1080,
                                   run_loo=False, lines=lines)
    assert all(entry["rms_px"] < 1.0 for entry in lines_only.line_errors)
    assert lines_only.model.yaw_deg == pytest.approx(truth.yaw_deg, abs=0.3)
    assert far_field_error_px(lines_only, truth) > 8.0

    with_horton = solve_calibration(
        ControlPointSet(points=far_points(truth)[:1]), solver_config(), 1920, 1080,
        run_loo=False, lines=lines)
    assert far_field_error_px(with_horton, truth) < 4.0


def test_aircraft_points_measure_the_timing_error():
    truth = true_camera()
    dt_true = 2.5
    points = ControlPointSet(points=far_points(truth)
                             + aircraft_points(truth, dt_true=dt_true, delay_s=12.0))
    lines = near_field_lines(truth, np.random.default_rng(4))

    result = solve_calibration(points, solver_config(), 1920, 1080,
                               run_loo=False, lines=lines)
    assert result.timing_offset_s == pytest.approx(dt_true, abs=0.3)
    # Frames were 2.5 s later than assumed: the delay was 2.5 s too long.
    assert result.suggested_encoder_delay_s == pytest.approx(12.0 - dt_true, abs=0.3)
    assert_recovers(result, truth)
    assert max(e.error_px for e in result.errors) < 2.0


def test_ignoring_timing_would_bend_the_camera():
    """The same scene with timing fitting off: the fit is visibly worse."""
    truth = true_camera()
    points = ControlPointSet(points=far_points(truth)
                             + aircraft_points(truth, dt_true=2.5))
    lines = near_field_lines(truth, np.random.default_rng(4))
    fitted = solve_calibration(points, solver_config(), 1920, 1080,
                               run_loo=False, lines=lines)
    unfitted = solve_calibration(points, solver_config(fit_timing=False), 1920, 1080,
                                 run_loo=False, lines=lines)
    assert unfitted.timing_offset_s is None
    assert unfitted.rms_px > 5 * max(fitted.rms_px, 0.2)


def test_mixed_capture_delays_are_warned_about():
    truth = true_camera()
    planes = aircraft_points(truth)
    planes[0].delay_s = 8.0
    points = ControlPointSet(points=far_points(truth) + planes)
    result = solve_calibration(points, solver_config(), 1920, 1080, run_loo=False,
                               lines=near_field_lines(truth))
    assert any("different encoder_delay_s" in w for w in result.warnings)


def test_too_little_evidence_is_refused():
    truth = true_camera()
    one_line = LineSet(lines=near_field_lines(truth).lines[:1])
    with pytest.raises(ValueError, match="a traced line counts as 2"):
        solve_calibration(ControlPointSet(), solver_config(), 1920, 1080,
                          lines=one_line)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def test_lines_round_trip_and_keep_history(tmp_path):
    path = tmp_path / "lines.json"
    lines = near_field_lines(true_camera())
    save_lines(path, lines)
    loaded = load_lines(path)
    assert [line.name for line in loaded.lines] == [line.name for line in lines.lines]
    assert loaded.lines[0].map[0] == pytest.approx(lines.lines[0].map[0])
    assert loaded.lines[1].elev_sigma_m == 0.0

    save_lines(path, LineSet(lines=loaded.lines[:1]))
    assert list((tmp_path / "state" / "lines_history").glob("lines-*.json"))
    assert load_lines(tmp_path / "missing.json").lines == []


def test_bad_lines_are_rejected():
    with pytest.raises(ValueError, match="at least 2 points in the frame"):
        LineFeature.from_json({"name": "x", "image": [[1, 2]],
                               "map": [[41.0, -72.0], [41.1, -72.1]]})
    with pytest.raises(ValueError, match="out of range"):
        LineFeature.from_json({"name": "x", "image": [[1, 2], [3, 4]],
                               "map": [[141.0, -72.0], [41.1, -72.1]]})


def test_aircraft_fields_round_trip_and_old_files_still_load(tmp_path):
    path = tmp_path / "points.csv"
    planes = aircraft_points(true_camera())
    save_points(path, ControlPointSet(points=far_points(true_camera()) + planes))
    loaded = load_points(path)
    plane = loaded.points[-1]
    assert plane.kind == KIND_AIRCRAFT
    assert plane.velocity == pytest.approx(planes[-1].velocity)
    assert plane.delay_s == 12.0
    assert loaded.points[0].velocity is None

    old = tmp_path / "old.csv"
    old.write_text("name,px,py,lat,lon,elev_m\nrock,10,20,41.2,-72.4,0\n")
    point = load_points(old).points[0]
    assert point.kind == "landmark" and point.velocity is None
